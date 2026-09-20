"""
Functional tests of authentication and rate limiting through the real
`require_auth` and `RateLimit` dependencies (`auth_client` fixture).
Only the settings are substituted; the Anthropic LLM is never called.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from starlette.requests import Request

import app.agent.graph as graph_mod
from app.auth.security import AuthConfigError, create_access_token, get_auth_settings
from app.main import app


class _AlwaysAnswersLLM:
    """Replaces `graph._llm`: answers directly, never requests a tool."""

    def invoke(self, messages):
        return AIMessage(content="Réponse de test.")


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(graph_mod, "_llm", _AlwaysAnswersLLM())


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _upload(client, pdf: bytes, headers: dict[str, str] | None = None):
    return client.post(
        "/api/upload",
        files={"file": ("bulletin.pdf", pdf, "application/pdf")},
        headers=headers or {},
    )


def _chat(client, headers: dict[str, str] | None = None):
    return client.post("/api/chat", json={"message": "Bonjour"}, headers=headers or {})


# --- login ----------------------------------------------------------------


def test_login_returns_a_bearer_token_valid_24h(auth_client, credentials):
    response = auth_client.post("/api/auth/login", json=credentials)

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 24 * 3600
    assert body["access_token"].count(".") == 2  # header.payload.signature


@pytest.mark.parametrize(
    "override",
    [{"password": "wrong password"}, {"username": "someone-else"}],
    ids=["wrong-password", "wrong-username"],
)
def test_login_rejects_bad_credentials_with_the_same_message(auth_client, credentials, override):
    response = auth_client.post("/api/auth/login", json={**credentials, **override})

    assert response.status_code == 401
    assert response.json()["detail"] == "Identifiant ou mot de passe incorrect."


def test_login_rejects_a_malformed_body(auth_client):
    assert auth_client.post("/api/auth/login", json={"username": "admin"}).status_code == 422


def test_health_stays_public(auth_client):
    assert auth_client.get("/api/health").status_code == 200


def test_app_refuses_to_start_without_auth_configuration(monkeypatch):
    # `with TestClient(...)` runs the lifespan. The settings check comes
    # before init_db(), so this never touches the real database file.
    monkeypatch.delenv("ADMIN_PASSWORD_HASH", raising=False)
    get_auth_settings.cache_clear()
    try:
        with pytest.raises(AuthConfigError, match="ADMIN_PASSWORD_HASH"):
            with TestClient(app):
                pass
    finally:
        get_auth_settings.cache_clear()


# --- protected routes: 401 ------------------------------------------------

PROTECTED = ["upload", "payslips", "chat"]


def _call(name: str, client, pdf: bytes, headers: dict[str, str]):
    if name == "upload":
        return _upload(client, pdf, headers)
    if name == "payslips":
        return client.get("/api/payslips", headers=headers)
    return _chat(client, headers)


@pytest.mark.parametrize("endpoint", PROTECTED)
def test_protected_route_without_token_is_401(auth_client, sample_pdf_bytes, endpoint):
    response = _call(endpoint, auth_client, sample_pdf_bytes, {})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("endpoint", PROTECTED)
@pytest.mark.parametrize(
    "authorization",
    ["Bearer not-a-jwt", "Basic YWRtaW46cGFzc3dvcmQ=", "Bearer "],
    ids=["garbage-token", "wrong-scheme", "empty-token"],
)
def test_protected_route_with_invalid_credentials_is_401(
    auth_client, sample_pdf_bytes, endpoint, authorization
):
    response = _call(endpoint, auth_client, sample_pdf_bytes, {"Authorization": authorization})
    assert response.status_code == 401


@pytest.mark.parametrize("endpoint", PROTECTED)
def test_protected_route_with_expired_token_is_401(
    auth_client, auth_settings, sample_pdf_bytes, endpoint
):
    expired = create_access_token(
        auth_settings.username, auth_settings, now=datetime.now(UTC) - timedelta(hours=25)
    )
    response = _call(endpoint, auth_client, sample_pdf_bytes, _bearer(expired))
    assert response.status_code == 401


def test_token_signed_with_another_secret_is_401(auth_client, auth_settings):
    forged = create_access_token(
        auth_settings.username, replace(auth_settings, jwt_secret="attacker-" + "z" * 40)
    )
    assert auth_client.get("/api/payslips", headers=_bearer(forged)).status_code == 401


# --- protected routes: 200 ------------------------------------------------


def test_payslips_with_valid_token_is_200(auth_client, token):
    response = auth_client.get("/api/payslips", headers=_bearer(token))
    assert response.status_code == 200
    assert response.json() == []


def test_upload_with_valid_token_is_200(auth_client, token, sample_pdf_bytes):
    response = _upload(auth_client, sample_pdf_bytes, _bearer(token))
    assert response.status_code == 200
    assert response.json()["mois_annee"] == "03/2025"


def test_chat_with_valid_token_is_200(auth_client, token, fake_llm):
    response = _chat(auth_client, _bearer(token))
    assert response.status_code == 200
    # No tool call in this fixture's response, so no RAG source to cite.
    assert response.json() == {"reply": "Réponse de test.", "sources": None}


# --- rate limiting --------------------------------------------------------


@pytest.fixture
def limit_of_two(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_HOURS", "1")


def test_upload_is_rate_limited(auth_client, token, sample_pdf_bytes, limit_of_two):
    for _ in range(2):
        assert _upload(auth_client, sample_pdf_bytes, _bearer(token)).status_code == 200

    response = _upload(auth_client, sample_pdf_bytes, _bearer(token))

    assert response.status_code == 429
    assert 3500 < int(response.headers["Retry-After"]) <= 3600
    assert "2 requêtes par heure" in response.json()["detail"]


def test_chat_is_rate_limited(auth_client, token, fake_llm, limit_of_two):
    for _ in range(2):
        assert _chat(auth_client, _bearer(token)).status_code == 200
    assert _chat(auth_client, _bearer(token)).status_code == 429


def test_upload_and_chat_have_separate_budgets(
    auth_client, token, sample_pdf_bytes, fake_llm, limit_of_two
):
    for _ in range(2):
        _upload(auth_client, sample_pdf_bytes, _bearer(token))
    assert _upload(auth_client, sample_pdf_bytes, _bearer(token)).status_code == 429

    assert _chat(auth_client, _bearer(token)).status_code == 200


def test_payslips_listing_is_not_rate_limited(auth_client, token, limit_of_two):
    for _ in range(5):
        assert auth_client.get("/api/payslips", headers=_bearer(token)).status_code == 200


def test_unauthenticated_requests_do_not_consume_the_budget(
    auth_client, token, sample_pdf_bytes, monkeypatch
):
    # Relies on FastAPI running router-level dependencies (auth, from
    # include_router in main.py) before route-level ones (the limiter).
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "1")
    for _ in range(3):
        assert _upload(auth_client, sample_pdf_bytes).status_code == 401

    assert _upload(auth_client, sample_pdf_bytes, _bearer(token)).status_code == 200


# --- login rate limiting (per IP) ------------------------------------------


@pytest.fixture
def login_limit_of_two(monkeypatch):
    monkeypatch.setenv("LOGIN_RATE_LIMIT_PER_15MIN", "2")


def test_login_rate_limit_rejects_after_the_configured_number_of_attempts(
    auth_client, credentials, login_limit_of_two
):
    wrong = {**credentials, "password": "wrong"}
    for _ in range(2):
        assert auth_client.post("/api/auth/login", json=wrong).status_code == 401

    response = auth_client.post("/api/auth/login", json=wrong)

    assert response.status_code == 429
    assert 800 < int(response.headers["Retry-After"]) <= 900
    assert "2 tentatives par 15 min" in response.json()["detail"]


def test_login_rate_limit_also_applies_to_successful_attempts(
    auth_client, credentials, login_limit_of_two
):
    # A real attacker's requests aren't all failures from the server's point
    # of view (a lucky guess still counts), so the limiter must not special
    # -case 200s.
    for _ in range(2):
        assert auth_client.post("/api/auth/login", json=credentials).status_code == 200

    assert auth_client.post("/api/auth/login", json=credentials).status_code == 429


def test_login_rate_limit_is_independent_per_ip(auth_client, credentials, login_limit_of_two):
    # This installed version of Starlette's `TestClient` hardcodes
    # `request.client` to `("testclient", 50000)` with no way to vary it
    # per instance, so a real end-to-end HTTP call can't simulate a second
    # IP. `require_login_rate_limit` is called directly instead, with a
    # hand-built `Request` carrying a different `client` tuple — this still
    # exercises the real dependency function (including its
    # `request.client.host` extraction), not just `LoginRateLimit` in
    # isolation (already covered in tests/unit/test_login_rate_limit.py).
    from app.auth.login_rate_limit import require_login_rate_limit

    def request_from(ip: str) -> Request:
        return Request({"type": "http", "client": (ip, 12345), "headers": []})

    for _ in range(2):
        require_login_rate_limit(request_from("203.0.113.1"))
    with pytest.raises(HTTPException) as excinfo:
        require_login_rate_limit(request_from("203.0.113.1"))
    assert excinfo.value.status_code == 429

    # A different IP is unaffected by 203.0.113.1's exhausted budget — and
    # confirmed through the real endpoint too, not just the dependency.
    require_login_rate_limit(request_from("203.0.113.2"))
    assert auth_client.post("/api/auth/login", json=credentials).status_code == 200


def test_login_rate_limit_falls_back_to_a_shared_key_without_a_client(login_limit_of_two):
    """`request.client` is `None` for some ASGI transports (never uvicorn's
    real HTTP server) — must not crash the request."""
    from app.auth.login_rate_limit import login_rate_limit, require_login_rate_limit

    login_rate_limit.reset()
    no_client_request = Request({"type": "http", "client": None, "headers": []})

    require_login_rate_limit(no_client_request)
    require_login_rate_limit(no_client_request)
    with pytest.raises(HTTPException) as excinfo:
        require_login_rate_limit(no_client_request)
    assert excinfo.value.status_code == 429


def test_login_rate_limit_resets_after_the_window(
    auth_client, credentials, login_limit_of_two, monkeypatch
):
    from app.auth import login_rate_limit as login_rate_limit_mod

    fake_now = [1_000_000.0]
    monkeypatch.setattr(login_rate_limit_mod.login_rate_limit, "_clock", lambda: fake_now[0])

    wrong = {**credentials, "password": "wrong"}
    for _ in range(2):
        assert auth_client.post("/api/auth/login", json=wrong).status_code == 401
    assert auth_client.post("/api/auth/login", json=wrong).status_code == 429

    fake_now[0] += 900  # exactly one window later
    assert auth_client.post("/api/auth/login", json=wrong).status_code == 401


def test_login_rate_limit_response_is_readable_cross_origin(
    auth_client, credentials, login_limit_of_two
):
    # `access-control-expose-headers` is only sent on the actual response,
    # not the OPTIONS preflight — so this triggers a real 429 (with the
    # frontend's own Origin) rather than inspecting a preflight response.
    # `Retry-After` must be in CORS's `expose_headers` or the browser hides
    # it from JS — checked here rather than assumed, since it's shared
    # config in main.py that a future change could silently narrow.
    origin = {"Origin": "http://localhost:5173"}
    wrong = {**credentials, "password": "wrong"}
    for _ in range(2):
        auth_client.post("/api/auth/login", json=wrong, headers=origin)

    response = auth_client.post("/api/auth/login", json=wrong, headers=origin)

    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()
