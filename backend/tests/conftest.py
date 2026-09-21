"""
Fixtures partagées par toute la suite (unit + functional).

Rien ici ne touche à `backend/data/` : le moteur SQL est un SQLite en
mémoire créé par test (`test_engine`), et `Extractor`/`VectorStore` sont
des doublures conformes aux Protocol de `app.interfaces`, injectées via
`app.dependency_overrides` — jamais les implémentations réelles
(`ChatAnthropic`, ChromaDB).
"""
from __future__ import annotations

from collections.abc import Iterator

from datetime import timedelta

import bcrypt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import app.services.analytics as analytics_mod
from app.auth.login_rate_limit import login_rate_limit
from app.auth.security import AuthSettings, get_auth_settings, require_auth
from app.db import get_session
from app.dependencies import get_extractor, get_vector_store
from app.interfaces import Extractor, VectorStore
from app.main import app
from app.models.payslip import PayslipExtraction
from app.rate_limit import chat_rate_limit, upload_rate_limit

# ---------------------------------------------------------------------------
# PDF fixtures
# ---------------------------------------------------------------------------
# Un PDF minimal construit à la main (pas de dépendance à reportlab) : un
# objet Catalog, une Page, une police Helvetica standard et un flux de
# contenu avec une seule instruction Tj. Suffisant pour que pdfplumber
# retrouve du texte réel, sans passer par un vrai fichier de bulletin.


def _build_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length %d>>stream\n" % len(content) + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1) + b"0000000000 65535 f \n"
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref,
    )
    return bytes(out)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Un PDF d'une page avec du texte réel et extractible."""
    return _build_pdf("BULLETIN DE PAIE MARS 2025 - Salaire brut 3000,00 EUR")


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """Un PDF d'une page sans aucun texte (simule un scan image) : la page
    existe mais son flux de contenu est vide, donc `extract_text()` renvoie
    None/"" pour cette page."""
    return _build_pdf("")


# ---------------------------------------------------------------------------
# Extraction fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_payslip_extraction() -> PayslipExtraction:
    return PayslipExtraction(
        mois_annee="03/2025",
        nom_entreprise="ACME SARL",
        salaire_brut=3000.0,
        net_imposable=2400.0,
        net_a_payer=2300.0,
        total_cotisations_salariales=600.0,
        total_cotisations_patronales=900.0,
        cotisations_retraite=350.0,
        prelevement_source=120.0,
    )


class FakeExtractor:
    """Doublure conforme au Protocol `Extractor` (app.interfaces).

    Configurable par test : `.result` pour le succès (déjà posé par la
    fixture), `.exception` pour simuler un échec d'extraction (mime le
    comportement réel de `ClaudeExtractor`, qui peut lever `ValueError` ou
    toute autre exception venant du LLM)."""

    def __init__(self, result: PayslipExtraction) -> None:
        self.result = result
        self.exception: Exception | None = None
        self.calls: list[str] = []

    def extract(self, raw_text: str) -> PayslipExtraction:
        self.calls.append(raw_text)
        if self.exception is not None:
            raise self.exception
        return self.result


@pytest.fixture
def fake_extractor(sample_payslip_extraction: PayslipExtraction) -> FakeExtractor:
    return FakeExtractor(result=sample_payslip_extraction)


# ---------------------------------------------------------------------------
# Vector store fixtures
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """Doublure conforme au Protocol `VectorStore` (app.interfaces).
    N'ouvre jamais de collection ChromaDB réelle."""

    def __init__(self) -> None:
        self.indexed: list[tuple[int, str]] = []
        self.deleted: list[int] = []
        self.hits: list[dict] = [
            {"text": "La CSG déductible est assise sur le salaire brut.", "mois_annee": "03/2025"}
        ]

    def index(self, payslip_id: int, mois_annee: str, raw_text: str) -> None:
        self.indexed.append((payslip_id, mois_annee))

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        return self.hits

    def delete(self, payslip_id: int) -> None:
        self.deleted.append(payslip_id)


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_engine():
    """SQLite en mémoire, une base fraîche par test. `StaticPool` : une
    connexion unique partagée, sinon SQLite en mémoire perdrait ses tables
    entre deux connexions du pool par défaut."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------


def _override_business_dependencies(
    test_engine,
    fake_extractor: Extractor,
    fake_vector_store: VectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _session_override() -> Iterator[Session]:
        with Session(test_engine) as session:
            yield session

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_extractor] = lambda: fake_extractor
    app.dependency_overrides[get_vector_store] = lambda: fake_vector_store
    # analytics.py reads via its own module-level `engine`, not through
    # `get_session` (see tests/unit/test_analytics.py) — the chat graph
    # now also queries it on every turn (the real available bulletin
    # period injected into the system prompt, see app/agent/graph.py), so
    # any fixture that can run a real chat turn must redirect it too,
    # never the real `backend/data/infopay.db`.
    monkeypatch.setattr(analytics_mod, "engine", test_engine)


@pytest.fixture
def _overridden_app(
    test_engine,
    fake_extractor: Extractor,
    fake_vector_store: VectorStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """The app with auth and rate limiting bypassed: business tests stay
    about business behaviour. Both are exercised for real by `auth_client`."""
    _override_business_dependencies(test_engine, fake_extractor, fake_vector_store, monkeypatch)
    app.dependency_overrides[require_auth] = lambda: TEST_USERNAME
    app.dependency_overrides[upload_rate_limit] = lambda: None
    app.dependency_overrides[chat_rate_limit] = lambda: None
    try:
        yield app
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client(_overridden_app) -> TestClient:
    # Volontairement PAS `with TestClient(app) as c:` : la forme
    # context-manager déclenche le lifespan/startup de FastAPI, qui appelle
    # `init_db()` -> `SQLModel.metadata.create_all(app.db.engine)` sur le
    # VRAI moteur (backend/data/infopay.db). L'instanciation simple ne
    # déclenche pas le startup, donc ne touche jamais le fichier réel.
    # `get_session` est de toute façon déjà substitué ci-dessus.
    return TestClient(_overridden_app)


@pytest.fixture
def client_no_raise(_overridden_app) -> TestClient:
    """Même client, mais sans re-lever les exceptions non gérées par une
    route (utile pour vérifier qu'un endpoint sans try/except renvoie bien
    une 500 plutôt que de planter le test)."""
    return TestClient(_overridden_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

TEST_USERNAME = "admin"
TEST_PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="session")
def credentials() -> dict[str, str]:
    """The account `auth_settings` accepts, as a login request body."""
    return {"username": TEST_USERNAME, "password": TEST_PASSWORD}


@pytest.fixture(scope="session")
def auth_settings() -> AuthSettings:
    # rounds=4, bcrypt's minimum: the hash is still real, just cheap to
    # verify, so the suite doesn't pay ~0.25 s per login test.
    return AuthSettings(
        username=TEST_USERNAME,
        password_hash=bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt(rounds=4)),
        jwt_secret="test-secret-" + "x" * 40,
        token_ttl=timedelta(hours=24),
    )


@pytest.fixture
def auth_client(
    test_engine,
    fake_extractor: Extractor,
    fake_vector_store: VectorStore,
    auth_settings: AuthSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Real `require_auth` and real rate limiters; only the settings (which
    would otherwise come from the environment) are substituted."""
    _override_business_dependencies(test_engine, fake_extractor, fake_vector_store, monkeypatch)
    app.dependency_overrides[get_auth_settings] = lambda: auth_settings
    # The limiters are module-level singletons: reset so no count leaks
    # between tests.
    upload_rate_limit.reset()
    chat_rate_limit.reset()
    login_rate_limit.reset()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        upload_rate_limit.reset()
        chat_rate_limit.reset()
        login_rate_limit.reset()


@pytest.fixture
def token(auth_client: TestClient, credentials: dict[str, str]) -> str:
    """A real, valid JWT for `auth_client`'s account -- shared across test
    modules that need to call a protected endpoint through `auth_client`
    (as opposed to `client`, which bypasses auth entirely)."""
    response = auth_client.post("/api/auth/login", json=credentials)
    assert response.status_code == 200
    return str(response.json()["access_token"])
