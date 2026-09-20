"""
Tests fonctionnels de `/api/upload` et `/api/payslips` via `TestClient`.

`Extractor` et `VectorStore` sont remplacés par les doublures de
conftest.py (`fake_extractor`, `fake_vector_store`) via
`app.dependency_overrides` ; la base est un SQLite en mémoire
(`test_engine`). Rien ici n'écrit dans `backend/data/`.
"""
from __future__ import annotations

from pydantic import ValidationError

from app.models.payslip import PayslipExtraction
from app.routers.upload import EXTRACTION_ERROR_MESSAGE
from app.services.extraction import PayslipExtractionError


def test_upload_happy_path_persists_and_indexes(client, fake_extractor, fake_vector_store, sample_pdf_bytes):
    response = client.post(
        "/api/upload", files={"file": ("bulletin.pdf", sample_pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["mois_annee"] == "03/2025"
    assert body["net_a_payer"] == 2300.0
    assert body["filename"] == "bulletin.pdf"
    assert "BULLETIN DE PAIE MARS 2025" in body["raw_text"]

    # L'extracteur a bien reçu le texte extrait du PDF, pas un texte vide.
    assert fake_extractor.calls and "BULLETIN DE PAIE MARS 2025" in fake_extractor.calls[0]

    # L'indexation vectorielle a lieu après le commit, avec l'id réel en base.
    assert fake_vector_store.indexed == [(1, "03/2025")]


def test_upload_rejects_non_pdf_content_type(client, fake_vector_store):
    response = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
    assert fake_vector_store.indexed == []  # jamais atteint


def test_upload_returns_422_with_readable_message_on_payslip_extraction_error(
    client, fake_extractor, fake_vector_store, sample_pdf_bytes
):
    """Cas `PayslipExtractionError` (ex : texte introuvable dans le PDF)."""
    fake_extractor.exception = PayslipExtractionError(
        "Impossible d'extraire du texte de ce PDF (scan image ?)."
    )

    response = client.post(
        "/api/upload", files={"file": ("bulletin.pdf", sample_pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    # Le message affiché à l'utilisateur est le texte lisible, jamais le
    # détail technique brut de l'exception.
    assert detail == EXTRACTION_ERROR_MESSAGE
    assert "scan image" not in detail
    # Rien n'a été indexé puisque la ligne n'a jamais été committée.
    assert fake_vector_store.indexed == []


def test_upload_returns_422_with_readable_message_on_pydantic_validation_error(
    client, fake_extractor, fake_vector_store, sample_pdf_bytes
):
    """Cas réel qui a motivé ce fix : le LLM renvoie une sortie structurée
    dont un champ ne respecte pas le schéma `PayslipExtraction` (ex : une
    chaîne là où un nombre est attendu, sur un bulletin à la mise en page
    atypique). Ça lève `pydantic.ValidationError`, dont le message technique
    brut ("Input should be a valid number, unable to parse string as a
    number [...] https://errors.pydantic.dev/...") ne doit jamais atteindre
    le frontend tel quel."""
    try:
        PayslipExtraction(
            mois_annee="06/2022",
            nom_entreprise="BatiRenov",
            salaire_brut=3000.0,
            net_imposable=2400.0,
            net_a_payer=2300.0,
            total_cotisations_salariales=600.0,
            total_cotisations_patronales="non numérique",  # type: ignore[arg-type]
            cotisations_retraite=350.0,
            prelevement_source=120.0,
        )
    except ValidationError as exc:
        fake_extractor.exception = exc
    else:  # pragma: no cover - garde-fou si le schéma change un jour
        raise AssertionError("PayslipExtraction aurait dû lever ValidationError ici")

    response = client.post(
        "/api/upload", files={"file": ("bulletin.pdf", sample_pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail == EXTRACTION_ERROR_MESSAGE
    # Le détail Pydantic brut (nom de champ interne, lien errors.pydantic.dev...)
    # ne doit plus jamais fuiter jusqu'à la réponse HTTP.
    assert "total_cotisations_patronales" not in detail
    assert "pydantic.dev" not in detail
    # Rien n'a été indexé puisque la ligne n'a jamais été committée.
    assert fake_vector_store.indexed == []


def test_list_payslips_empty_then_populated(client, sample_pdf_bytes):
    assert client.get("/api/payslips").json() == []

    client.post("/api/upload", files={"file": ("bulletin.pdf", sample_pdf_bytes, "application/pdf")})

    payslips = client.get("/api/payslips").json()
    assert len(payslips) == 1
    assert payslips[0]["mois_annee"] == "03/2025"


def test_delete_payslip_removes_it_from_the_db_and_the_vector_store(
    client, fake_vector_store, sample_pdf_bytes
):
    upload_response = client.post(
        "/api/upload", files={"file": ("bulletin.pdf", sample_pdf_bytes, "application/pdf")}
    )
    payslip_id = upload_response.json()["id"]
    assert client.get("/api/payslips").json() != []

    response = client.delete(f"/api/payslips/{payslip_id}")

    assert response.status_code == 204
    assert response.content == b""
    assert client.get("/api/payslips").json() == []
    # The vector store was asked to remove this exact payslip's chunks.
    assert fake_vector_store.deleted == [payslip_id]


def test_delete_payslip_404_on_an_id_that_does_not_exist(client, fake_vector_store):
    response = client.delete("/api/payslips/999")

    assert response.status_code == 404
    assert "999" in response.json()["detail"]
    # Never reached the vector store for an id that was never persisted.
    assert fake_vector_store.deleted == []


def test_delete_payslip_requires_authentication(auth_client, token, sample_pdf_bytes):
    upload_response = auth_client.post(
        "/api/upload",
        files={"file": ("bulletin.pdf", sample_pdf_bytes, "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )
    payslip_id = upload_response.json()["id"]

    unauthenticated = auth_client.delete(f"/api/payslips/{payslip_id}")
    assert unauthenticated.status_code == 401

    authenticated = auth_client.delete(
        f"/api/payslips/{payslip_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert authenticated.status_code == 204
