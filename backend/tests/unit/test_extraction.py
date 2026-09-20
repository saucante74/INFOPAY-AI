"""
Tests unitaires du pipeline d'extraction (`app/services/extraction.py`).

`ChatAnthropic` est mocké à chaque test (patché dans le namespace du module,
là où il est importé) : aucun test de ce fichier ne peut atteindre l'API
Anthropic réelle, avec ou sans clé valide dans `.env`.
"""
from __future__ import annotations

import pytest

import app.services.extraction as extraction_mod
from app.models.payslip import PayslipExtraction
from app.services.extraction import (
    ClaudeExtractor,
    PayslipExtractionError,
    extract_text_from_pdf,
    get_default_extractor,
)


@pytest.fixture(autouse=True)
def _clear_extractor_cache():
    # `get_default_extractor` est @lru_cache(maxsize=1) : sans ce nettoyage,
    # un test pourrait recevoir l'instance (et le mock déjà consommé) d'un
    # test précédent.
    get_default_extractor.cache_clear()
    yield
    get_default_extractor.cache_clear()


# ---------------------------------------------------------------------------
# extract_text_from_pdf — pas de LLM impliqué, pdfplumber pur
# ---------------------------------------------------------------------------


def test_extract_text_from_pdf_returns_real_text(sample_pdf_bytes):
    text = extract_text_from_pdf(sample_pdf_bytes)
    assert "BULLETIN DE PAIE MARS 2025" in text


def test_extract_text_from_pdf_blank_page_returns_empty_string(blank_pdf_bytes):
    assert extract_text_from_pdf(blank_pdf_bytes) == ""


# ---------------------------------------------------------------------------
# ClaudeExtractor.extract() — ChatAnthropic mocké
# ---------------------------------------------------------------------------


def _mock_chat_anthropic(mocker, invoke_return):
    """Patch `ChatAnthropic` dans le namespace de extraction.py et câble la
    chaîne `ChatAnthropic(...).with_structured_output(...).invoke(...)`."""
    mock_cls = mocker.patch.object(extraction_mod, "ChatAnthropic")
    mock_structured = mock_cls.return_value.with_structured_output.return_value
    mock_structured.invoke.return_value = invoke_return
    return mock_cls, mock_structured


def test_extract_calls_llm_and_returns_the_structured_result(mocker, sample_payslip_extraction):
    mock_cls, mock_structured = _mock_chat_anthropic(mocker, sample_payslip_extraction)

    extractor = ClaudeExtractor()
    result = extractor.extract("BULLETIN DE PAIE MARS 2025 - Salaire brut 3000,00 EUR")

    assert result is sample_payslip_extraction
    mock_cls.return_value.with_structured_output.assert_called_once_with(PayslipExtraction)
    mock_structured.invoke.assert_called_once()
    prompt = mock_structured.invoke.call_args[0][0]
    assert "BULLETIN DE PAIE MARS 2025 - Salaire brut 3000,00 EUR" in prompt


def test_extract_raises_on_empty_text_without_calling_the_llm(mocker, sample_payslip_extraction):
    mock_cls, _ = _mock_chat_anthropic(mocker, sample_payslip_extraction)

    extractor = ClaudeExtractor()
    with pytest.raises(PayslipExtractionError, match="Impossible d'extraire"):
        extractor.extract("")

    mock_cls.assert_not_called()


def test_llm_client_is_built_lazily_on_first_extract(mocker, sample_payslip_extraction):
    mock_cls, _ = _mock_chat_anthropic(mocker, sample_payslip_extraction)

    extractor = ClaudeExtractor()
    mock_cls.assert_not_called()  # __init__ ne construit rien

    extractor.extract("texte du bulletin")
    mock_cls.assert_called_once()

    extractor.extract("un autre bulletin")
    mock_cls.assert_called_once()  # cached_property : pas reconstruit


def test_get_default_extractor_is_a_singleton():
    assert get_default_extractor() is get_default_extractor()
