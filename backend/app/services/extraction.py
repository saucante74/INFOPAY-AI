"""
Pipeline d'extraction :

    PDF (bytes) --pdfplumber--> texte brut --LLM structuré--> PayslipExtraction

On extrait via le LLM plutôt que par regex/positions fixes car les
bulletins de paie ont des mises en page variées selon les logiciels de
paie (Silae, ADP, PayFit, etc.). Le LLM lit le texte brut et le fait
correspondre au schéma Pydantic, quelle que soit la mise en page.

`ClaudeExtractor` est l'implémentation Claude du Protocol
`app.interfaces.Extractor`. Une implémentation alternative (autre
fournisseur de LLM, extracteur par regex pour les tests) n'a qu'à exposer
la même méthode `extract()`.
"""
import io
from functools import cached_property, lru_cache
from typing import Any, cast

import pdfplumber
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.models.payslip import PayslipExtraction

EXTRACTION_MODEL = "claude-sonnet-4-6"


class PayslipExtractionError(Exception):
    """Échec d'extraction imputable au document (texte introuvable, mise en
    page trop atypique pour que le LLM renseigne tous les champs...), par
    opposition à une erreur d'infrastructure (API Anthropic injoignable,
    etc.) qui doit continuer à se propager telle quelle.

    Type dédié plutôt qu'un `ValueError` générique : `upload.py` peut la
    catcher spécifiquement (avec `pydantic.ValidationError`, levée quand la
    sortie structurée du LLM ne respecte pas le schéma `PayslipExtraction`)
    pour afficher un message lisible côté utilisateur sans jamais lui
    exposer le détail technique brut."""

EXTRACTION_PROMPT = (
    "Tu es un extracteur de données de bulletins de paie français. "
    "Analyse le texte brut suivant, extrait uniquement du bulletin, et "
    "renvoie les champs demandés. Les montants sont en euros, utilise "
    "le point comme séparateur décimal. Si un champ correspond à "
    "plusieurs lignes du bulletin (ex: cotisations retraite = base + "
    "complémentaire), fais la somme.\n\n"
    "--- TEXTE DU BULLETIN ---\n{raw_text}\n--- FIN DU TEXTE ---"
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Texte brut d'un PDF. Volontairement hors du Protocol `Extractor` :
    c'est du parsing de fichier, pas de l'inférence LLM."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


class ClaudeExtractor:
    """Implémentation de `Extractor` basée sur Claude (`langchain-anthropic`).

    Le client LLM est construit à la première utilisation (`cached_property`)
    et non dans `__init__` : instancier l'extracteur reste gratuit, et rien
    ne touche à l'API Anthropic tant qu'aucun bulletin n'est envoyé.
    """

    def __init__(self, model: str = EXTRACTION_MODEL, temperature: float = 0) -> None:
        self._model = model
        self._temperature = temperature

    @cached_property
    def _structured_llm(self) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        llm = ChatAnthropic(model=self._model, temperature=self._temperature)
        return llm.with_structured_output(PayslipExtraction)

    def extract(self, raw_text: str) -> PayslipExtraction:
        if not raw_text:
            raise PayslipExtractionError("Impossible d'extraire du texte de ce PDF (scan image ?).")

        result = self._structured_llm.invoke(EXTRACTION_PROMPT.format(raw_text=raw_text))
        # with_structured_output() est typé `dict | BaseModel` ; il renvoie ici
        # le schéma passé en argument, donc toujours un PayslipExtraction.
        return cast(PayslipExtraction, result)


@lru_cache(maxsize=1)
def get_default_extractor() -> ClaudeExtractor:
    """Instance partagée par défaut. Câblée dans `app/dependencies.py`."""
    return ClaudeExtractor()
