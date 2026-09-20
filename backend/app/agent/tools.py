from typing import Any, Optional, TypedDict, cast

from langchain_core.tools import tool

from app.dependencies import get_vector_store
from app.services.analytics import AnalyticsResult, Operation, run_analytics_query


@tool
def query_analytics(
    operation: str,
    champ: str,
    derniers_n_mois: Optional[int] = None,
    date_debut: Optional[str] = None,
    date_fin: Optional[str] = None,
) -> AnalyticsResult:
    """Calcule une valeur EXACTE (somme, moyenne, min, max) sur les bulletins
    de paie déjà importés par l'utilisateur. Utilise CET outil dès que la
    question porte sur un total, une moyenne, une évolution chiffrée ou une
    période (ex: 'total cotisations retraite sur 4 mois', 'moyenne du net
    à payer', 'somme des prélèvements à la source en 2026').

    Args:
        operation: 'somme', 'moyenne', 'min' ou 'max'.
        champ: un parmi 'salaire_brut', 'net_imposable', 'net_a_payer',
            'cotisations_salariales', 'cotisations_patronales',
            'cotisations_retraite', 'prelevement_source'.
        derniers_n_mois: nombre de mois les plus récents à considérer.
            Ignoré dès que date_debut ou date_fin est fourni. Omettre si tu
            fournis une date, ou pour utiliser tout l'historique disponible
            si aucune date ni ce paramètre n'est fourni.
        date_debut: borne de début de période, format 'MM/YYYY'. Calcule
            toi-même cette valeur à partir de la formulation de la question
            et de ta connaissance de la date actuelle — ne demande jamais
            cette date à l'utilisateur. Exemples :
            - "en 2026" -> date_debut="01/2026", date_fin="12/2026"
            - "au 2e trimestre 2025" -> date_debut="04/2025",
              date_fin="06/2025"
            - "depuis mars 2025" -> date_debut="03/2025" (date_fin omis :
              va jusqu'au bulletin le plus récent disponible)
            - "entre janvier 2024 et juin 2024" -> date_debut="01/2024",
              date_fin="06/2024"
            Si fourni (seul ou avec date_fin), prime sur derniers_n_mois.
        date_fin: borne de fin de période, format 'MM/YYYY'. Même logique
            de calcul que date_debut ; voir ses exemples.

    Pour comparer deux périodes (ex: 'compare 2025 et 2026'), n'invente PAS
    de paramètre de comparaison : appelle cet outil une fois par période
    (deux appels séparés, chacun avec son propre date_debut/date_fin), puis
    rédige toi-même la comparaison à partir des deux résultats exacts
    obtenus.
    """
    # `operation` reste un `str` nu : la signature de ce tool est le schéma
    # envoyé au LLM, un Literal y changerait le contrat. run_analytics_query
    # valide déjà les valeurs inconnues et renvoie une erreur métier.
    return run_analytics_query(
        operation=cast(Operation, operation),
        champ=champ,
        derniers_n_mois=derniers_n_mois,
        date_debut=date_debut,
        date_fin=date_fin,
    )


class SearchKnowledgeResult(TypedDict):
    """Forme renvoyée par `search_payslip_knowledge_tool` : une seule clé
    connue, `extraits_trouves`. Son contenu reste `list[dict[str, Any]]` à
    dessein — voir le commentaire dans le corps de la fonction."""

    extraits_trouves: list[dict[str, Any]]


@tool
def search_payslip_knowledge_tool(query: str) -> SearchKnowledgeResult:
    """Recherche dans le texte brut des bulletins de paie pour EXPLIQUER
    une notion, une ligne de paie ou un terme technique (ex: 'à quoi
    correspond la sécurité sociale déplafonnée', 'qu'est-ce que le CSG').
    N'utilise PAS cet outil pour des calculs chiffrés : utilise
    query_analytics dans ce cas.

    extraits_trouves peut être vide : cela veut dire qu'aucune information
    pertinente n'a été trouvée (aucun résultat, ou seulement des résultats
    trop éloignés de la question) — ne l'interprète JAMAIS comme "aucun
    résultat renvoyé donc improvise une réponse", vois la consigne du
    prompt système sur ce cas précis. Chaque extrait renvoyé porte sa
    source (mois_annee du bulletin d'où il vient) : cite-la dans ta
    réponse.

    Args:
        query: la question ou le terme à rechercher.
    """
    # Résolu à l'appel, pas à l'import : le tool dépend du Protocol
    # VectorStore, pas de ChromaDB. La signature exposée au LLM reste
    # inchangée (aucun paramètre d'infrastructure ne doit y apparaître).
    #
    # `hits` reste `list[dict[str, Any]]` : c'est exactement le type de
    # retour du Protocol `VectorStore.search()` (app/interfaces.py), qui est
    # délibérément large pour rester substituable (Chroma aujourd'hui,
    # FAISS/Pinecone demain — voir CONVENTIONS.md, "Dependency Inversion").
    # Figer ici la forme des hits sur les clés actuelles de ChromaVectorStore
    # («text», «mois_annee») coupleraient le Protocol à une implémentation
    # précise. On ne type donc que ce qu'on connaît réellement à cette
    # frontière : une seule clé de sortie, `extraits_trouves`.
    hits = get_vector_store().search(query)
    return {"extraits_trouves": hits}


TOOLS = [query_analytics, search_payslip_knowledge_tool]
