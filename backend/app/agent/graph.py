"""
Agent hybride construit avec LangGraph.

Le graphe est volontairement écrit à la main (plutôt qu'avec
create_react_agent) pour que l'architecture soit explicite et
explicable :

        ┌────────────┐        tool_calls ?        ┌───────────┐
   ───▶ │   agent    │ ─────────────yes──────────▶ │   tools   │
        │ (Claude +  │                             │(ToolNode) │
        │ tool-call) │ ◀───────────────────────────┘           │
        └────────────┘                                          
              │ no tool_calls
              ▼
             END

- Le nœud "agent" appelle Claude avec les 2 tools déclarés dans tools.py.
  Claude décide, via tool-calling natif, s'il doit :
    a) appeler query_analytics (calcul Pandas exact)
    b) appeler search_payslip_knowledge_tool (recherche vectorielle)
    c) les deux
    d) répondre directement (pas de tool_call -> fin du graphe)
- Le nœud "tools" exécute les tools appelés et renvoie les résultats
  au modèle, qui reformule alors une réponse finale en langage naturel.
"""
import json
from datetime import date
from typing import Annotated, TypedDict, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.tools import TOOLS
from app.services.analytics import get_available_period

_SEARCH_TOOL_NAME = "search_payslip_knowledge_tool"


class ChatSource(TypedDict):
    """Une source RAG citée dans une réponse du chat : le mois du bulletin
    d'où vient l'extrait, et l'extrait de texte lui-même. Affichage
    structuré côté frontend, indépendant de ce que le LLM choisit d'écrire
    dans le texte de sa réponse (qui reste par ailleurs instruit de citer
    sa source, voir _build_system_prompt() — les deux ne sont pas
    redondants : l'un est fiable mécaniquement, l'autre ne l'est pas)."""

    mois_annee: str
    extrait: str


class ChatResult(TypedDict):
    """Forme renvoyée par run_chat() au routeur /api/chat."""

    reply: str
    # None quand search_payslip_knowledge_tool n'a pas été appelé pendant
    # la conversation, ou l'a été mais sans résultat pertinent
    # (extraits_trouves vide, cf. le garde-fou de vectorstore.py) — jamais
    # une liste vide : le routeur/le frontend n'ont besoin de distinguer
    # que "il y a des sources à afficher" de "il n'y en a pas".
    sources: list[ChatSource] | None


CHAT_MODEL = "claude-sonnet-4-6"

# Troncature simple par nombre de messages, pas par budget de tokens : ce
# projet reste un prototype avec des conversations courtes (quelques
# échanges par session de chat), donc borner le coût/la latence n'exige
# pas la complexité d'un vrai compteur de tokens (tokenizer, taille
# variable des tool_calls et de leurs résultats, etc.) — hors de
# proportion avec l'usage réel. 20 messages = 10 échanges user/assistant :
# largement assez pour qu'un suivi ("et pour février ?") reste compris,
# tout en bornant la taille envoyée à l'API Anthropic à chaque appel si
# une conversation s'éternise. Les N derniers plutôt que les N premiers :
# le contexte le plus récent est presque toujours le plus pertinent pour
# une question de suivi. À revoir si des conversations nettement plus
# longues deviennent courantes en usage réel.
MAX_HISTORY_MESSAGES = 20


def _today() -> date:
    """Enveloppe `date.today()` dans une fonction du module, pour pouvoir
    la monkeypatcher directement dans les tests (`graph_mod._today`) —
    `datetime.date.today` lui-même, type immuable implémenté en C, n'est
    pas patchable proprement avec `monkeypatch.setattr`."""
    return date.today()


def _build_system_prompt() -> SystemMessage:
    """Reconstruit le prompt système à CHAQUE appel du nœud "agent" (pas un
    `SystemMessage` figé une fois pour toutes au chargement du module) :
    la date du jour et la période des bulletins réellement importés
    changent avec le temps, et le LLM ne doit jamais les deviner depuis sa
    mémoire d'entraînement — c'est exactement le bug que ce mécanisme
    corrige (une question sans année précisée amenait le LLM à halluciner
    une année par défaut plausible mais fausse)."""
    today_str = _today().strftime("%d/%m/%Y")
    period = get_available_period()
    period_str = (
        "Aucun bulletin de paie n'a encore été importé."
        if period is None
        else (
            f"Les bulletins actuellement importés couvrent la période de "
            f"{period['premier_mois']} à {period['dernier_mois']} "
            f"({period['nombre_bulletins']} bulletin(s))."
        )
    )

    return SystemMessage(
        content=(
            "Tu es l'assistant InfoPay AI, spécialisé dans l'analyse de bulletins "
            "de paie français. "
            f"Nous sommes aujourd'hui le {today_str}. {period_str} "
            "Tu as deux outils : query_analytics pour tout calcul chiffré exact "
            "(sommes, moyennes, périodes), et search_payslip_knowledge_tool pour "
            "expliquer une notion ou une ligne de paie à partir du contenu réel "
            "des bulletins importés. Ne calcule JAMAIS un total ou une moyenne "
            "toi-même : utilise systématiquement query_analytics pour cela. "
            "N'invente JAMAIS d'année ou de date par défaut : si une question ne "
            "précise pas d'année (par exemple « les cotisations de février et "
            "mars »), base-toi sur la date du jour et sur la période des "
            "bulletins importés indiquées ci-dessus pour déterminer l'année la "
            "plus probable, ou pour demander une clarification qui porte sur "
            "les VRAIES années disponibles dans les bulletins importés — jamais "
            "sur une année supposée depuis ta mémoire d'entraînement. "
            "N'utilise PAS search_payslip_knowledge_tool pour une question de "
            "culture générale, une définition théorique ou une question de "
            "droit du travail sans lien avec le contenu réel d'un bulletin "
            "importé (par exemple « quelle différence entre salaire moyen et "
            "salaire médian ? ») : réponds directement avec tes connaissances "
            "générales, sans chercher dans les documents. Réserve ce tool aux "
            "questions qui portent explicitement sur le contenu d'un bulletin "
            "de l'utilisateur (une ligne, un montant, un terme qui y figure). "
            "Quand search_payslip_knowledge_tool renvoie un extraits_trouves "
            "vide, cela signifie qu'aucune information pertinente n'a été "
            "trouvée dans les bulletins importés : dis-le honnêtement à "
            "l'utilisateur (par exemple « Je n'ai trouvé aucune information à "
            "ce sujet dans vos bulletins importés. »), n'essaie JAMAIS de "
            "construire une explication à partir d'un extrait non pertinent. "
            "Quand search_payslip_knowledge_tool renvoie des extraits, cite "
            "systématiquement leur source dans ta réponse en utilisant le champ "
            "mois_annee de chaque extrait (par exemple « D'après votre bulletin "
            "de mars 2026 : ... »). "
            "Réponds toujours en texte brut, SANS AUCUNE syntaxe Markdown (pas "
            "de **gras**, pas de # titres, pas de listes à tirets ou "
            "numérotées, pas de tableaux) : l'interface affiche ta réponse "
            "telle quelle, sans interprétation du Markdown. Cette règle "
            "s'applique MÊME SI l'utilisateur demande explicitement une liste, "
            "des puces ou un tableau : reformule toujours sa demande en texte "
            "brut plutôt que de t'y conformer littéralement. Si tu dois "
            "énumérer plusieurs éléments (des cotisations, des étapes, une "
            "comparaison, des conseils), ne mets JAMAIS un élément par ligne "
            "précédé d'un tiret ou d'un numéro : rédige des phrases complètes, "
            "ou des paragraphes courts séparés par des sauts de ligne, chacun "
            "introduit par son libellé suivi de deux-points plutôt que d'un "
            "tiret ou d'un numéro (par exemple « Salaire brut : 3000 euros. » "
            "plutôt que « - Salaire brut : 3000 euros » ou « 1. Salaire brut : "
            "3000 euros »). Réponds toujours en français, de façon claire et "
            "concise."
        )
    )


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


_llm = ChatAnthropic(model=CHAT_MODEL, temperature=0).bind_tools(TOOLS)


def _call_model(state: AgentState) -> AgentState:
    messages = [_build_system_prompt(), *state["messages"]]
    response = _llm.invoke(messages)
    return {"messages": [cast(AnyMessage, response)]}


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


_tool_node = ToolNode(TOOLS)

_graph_builder = StateGraph(AgentState)
_graph_builder.add_node("agent", _call_model)
_graph_builder.add_node("tools", _tool_node)
_graph_builder.set_entry_point("agent")
_graph_builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
_graph_builder.add_edge("tools", "agent")

agent_graph = _graph_builder.compile()


def _extract_sources(messages: list[AnyMessage]) -> list[ChatSource] | None:
    """Rassemble les extraits RAG effectivement utilisés pendant la
    conversation, à partir des ToolMessage produits par ToolNode pour
    search_payslip_knowledge_tool — pas seulement le message final : le LLM
    peut appeler ce tool plusieurs fois (ou en combinaison avec
    query_analytics), et chaque appel produit son propre ToolMessage dans
    l'historique. Renvoie None si le tool n'a jamais été appelé, ou l'a été
    sans trouver de résultat pertinent (extraits_trouves vide)."""
    sources: list[ChatSource] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != _SEARCH_TOOL_NAME:
            continue
        # ToolMessage.content : ToolNode sérialise en JSON le TypedDict
        # SearchKnowledgeResult renvoyé par le tool (voir tools.py) — un
        # simple `str`, ce tool ne renvoyant jamais de contenu multimodal.
        payload = json.loads(cast(str, message.content))
        for hit in payload["extraits_trouves"]:
            sources.append({"mois_annee": hit["mois_annee"], "extrait": hit["text"]})
    return sources or None


def run_chat(user_message: str, history: list[AnyMessage] | None = None) -> ChatResult:
    """Point d'entrée appelé par le routeur FastAPI /api/chat."""
    from langchain_core.messages import HumanMessage

    truncated_history = (history or [])[-MAX_HISTORY_MESSAGES:]
    messages = truncated_history + [HumanMessage(content=user_message)]
    result = agent_graph.invoke({"messages": messages})
    final_message = result["messages"][-1]
    # BaseMessage.content est `str | list[...]` ; nos réponses finales, sans
    # tool_call, sont toujours du texte.
    reply = cast(str, final_message.content)
    # _extract_sources ne doit regarder que les messages produits PENDANT
    # ce tour (à partir de l'index len(messages), donc après le dernier
    # HumanMessage envoyé) — pas tout `result["messages"]`, qui contient
    # aussi l'historique rejoué. Sans cette coupure, une réponse RAG
    # produite lors d'un tour précédent réapparaîtrait comme "source" du
    # tour actuel à chaque appel suivant, tant qu'elle reste dans la
    # fenêtre de troncature.
    new_messages = result["messages"][len(messages) :]
    return {"reply": reply, "sources": _extract_sources(new_messages)}
