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
from typing import Annotated, TypedDict, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.tools import TOOLS

_SEARCH_TOOL_NAME = "search_payslip_knowledge_tool"


class ChatSource(TypedDict):
    """Une source RAG citée dans une réponse du chat : le mois du bulletin
    d'où vient l'extrait, et l'extrait de texte lui-même. Affichage
    structuré côté frontend, indépendant de ce que le LLM choisit d'écrire
    dans le texte de sa réponse (qui reste par ailleurs instruit de citer
    sa source, voir SYSTEM_PROMPT — les deux ne sont pas redondants : l'un
    est fiable mécaniquement, l'autre ne l'est pas)."""

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

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Tu es l'assistant InfoPay AI, spécialisé dans l'analyse de bulletins "
        "de paie français. Tu as deux outils : query_analytics pour tout "
        "calcul chiffré exact (sommes, moyennes, périodes), et "
        "search_payslip_knowledge_tool pour expliquer une notion ou une "
        "ligne de paie. Ne calcule JAMAIS un total ou une moyenne toi-même : "
        "utilise systématiquement query_analytics pour cela. "
        "Quand search_payslip_knowledge_tool renvoie un extraits_trouves "
        "vide, cela signifie qu'aucune information pertinente n'a été "
        "trouvée dans les bulletins importés : dis-le honnêtement à "
        "l'utilisateur (par exemple « Je n'ai trouvé aucune information à "
        "ce sujet dans vos bulletins importés. »), n'essaie JAMAIS de "
        "construire une explication à partir d'un extrait non pertinent. "
        "Quand search_payslip_knowledge_tool renvoie des extraits, cite "
        "systématiquement leur source dans ta réponse en utilisant le champ "
        "mois_annee de chaque extrait (par exemple « D'après votre bulletin "
        "de mars 2026 : ... »). Réponds toujours en français, de façon "
        "claire et concise."
    )
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


_llm = ChatAnthropic(model=CHAT_MODEL, temperature=0).bind_tools(TOOLS)


def _call_model(state: AgentState) -> AgentState:
    messages = [SYSTEM_PROMPT, *state["messages"]]
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

    messages = (history or []) + [HumanMessage(content=user_message)]
    result = agent_graph.invoke({"messages": messages})
    final_message = result["messages"][-1]
    # BaseMessage.content est `str | list[...]` ; nos réponses finales, sans
    # tool_call, sont toujours du texte.
    reply = cast(str, final_message.content)
    return {"reply": reply, "sources": _extract_sources(result["messages"])}
