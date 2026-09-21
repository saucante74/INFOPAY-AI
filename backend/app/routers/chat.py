from typing import Literal

from fastapi import APIRouter, Depends
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import BaseModel

from app.agent.graph import ChatSource, run_chat
from app.rate_limit import chat_rate_limit

router = APIRouter(prefix="/api", tags=["chat"])


class ChatHistoryMessage(BaseModel):
    """Un message déjà échangé dans la conversation, tel que le frontend
    le garde (voir ChatPanel.tsx's `messages` state) — rôle + contenu
    texte seulement. Pas de tool_calls/sources : rejouer le raisonnement
    interne d'un tour précédent n'apporte rien au LLM, seul le texte
    échangé compte pour comprendre une question de suivi."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    # Historique de la conversation avant ce message. Le frontend le garde
    # en mémoire (état React, voir ChatPanel.tsx) et le renvoie en entier à
    # chaque appel — pas de session côté serveur, cohérent avec le reste de
    # l'architecture (JWT stateless). Défaut `[]` : les appelants existants
    # (et les anciens clients qui n'envoient pas encore ce champ) continuent
    # de fonctionner comme une conversation sans historique.
    history: list[ChatHistoryMessage] = []


class ChatResponse(BaseModel):
    reply: str
    # Rempli uniquement si search_payslip_knowledge_tool a été appelé et a
    # trouvé des extraits pertinents pendant la conversation (voir
    # run_chat()/_extract_sources()) — None sinon, jamais une liste vide.
    sources: list[ChatSource] | None = None


def _to_langchain_messages(history: list[ChatHistoryMessage]) -> list[AnyMessage]:
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in history
    ]


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(chat_rate_limit)])
def chat(request: ChatRequest) -> ChatResponse:
    result = run_chat(request.message, history=_to_langchain_messages(request.history))
    return ChatResponse(reply=result["reply"], sources=result["sources"])
