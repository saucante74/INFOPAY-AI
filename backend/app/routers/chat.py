from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.agent.graph import ChatSource, run_chat
from app.rate_limit import chat_rate_limit

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    # Rempli uniquement si search_payslip_knowledge_tool a été appelé et a
    # trouvé des extraits pertinents pendant la conversation (voir
    # run_chat()/_extract_sources()) — None sinon, jamais une liste vide.
    sources: list[ChatSource] | None = None


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(chat_rate_limit)])
def chat(request: ChatRequest) -> ChatResponse:
    result = run_chat(request.message)
    return ChatResponse(reply=result["reply"], sources=result["sources"])
