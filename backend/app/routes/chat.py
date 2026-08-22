from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import get_bot_reply

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    reply = get_bot_reply(request.message, session_id=request.session_id)
    return ChatResponse(reply=reply)
