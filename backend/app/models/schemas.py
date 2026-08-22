from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message to the chatbot")
    session_id: Optional[str] = Field(
        default="default",
        description="Identifier for the conversation, so replies stay in context.",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The chatbot's reply")
