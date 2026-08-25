"""
Core chatbot logic — powered by Google Gemini via LangChain.

Each session_id keeps its own short conversation history in memory (a simple
dict), so replies stay context-aware within a chat. History resets whenever
the server restarts; swap the in-memory dict for Redis/a database if you need
persistence across restarts.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.database.database import save_chat_message, get_chat_history

from app.config import GEMINI_MODEL, GOOGLE_API_KEY

SYSTEM_PROMPT = (
    "You are a friendly, helpful chatbot. Keep answers concise and conversational."
)


_llm = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Lazily create the LLM client so a missing API key doesn't crash imports."""
    global _llm
    if _llm is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to backend/.env (see .env.example)."
            )
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            # temperature=0.7,
        )
    return _llm


def _extract_text(content) -> str:
    """
    Normalize a LangChain message's `.content` into a plain string.

    Some models (e.g. newer Gemini versions) return content as a list of
    blocks like [{"type": "text", "text": "..."}] instead of a plain string.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Common shapes: {"type": "text", "text": "..."} or {"text": "..."}
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
        return "".join(parts).strip()

    return str(content)


def get_bot_reply(message: str, session_id: str = "default") -> str:
    text = message.strip()
    if not text:
        return "Say something and I'll respond!"

    rows = get_chat_history(session_id)
    history = []

    for row in rows[-20:]:
        if row["role"] == "user":
            history.append(HumanMessage(content=row["message"]))
        elif row["role"] == "assistant":
            history.append(AIMessage(content=row["message"]))

    messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    *history,
    HumanMessage(content=text)
]

    try:
        save_chat_message(session_id, "user", text)
        llm = _get_llm()
        response = llm.invoke(messages)
        reply = _extract_text(response.content)
        if not reply:
            reply = "Sorry, I didn't get a usable response from Gemini. Please try again."
        save_chat_message(session_id, "assistant", reply)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message either way
        reply = f"Sorry, I hit an error talking to Gemini: {exc}"
        return reply

   

    return reply
