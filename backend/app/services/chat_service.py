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
from langchain_community.tools import DuckDuckGoSearchRun
from app.config import GEMINI_MODEL, GOOGLE_API_KEY
from langchain_tavily import TavilySearch
from app.services.rag_service import retrieve_context


SYSTEM_PROMPT = (
    "You are a friendly, helpful chatbot. Keep answers concise and conversational."
)



#duckduckgo is free search engine, but it has a limit of 100 searches per day. Tavily is a paid search engine that allows more searches per day. You can choose either one based on your needs.
_search_tool = DuckDuckGoSearchRun()
#_search_tool = TavilySearch(
#    max_results=2
#)
_llm_with_tools = None

def _get_llm() -> ChatGoogleGenerativeAI:
    """Lazily create the LLM client so a missing API key doesn't crash imports."""
    global _llm
    global _llm_with_tools
    if _llm is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to backend/.env (see .env.example)."
            )
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        #tell llm it can use the search tool
        #_llm_with_tools = _llm.bind_tools([_search_tool])
    return _llm

_llm = _get_llm

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

    rag_results  = retrieve_context(text)

    context = "\n\n".join(
        doc.page_content for doc in rag_results
    )

    print("RAG CONTEXT:")
    print(context)


    rows = get_chat_history(session_id)
    history = []

    for row in rows[-20:]:
        if row["role"] == "user":
            history.append(HumanMessage(content=row["message"]))
        elif row["role"] == "assistant":
            history.append(AIMessage(content=row["message"]))

    messages = [
    SystemMessage(content=SYSTEM_PROMPT),

    SystemMessage(
        content=f"""
        Use the following retrieved information when it is relevant
        to the user's question.

        Retrieved information:
        {context}
        """
    ),

    *history,
    HumanMessage(content=text)
]

    try:
        save_chat_message(session_id, "user", text)
        llm = _get_llm()
        # _llm_with_tools = _get_llm_with_tools()

        #replace the llm.invoke with _llm_with_tools.invoke to enable tool usage
        #response = llm.invoke(messages)

        #call the llm with tools to get the response, which may include a tool call
        # response = _llm_with_tools.invoke(messages)

        response = llm.invoke(messages)
        print(response.tool_calls)
        print("inside")

        answer = run_search_agent(response,text
        )   

        print("\nFINAL ANSWER:")
        print(answer)

    

        reply = answer
        if not reply:
            reply = "Sorry, I didn't get a usable response from Gemini. Please try again."
        save_chat_message(session_id, "assistant", reply)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message either way
        reply = f"Sorry, I hit an error talking to Gemini: {exc}"
        return reply

   

    return reply

# def _get_llm_with_tools():
#     global _llm_with_tools

#     if _llm_with_tools is None:
#         llm = _get_llm()
#         _llm_with_tools = llm.bind_tools([_search_tool])

#     return _llm_with_tools

def run_search_agent(response, user_question):
   
    response = _llm_with_tools.invoke(user_question)

    
    
    #Gemini generates query: "latest major developments in Java in 2026", notice it is not what we inputed
    if response.tool_calls:


        tool_call = response.tool_calls[0]

        query = tool_call["args"]["query"]
        try:

            #now we actually call the tool with the query generated by Gemini
            search_result = _search_tool.invoke(query)

            #simulate network failure by calling a function that raises an exception to tesst nw failure handling
            #search_result = failing_search(query)

            

            #call gemini again with the duckduckgo search result to get the final answer
            final_response = _llm.invoke(
            f"""Answer the user's question using the following web search result.

            User question:
            {user_question}

            Web search result:
            {search_result}
            """
            )
            #return only the text content of the final response, not the entire response object
            return final_response.content[0]["text"]
            

        except Exception as e:
            print(f"Error occurred while searching: {e}")
            return "Search failed due to network error. Please rely on your internal knowledge if possible."
    else:
        #if gemini decided not to use the search tool, just return the text content of the response
        print("No search required")
        return response.content[0]["text"]


def failing_search(query):
    raise Exception("Simulated network error")
