from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from app.config import QDRANT_URL, QDRANT_API_KEY
from langchain_core.tools import tool

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="payment_policy",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def retrieve_context(question: str):
    
    results = vector_store.similarity_search(
        question,
        k=3
    )

    return results


@tool
def search_internal_knowledge(question: str) -> str:
    """
    Search only the internal documents provided to the application.

    Use this tool when the user asks about information contained
    in the company's internal documents.

    Do NOT use this tool for current, public, or internet information.
    """
    try:

        if not question or not question.strip():
            return "No valid question was provided."
        
        results = vector_store.similarity_search(question, k=3)

        if not results:
            return "No relevant internal information was found."

        for i, doc in enumerate(results, start=1):
            print(f"\n--- CHUNK {i} ---")
            print(doc.page_content)

        return "\n\n".join(
            doc.page_content for doc in results
        )
    except Exception as e:
        return f"An error occurred while searching internal knowledge: {str(e)}"






# if __name__ == "__main__":
#     results = retrieve_context(
#         "What is the deadline for requesting a refund?"
#     )

#     for doc in results:
#         print(doc.page_content)