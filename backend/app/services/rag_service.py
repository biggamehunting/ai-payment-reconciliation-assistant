from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from app.config import QDRANT_URL, QDRANT_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
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


if __name__ == "__main__":
    results = retrieve_context(
        "What is the deadline for requesting a refund?"
    )

    for doc in results:
        print(doc.page_content)