from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import QDRANT_URL, QDRANT_API_KEY


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# 2. Connect to existing Qdrant collection
# --------------------------------------------------

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="payment_policy_local",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# --------------------------------------------------
# 3. Ask a question
# --------------------------------------------------

question = "What was Voltalia's EBITDA in H1 2026?"


# --------------------------------------------------
# 4. Retrieve the 5 most similar chunks
# --------------------------------------------------

results = vector_store.similarity_search(
    question,
    k=10,
)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("Question:")
print(question)

print("\nNumber of results:", len(results))


for i, result in enumerate(results):

    print("\n==============================")
    print("RESULT:", i + 1)
    print("==============================")

    print("SOURCE:", result.metadata.get("source"))
    print("SECTION:", result.metadata.get("section"))

    print("\nCONTENT:")
    print(result.page_content[:500])