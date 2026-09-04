from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.rag_service import vector_store


COLLECTION_NAME = "payment_policy_local"


def load_documents_from_qdrant():
    documents = []

    offset = None

    while True:
        points, offset = vector_store.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})

            if page_content:
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                    )
                )

        if offset is None:
            break

    return documents


documents = load_documents_from_qdrant()

print("BM25 documents loaded:", len(documents))


bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 5


def bm25_search(question: str):
    print("🔎 BM25 search started...")
    return bm25_retriever.invoke(question)