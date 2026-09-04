from app.services.rag_service import vector_store
from app.services.bm25_service import bm25_search

from langchain_core.tools import tool

from app.services.rag_service import vector_store
from app.services.bm25_service import bm25_search
from app.services.reranker_service import rerank_documents

@tool
def search_internal_knowledge(question: str) -> str:
    """
    Search only the internal documents provided to the application.

    Uses vector search + BM25 + Cohere reranking.

    Do NOT use this tool for current, public, or internet information.
    """
    try:

        if not question or not question.strip():
            return "No valid question was provided."

        results = hybrid_search(
            question,
            k=5,
            top_n=3,
        )

        if not results:
            return "No relevant internal information was found."

        for i, result in enumerate(results):

            print("\n==============================")
            print("RERANKED RESULT:", i + 1)
            print("==============================")

            print("SOURCE:", result.metadata.get("source"))
            print("SECTION:", result.metadata.get("section"))

            print("\nCONTENT:")
            print(result.page_content[:500])

        return "\n\n".join(
            doc.page_content
            for doc in results
        )

    except Exception as e:
        return f"An error occurred while searching internal knowledge: {str(e)}"


def hybrid_search(question: str, k: int = 5, top_n: int = 3):

    vector_results = vector_store.similarity_search(
        question,
        k=k,
    )

    bm25_results = bm25_search(question)

    combined = vector_results + bm25_results

    unique_results = []
    seen = set()

    for doc in combined:
        key = (
            doc.metadata.get("document_id"),
            doc.page_content,
        )

        if key not in seen:
            seen.add(key)
            unique_results.append(doc)

    reranked_results = rerank_documents(
        question,
        unique_results,
        top_n=top_n,
    )

    return reranked_results

@tool
def delete_payment(payment_id: str) -> str:
    """
    Delete a payment.

    This is a sensitive operation and requires human approval.
    """
    print(f"\n⚠️ APPROVAL REQUIRED: Delete payment {payment_id}?")

    approval = input("Approve? (yes/no): ").strip().lower()

    if approval != "yes":
        return f"Deletion of payment {payment_id} was rejected by the user."

    return f"Payment {payment_id} deleted successfully."



# def hybrid_search(question: str, k: int = 5):
#     # 1. Vector search
#     vector_results = vector_store.similarity_search(
#         question,
#         k=k,
#     )

#     # 2. BM25 search
#     bm25_results = bm25_search(question)

#     # 3. Combine results
#     combined = vector_results + bm25_results

#     # 4. Remove duplicate chunks
#     unique_results = []
#     seen = set()

#     for doc in combined:
#         key = (
#             doc.metadata.get("document_id"),
#             doc.page_content,
#         )

#         if key not in seen:
#             seen.add(key)
#             unique_results.append(doc)

#     return unique_results