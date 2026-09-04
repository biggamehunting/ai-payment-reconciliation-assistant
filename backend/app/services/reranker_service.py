import os

import cohere
from dotenv import load_dotenv


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(COHERE_API_KEY)


def rerank_documents(question: str, documents, top_n: int = 3):
    texts = [doc.page_content for doc in documents]
    print("🔄 Cohere reranking started...")
    response = co.rerank(
        model="rerank-v4.0-fast",
        query=question,
        documents=texts,
        top_n=top_n,
    )
    print("✅ Cohere reranking completed.")
    return [
        documents[result.index]
        for result in response.results
    ]

# def rerank_documents(question: str, documents, top_n: int = 3):
#     texts = [
#         doc.page_content
#         for doc in documents
#     ]

#     response = co.rerank(
#         model="rerank-v4.0-fast",
#         query=question,
#         documents=texts,
#         top_n=top_n,
#     )

#     reranked_documents = []

#     for result in response.results:
#         document = documents[result.index]

#         reranked_documents.append(
#             (
#                 document,
#                 result.relevance_score,
#             )
#         )

#     return reranked_documents