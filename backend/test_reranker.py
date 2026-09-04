from app.services.hybrid_service import hybrid_search
from app.services.reranker_service import rerank_documents


question = "What was Discovery Group's revenue in H1 2026?"

print("=" * 60)
print("HYBRID SEARCH")
print("=" * 60)

candidates = hybrid_search(question, k=5)

print("Candidates:", len(candidates))

for i, doc in enumerate(candidates, 1):
    print(
        i,
        doc.metadata.get("source")
    )


print("\n" + "=" * 60)
print("COHERE RERANKING")
print("=" * 60)

results = rerank_documents(
    question,
    candidates,
    top_n=3,
)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nRANK {i}")
    print("SCORE:", score)
    print("SOURCE:", doc.metadata.get("source"))
    print("SECTION:", doc.metadata.get("section"))
    print("TEXT:")
    print(doc.page_content[:500])