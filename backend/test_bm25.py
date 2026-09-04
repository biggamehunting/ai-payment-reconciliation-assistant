from app.services.bm25_service import bm25_search


question = "What was Discovery Group's revenue in H1 2026?"

results = bm25_search(question)

print("=" * 60)
print("BM25 RESULTS")
print("=" * 60)

for i, doc in enumerate(results, 1):
    print(f"\nRESULT {i}")
    print("SOURCE:", doc.metadata.get("source"))
    print("SECTION:", doc.metadata.get("section"))
    print("TEXT:")
    print(doc.page_content[:500])