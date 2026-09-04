from app.services.hybrid_service import hybrid_search


question = "What was Discovery Group's revenue in H1 2026?"

results = hybrid_search(
    question,
    k=5,
    top_n=3,
)

print("=" * 60)
print("HYBRID + RERANKING RESULTS")
print("=" * 60)

for i, doc in enumerate(results, 1):
    print(f"\nRANK {i}")
    print("SOURCE:", doc.metadata.get("source"))
    print("SECTION:", doc.metadata.get("section"))
    print("TEXT:")
    print(doc.page_content[:500])

# question = "What was Discovery Group's revenue in H1 2026?"

# results = hybrid_search(question, k=5)

# print("=" * 60)
# print("HYBRID RESULTS")
# print("=" * 60)

# for i, doc in enumerate(results, 1):
#     print(f"\nRESULT {i}")
#     print("SOURCE:", doc.metadata.get("source"))
#     print("SECTION:", doc.metadata.get("section"))
#     print("TEXT:")
#     print(doc.page_content[:500])