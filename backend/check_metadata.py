from collections import Counter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.config import QDRANT_URL, QDRANT_API_KEY


# --------------------------------------------------
# 1. Connect to Qdrant
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="payment_policy",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# --------------------------------------------------
# 2. Read all points
# --------------------------------------------------

offset = None
points = []

while True:

    batch, offset = vector_store.client.scroll(
        collection_name="payment_policy",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    points.extend(batch)

    if offset is None:
        break


print("=" * 70)
print("TOTAL POINTS:", len(points))
print("=" * 70)


# --------------------------------------------------
# 3. Count sources
# --------------------------------------------------

source_counts = Counter()

for point in points:

    payload = point.payload or {}
    metadata = payload.get("metadata", {})

    source = metadata.get("source", "<NO SOURCE>")

    source_counts[source] += 1


print("\nSOURCE COUNTS")
print("-" * 70)

for source, count in source_counts.most_common():

    print(f"{count:5}  {source}")


# --------------------------------------------------
# 4. Show earnings-document metadata
# --------------------------------------------------

print("\n\nEARNINGS DOCUMENT SAMPLES")
print("=" * 70)

shown = 0

for point in points:

    payload = point.payload or {}
    metadata = payload.get("metadata", {})

    header = metadata.get("Header 2", "")
    source = metadata.get("source", "")
    document_id = metadata.get("document_id", "")
    section = metadata.get("section", "")

    # Look for the earnings documents using their headings
    if any(
        keyword in header.lower()
        for keyword in [
            "netapp",
            "voltalia",
            "discovery",
        ]
    ):

        print("\nPOINT ID:", point.id)
        print("SOURCE:", source)
        print("DOCUMENT ID:", document_id)
        print("HEADER 2:", header)
        print("SECTION:", section)

        shown += 1

        if shown >= 20:
            break


print("\n" + "=" * 70)
print("Displayed:", shown, "sample earnings points")
print("=" * 70)