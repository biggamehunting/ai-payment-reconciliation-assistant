from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import PointIdsList

from app.config import QDRANT_URL, QDRANT_API_KEY


# --------------------------------------------------
# 1. Connect to Qdrant
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="payment_policy",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# --------------------------------------------------
# 2. Find incorrectly labelled NetApp points
# --------------------------------------------------

print("Searching for incorrectly labelled NetApp points...")

offset = None
bad_ids = []

while True:

    points, offset = vector_store.client.scroll(
        collection_name="payment_policy",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:

        payload = point.payload or {}
        metadata = payload.get("metadata", {})

        source = metadata.get("source", "")
        header = metadata.get("Header 2", "")

        if (
            source == "Voltalia H1 2026"
            and "NetApp" in header
        ):
            bad_ids.append(point.id)

    if offset is None:
        break


# --------------------------------------------------
# 3. Show what we found
# --------------------------------------------------

print()
print("=" * 60)
print("BAD NETAPP POINTS FOUND:", len(bad_ids))
print("=" * 60)


# --------------------------------------------------
# 4. Delete only those points
# --------------------------------------------------

if bad_ids:

    print("\nDeleting incorrectly labelled NetApp points...")

    vector_store.client.delete(
        collection_name="payment_policy",
        points_selector=PointIdsList(
            points=bad_ids
        ),
    )

    print(
        "Deleted",
        len(bad_ids),
        "bad NetApp points."
    )

else:

    print("No bad NetApp points found.")


# --------------------------------------------------
# 5. Check collection size
# --------------------------------------------------

collection_info = vector_store.client.get_collection(
    "payment_policy"
)

print()
print("=" * 60)
print("CLEANUP COMPLETED")
print("=" * 60)
print(
    "Points after cleanup:",
    collection_info.points_count
)
print("=" * 60)