from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import QDRANT_URL, QDRANT_API_KEY


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

collection_name = "payment_policy_local"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    ),
)

print("Collection created successfully.")
print("Collection:", collection_name)
print("Vector size: 384")
print("Distance: COSINE")