import os
import re
import uuid

from app.services.document_processor import extract_article_html
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import PointIdsList

from app.config import QDRANT_URL, QDRANT_API_KEY


# ==========================================================
# 1. File to ingest
# ==========================================================

file_path = r"C:\Users\omkar\Downloads\AI\03 -20260903T094606Z-1-001\03\Earnings call transcript_ Discovery Group posts strong H2 2026 profit growth By Investing.html"


# ==========================================================
# 2. Create a unique document source from filename
# ==========================================================

file_name = os.path.basename(file_path)

# Remove .html extension
source = os.path.splitext(file_name)[0]

# Clean source name
source = re.sub(r"\s+", " ", source).strip()

# Create a stable ID-safe document key
document_key = re.sub(
    r"[^a-zA-Z0-9]+",
    "-",
    source
).strip("-").lower()

print("=" * 60)
print("DOCUMENT")
print("=" * 60)
print("File:", file_name)
print("Source:", source)
print("Document key:", document_key)


# ==========================================================
# 3. Extract cleaned HTML
# ==========================================================

print("\nExtracting HTML...")

html = extract_article_html(file_path)

print("HTML extraction completed.")


# ==========================================================
# 4. Split by HTML headings
# ==========================================================

header_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)

chunks = header_splitter.split_text(html)

print("Semantic chunks:", len(chunks))


# ==========================================================
# 5. Remove heading-only chunks
# ==========================================================

filtered_chunks = []

for chunk in chunks:

    header = chunk.metadata.get("Header 2")

    if header and chunk.page_content.strip() == header.strip():
        continue

    filtered_chunks.append(chunk)

chunks = filtered_chunks

print("After removing heading-only chunks:", len(chunks))


# ==========================================================
# 6. Split large sections
# ==========================================================

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

final_chunks = recursive_splitter.split_documents(chunks)

print("Final chunks:", len(final_chunks))


# ==========================================================
# 7. Add metadata
# ==========================================================

for chunk in final_chunks:

    # Actual document source
    chunk.metadata["source"] = source

    # Document key
    chunk.metadata["document_id"] = document_key

    # Section
    if "Header 2" in chunk.metadata:
        chunk.metadata["section"] = chunk.metadata["Header 2"]


# ==========================================================
# 8. Create embedding model
# ==========================================================

print("\nCreating embedding model...")

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
# )

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==========================================================
# 9. Connect to existing Qdrant collection
# ==========================================================

print("Connecting to Qdrant...")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="payment_policy_local",
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ==========================================================
# 10. Find existing chunks for THIS document only
# ==========================================================

print("\nChecking existing Qdrant points for this document...")

offset = None
existing_ids = []

while True:

    points, offset = vector_store.client.scroll(
        collection_name="payment_policy_local",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:

        payload = point.payload or {}

        metadata = payload.get("metadata", {})

        if metadata.get("source") == source:
            existing_ids.append(point.id)

    if offset is None:
        break


print(
    "Existing points for this document:",
    len(existing_ids)
)


# ==========================================================
# 11. Delete previous version of THIS document
# ==========================================================

if existing_ids:

    print("\nDeleting previous version of this document...")

    vector_store.client.delete(
        collection_name="payment_policy_local",
        points_selector=PointIdsList(
            points=existing_ids
        ),
    )

    print(
        "Deleted",
        len(existing_ids),
        "old points."
    )

else:

    print("No previous version found.")


# ==========================================================
# 12. Create deterministic UUIDs
# ==========================================================
#
# Each document gets its own UUID namespace.
#
# Example:
#
# NetApp chunk 0
# NetApp chunk 1
# NetApp chunk 2
#
# Voltalia chunk 0
# Voltalia chunk 1
# Voltalia chunk 2
#
# They can never accidentally share IDs.
# ==========================================================

ids = []

for i in range(len(final_chunks)):

    chunk_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{document_key}-chunk-{i}"
    )

    ids.append(str(chunk_id))


# ==========================================================
# 13. Upload chunks in batches
# ==========================================================

batch_size = 10

print("\nStarting upload...")

for i in range(
    0,
    len(final_chunks),
    batch_size
):

    batch = final_chunks[
        i:i + batch_size
    ]

    batch_ids = ids[
        i:i + len(batch)
    ]

    print(
        f"Uploading chunks "
        f"{i + 1} to "
        f"{i + len(batch)} "
        f"of {len(final_chunks)}..."
    )

    vector_store.add_documents(
        batch,
        ids=batch_ids,
    )


# ==========================================================
# 14. Check final collection size
# ==========================================================

print("\nChecking final collection size...")

collection_info = vector_store.client.get_collection(
    "payment_policy_local"
)

print("=" * 60)
print("INGESTION COMPLETED")
print("=" * 60)

print("Document:", source)
print("New chunks:", len(final_chunks))
print("Points after ingestion:", collection_info.points_count)
print("=" * 60)