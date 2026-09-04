from app.services.document_processor import extract_article_html
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

file_path = r"C:\Users\omkar\Downloads\AI\03 -20260903T094606Z-1-001\03\Earnings call transcript_ Voltalia posts stronger H1 2026 EBITDA, stock falls By Investing.html"


html = extract_article_html(file_path)

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)

chunks = splitter.split_text(html)

# Remove chunks that contain only the heading itself
filtered_chunks = []

for chunk in chunks:
    header = chunk.metadata.get("Header 2")

    if header and chunk.page_content.strip() == header.strip():
        continue

    filtered_chunks.append(chunk)

chunks = filtered_chunks

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:10]):
    print("\n====================")
    print("CHUNK:", i + 1)
    print("METADATA:", chunk.metadata)
    print("CONTENT:")
    print(chunk.page_content[:500])

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

final_chunks = recursive_splitter.split_documents(chunks)

for chunk in final_chunks:
    chunk.metadata["source"] = "Voltalia H1 2026"
    
    if "Header 2" in chunk.metadata:
        chunk.metadata["section"] = chunk.metadata["Header 2"]

print("Number of final chunks:", len(final_chunks))

for i, chunk in enumerate(final_chunks[:15]):
    print("\n====================")
    print("CHUNK:", i + 1)
    print("METADATA:", chunk.metadata)
    print("CONTENT:")
    print(chunk.page_content[:500])