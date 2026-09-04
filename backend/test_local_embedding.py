from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


text = "What was Discovery Group's revenue in H1 2026?"

vector = embeddings.embed_query(text)


print("Embedding created successfully.")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])