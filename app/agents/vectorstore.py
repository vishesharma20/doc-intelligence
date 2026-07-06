import chromadb
from sentence_transformers import SentenceTransformer

# Loads once, reused for all embedding calls
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

_client = chromadb.Client()  # in-memory, resets each run
_collection = _client.get_or_create_collection(name="documents")

def embed_text(text: str):
    return _embedder.encode(text).tolist()

def add_chunks(chunks: list[str], doc_id: str = "doc1"):
    """Embed and store chunks in the vector store."""
    embeddings = [embed_text(chunk) for chunk in chunks]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

    _collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )

def query_chunks(question: str, n_results: int = 3) -> list[str]:
    """Retrieve the most relevant chunks for a question."""
    query_embedding = embed_text(question)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]