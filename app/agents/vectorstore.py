import chromadb
from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_client = None
_collection = None

def _init_client():
    global _client, _collection
    _client = chromadb.Client()
    _collection = _client.get_or_create_collection(name="documents")

_init_client()

def embed_text(text: str):
    return _embedder.encode(text).tolist()

def clear_collection():
    """Wipes everything by rebuilding the client from scratch — no leftover state possible."""
    _init_client()

def add_chunks(chunks: list[str], doc_id: str = "doc1"):
    embeddings = [embed_text(chunk) for chunk in chunks]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    _collection.add(embeddings=embeddings, documents=chunks, ids=ids)

def query_chunks(question: str, n_results: int = 3) -> list[str]:
    query_embedding = embed_text(question)
    results = _collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["documents"][0]