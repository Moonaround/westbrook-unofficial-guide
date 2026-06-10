"""
retriever.py — Embedding and retrieval module for the Westbrook Unofficial Housing Guide.

Responsibilities:
  1. Embed all document chunks using sentence-transformers (all-MiniLM-L6-v2)
  2. Store chunks + embeddings in a ChromaDB persistent collection
  3. Given a user query, return the top-k most semantically similar chunks

Embedding model choice:
  all-MiniLM-L6-v2 was chosen because:
  - Runs locally with no API key or rate limits
  - Fast inference (~14ms per chunk on CPU)
  - Strong performance on short-to-medium informal text (reviews, FAQs)
  - 384-dimensional embeddings balance accuracy and storage efficiency

  Production tradeoffs we'd consider:
  - text-embedding-3-small (OpenAI): higher accuracy, especially for domain-specific
    terms, but adds cost (~$0.02/1M tokens) and API latency
  - multilingual-e5-large: needed if supporting non-English queries from
    international students
  - Larger local models (e5-large-v2): better accuracy for long passages but
    slower and heavier; overkill for 400-char housing review chunks
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from ingest import ingest_documents

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "westbrook_housing"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def get_collection(persist_path: str = CHROMA_DB_PATH):
    """
    Initialize (or load) the ChromaDB collection.
    Uses sentence-transformers embedding function so ChromaDB handles
    embedding automatically on add and query.
    """
    client = chromadb.PersistentClient(path=persist_path)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for text
    )
    return collection


def embed_and_store(chunks: list[dict], collection) -> None:
    """
    Embed all chunks and store them in ChromaDB.

    Each chunk is stored with:
      - document (text content)
      - id (unique string: source + chunk_id)
      - metadata (source filename, chunk_id for provenance)
    """
    documents = [c["text"] for c in chunks]
    ids = [f"{c['source']}__chunk{c['chunk_id']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    # ChromaDB add() handles embedding via the collection's embedding_function
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")


def retrieve(query: str, collection, k: int = TOP_K) -> list[dict]:
    """
    Semantic search: return the top-k chunks most relevant to `query`.

    Returns a list of dicts:
      {
        "text":     chunk text,
        "source":   originating filename,
        "chunk_id": position within source document,
        "distance": cosine distance (lower = more similar),
      }

    Results with distance > 0.75 are filtered out as too loosely related.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # _collection.query() returns nested lists (one inner list per query).
    # Since we send a single query at a time, we index [0] to unwrap.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    chunks = []
    for text, meta, dist in zip(docs, metas, dists):
        if dist <= 0.75:   # relevance threshold — drop weak matches
            chunks.append({
                "text": text,
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "distance": dist,
            })

    return chunks


def build_vector_store() -> object:
    """
    Full pipeline: ingest documents → embed → store in ChromaDB.
    Skips ingestion if the collection already has data (idempotent).
    Returns the ChromaDB collection object.
    """
    collection = get_collection()

    if collection.count() > 0:
        print(f"Vector store already populated ({collection.count()} chunks). Skipping ingestion.")
    else:
        print("Building vector store from scratch...")
        chunks = ingest_documents()
        embed_and_store(chunks, collection)

    return collection


if __name__ == "__main__":
    collection = build_vector_store()

    # Quick retrieval smoke-test
    test_queries = [
        "Which apartments are closest to Westbrook campus?",
        "How do I get my security deposit back?",
        "What bus routes serve off-campus housing?",
    ]

    print("\n--- Retrieval Test ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, collection)
        for r in results:
            print(f"  [{r['source']}] (dist: {r['distance']:.3f}) {r['text'][:100]}...")
