"""
query.py — End-to-end query pipeline for the Westbrook Unofficial Housing Guide.

Wires together:
  retriever.retrieve()  →  generator.generate_response()

The `ask()` function is the single entry point used by app.py (Gradio UI)
and can be imported for evaluation scripts or CLI use.
"""

from retriever import build_vector_store, retrieve
from generator import generate_response

# Build / load the vector store once at module import time.
# Subsequent calls to ask() reuse the same collection object.
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = build_vector_store()
    return _collection


def ask(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks, generate grounded answer.

    Args:
        question: the user's plain-language question
        top_k:    number of chunks to retrieve (default 5)

    Returns:
        {
            "answer":      str  — the LLM's grounded response
            "sources":     list — unique source filenames cited
            "chunks_used": int  — number of chunks passed to LLM
            "chunks":      list — the actual retrieved chunk dicts (for debugging)
        }
    """
    collection = get_collection()
    chunks = retrieve(question, collection, k=top_k)
    result = generate_response(question, chunks)
    result["chunks"] = chunks
    return result


if __name__ == "__main__":
    print("Westbrook Unofficial Housing Guide — CLI Mode")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = ask(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources']) if result['sources'] else 'None'}")
        print(f"(Retrieved {result['chunks_used']} chunks)\n")
        print("=" * 60)
