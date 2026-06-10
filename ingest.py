"""
ingest.py — Document ingestion pipeline for the Westbrook Unofficial Housing Guide.

Responsibilities:
  1. Load raw .txt documents from the /docs folder
  2. Clean each document (strip excess whitespace, normalize line endings)
  3. Split documents into overlapping chunks suitable for embedding

Chunking strategy:
  - Chunk size: 400 characters
  - Overlap: 80 characters
  - Rationale: Housing reviews and FAQ text are semi-structured, with key facts
    often contained in 1-3 sentences. 400 characters captures 2-4 sentences,
    preserving enough context for semantic meaning without diluting the signal.
    80-character overlap prevents key facts split across boundaries from being
    lost entirely (e.g., a sentence beginning with "This means..." that only
    makes sense with the preceding sentence).
"""

import os
import re


DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 400      # characters
CHUNK_OVERLAP = 80    # characters


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """
    Load all .txt files from the docs directory.
    Returns a list of dicts: {"source": filename, "text": raw_text}
    """
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
            documents.append({"source": filename, "text": raw_text})
            print(f"  Loaded: {filename} ({len(raw_text):,} chars)")
    return documents


def clean_document(text: str) -> str:
    """
    Clean raw document text:
    - Normalize Windows line endings to Unix
    - Collapse runs of 3+ blank lines to 2 (preserve paragraph breaks)
    - Strip leading/trailing whitespace
    - Remove lines that are only dashes (section dividers used in source docs)
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip document header lines (title + "Source: ..." attribution) so they
    # don't dominate the first chunk and bury the actual content in its embedding.
    # Matches "Westbrook University ..." title lines and "Source: ..." lines.
    text = re.sub(r"^Westbrook University.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Westbrook Area.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Source:.*$", "", text, flags=re.MULTILINE)

    # Remove lines that are purely dashes or equals (decorative separators)
    text = re.sub(r"^\s*[-=]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip overall leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split cleaned text into overlapping chunks.

    Uses a sliding window over the text:
      - Start each new chunk at (previous_start + chunk_size - overlap)
      - This ensures CHUNK_OVERLAP characters of context carry over

    Each chunk dict contains:
      - "text":     the chunk string
      - "source":   the originating filename
      - "chunk_id": zero-based index within the document
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) > 20:   # skip trivially short fragments
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": chunk_index,
            })
            chunk_index += 1

        # Advance start by (chunk_size - overlap) so the next chunk
        # re-reads the last `overlap` characters of this chunk
        start += chunk_size - overlap

    return chunks


def ingest_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """
    Full ingestion pipeline: load → clean → chunk.
    Returns a flat list of all chunks across all documents.
    """
    print("=== Document Ingestion ===")
    raw_docs = load_documents(docs_dir)

    all_chunks = []
    for doc in raw_docs:
        cleaned = clean_document(doc["text"])
        chunks = chunk_text(cleaned, doc["source"])
        all_chunks.extend(chunks)
        print(f"  Chunked {doc['source']}: {len(chunks)} chunks")

    print(f"\nTotal chunks produced: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_documents()

    print("\n--- Sample Chunks (first 5) ---")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n[Chunk {i}] source={chunk['source']} | chunk_id={chunk['chunk_id']}")
        print(chunk["text"])
        print("-" * 60)
