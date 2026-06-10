"""
generator.py — Grounded response generation for the Westbrook Unofficial Housing Guide.

Responsibilities:
  1. Format retrieved chunks as a numbered context block
  2. Call the Groq LLM with a strong grounding system prompt
  3. Return the answer text and the list of source documents cited

Grounding strategy:
  The system prompt explicitly prohibits the model from using outside knowledge.
  This is enforced with two specific instructions:
    (a) "Answer using ONLY the information in the provided documents below."
    (b) "If the documents do not contain enough information to answer the question,
         say exactly: 'I don't have enough information in my sources to answer that.'"

  Context is formatted as numbered passages with source labels so the model can
  cite specific sources and so we can programmatically verify attribution.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Westbrook University Unofficial Housing Guide — a helpful assistant that answers student questions about off-campus housing near Westbrook University.

CRITICAL RULES:
1. Answer using ONLY the information in the provided source documents below. Do not use any outside knowledge, general knowledge about housing, or anything from your training data.
2. If the documents do not contain enough information to answer the question, say exactly: "I don't have enough information in my sources to answer that."
3. Always cite which source document(s) your answer comes from, using the format: [Source: filename]
4. Be direct and specific. Students need actionable information.
5. If multiple documents give relevant information, synthesize them and cite all relevant sources.
6. Never guess, speculate, or fill gaps with general knowledge. Only state what is explicitly in the provided documents."""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as a numbered context block for the LLM prompt.
    Each chunk is labeled with its source document name.
    """
    if not chunks:
        return "No relevant documents found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"].replace("_", " ").replace(".txt", "")
        context_parts.append(f"[Document {i} — Source: {chunk['source']}]\n{chunk['text']}")

    return "\n\n".join(context_parts)


def generate_response(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded response using retrieved chunks as the only context.

    Returns:
      {
        "answer":   the LLM's response text,
        "sources":  list of unique source filenames cited in retrieved chunks,
        "chunks_used": number of chunks passed as context,
      }

    If no chunks were retrieved (empty list), returns the fallback message
    without calling the LLM.
    """
    if not chunks:
        return {
            "answer": "I don't have enough information in my sources to answer that.",
            "sources": [],
            "chunks_used": 0,
        }

    context = format_context(chunks)

    user_message = f"""Here are the relevant documents from the Westbrook housing knowledge base:

{context}

---

Student question: {query}

Answer the question using only the information from the documents above. Cite your sources."""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,   # low temperature for factual, consistent answers
    )

    answer = response.choices[0].message.content.strip()

    # Collect unique sources from retrieved chunks for programmatic attribution
    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }


if __name__ == "__main__":
    # Quick smoke-test (requires vector store to be built first)
    from retriever import build_vector_store, retrieve

    collection = build_vector_store()

    test_queries = [
        "What is the rent at Maplewood Apartments?",
        "How do I get my security deposit back?",
        "What is the best pizza place near Westbrook?",   # out-of-scope test
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        chunks = retrieve(q, collection)
        result = generate_response(q, chunks)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)
