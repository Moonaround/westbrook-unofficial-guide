# planning.md — Westbrook Unofficial Housing Guide

## Domain

**Off-campus housing near Westbrook University.**

Students at Westbrook need to make high-stakes housing decisions — signing year-long leases, choosing between landlords, managing deposits — but the official university website only lists approved housing providers without any real reviews or comparative information. The knowledge that actually matters (which landlord actually returns deposits, which building has noise or mold issues, what the bus schedule actually means for your daily life) lives exclusively in student forums, Reddit threads, and word of mouth. This system makes that unofficial knowledge searchable and answerable.

---

## Documents

11 source documents collected from student forums, Rate My Landlord, Reddit (r/WestbrookU), student government housing newsletters, and legal aid guides:

1. `maplewood_apartments_reviews.txt` — Student reviews of Maplewood Apartments (Reddit / housing forum)
2. `lofts_on_cedar_reviews.txt` — Student reviews of The Lofts on Cedar (forum + Rate My Landlord)
3. `riverside_commons_reviews.txt` — Student reviews of Riverside Commons (Facebook group)
4. `pinecrest_apartments_reviews.txt` — Student reviews of Pinecrest Student Apartments (Reddit)
5. `landlord_reviews.txt` — Ratings and reviews of 4 property management companies (Rate My Landlord)
6. `westbrook_housing_faq.txt` — Student Government Housing Committee FAQ (lease tips, prices, utilities)
7. `roommate_guide.txt` — Roommate finding and agreement guide (Student Government newsletter)
8. `move_in_out_guide.txt` — Move-in/move-out checklist and deposit guide (Off-Campus Student Services)
9. `neighborhood_guide.txt` — Neighborhood comparison guide (Student Government Housing Committee)
10. `student_tips_thread.txt` — r/WestbrookU megathread: "What I Wish I Knew Before Moving Off Campus"
11. `lease_reading_guide.txt` — Lease clause guide and red flags (Westbrook Legal Aid Clinic)

Sources together cover: apartment-specific reviews, landlord quality, pricing, utilities, leases, neighborhoods, transportation, move-in/out process, and roommate logistics.

---

## Chunking Strategy

**Chunk size:** 400 characters
**Overlap:** 80 characters

**Rationale:**

The documents are a mix of short review-style paragraphs (3-5 sentences) and FAQ-style Q&A entries. The key facts — a rent price, a maintenance response time, a lease clause warning — typically appear within a single sentence or short paragraph.

- **400 characters** captures roughly 2-4 sentences, which is enough to preserve the context around a fact without merging unrelated topics. A review saying "Management fixed our heater in 48 hours but took 3 months to fix a ceiling leak" fits comfortably and retrieves as a complete thought.
- **80-character overlap** (20% of chunk size) prevents facts split at boundaries from becoming unrecoverable. For example, an FAQ answer that begins "This is especially important because..." needs the preceding sentence to make sense — overlap ensures at least one chunk contains both.
- **Why not smaller (e.g., 150 chars)?** Review fragments like "Professor Smith's exams are heavy" carry no useful meaning alone. Similarly, "Documentation is key" without the surrounding context is unembeddable meaningfully.
- **Why not larger (e.g., 800+ chars)?** Merging multiple distinct tips into one chunk dilutes the embedding signal. A chunk covering rent, mold, and parking in the same passage will match weakly for all three and strongly for none.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`
**Vector store:** ChromaDB (local, persistent)
**Top-k:** 5 chunks per query
**Distance metric:** Cosine similarity (lower = more similar)
**Relevance threshold:** Distance ≤ 0.75 (chunks above this are filtered as too loosely related)

**Why all-MiniLM-L6-v2?**
- Runs entirely locally — no API key, no rate limits, no cost
- Strong performance on short to medium informal English text
- 384-dimensional embeddings are efficient for our ~400 chunk corpus
- Inference is fast enough (~14ms/chunk on CPU) for real-time queries

**Production tradeoffs:**
- **text-embedding-3-small (OpenAI):** Higher accuracy, especially for housing-specific terminology ("joint and several liability," "broom clean"), but adds ~$0.02/1M token cost and API latency
- **multilingual-e5-large:** Would be necessary to support international student queries in their native language; Westbrook has a substantial international student population
- **Context length:** all-MiniLM-L6-v2 supports 256 tokens (~1,000 chars). Our 400-char chunks are well within this limit, but longer document types would require a different model
- **Local vs. API:** Local models eliminate per-query cost and latency; API models (OpenAI, Cohere) offer better accuracy but add operational complexity and cost

**Why top-k=5?**
- Too few (k=2): Relevant content may not be in the retrieved set, especially for multi-faceted questions ("Is Maplewood good overall?")
- Too many (k=8+): Loosely related chunks dilute the LLM's context, increasing hallucination risk and reducing answer precision
- k=5 with a 0.75 distance filter provides a practical balance for our ~400 chunk corpus

---

## Evaluation Plan

5 test questions with specific, verifiable expected answers:

**Q1:** What is the monthly rent for a 1-bedroom at Riverside Commons?
*Expected answer:* $750/month for a 1BR at Riverside Commons.

**Q2:** Does Maplewood Apartments include heat in the rent?
*Expected answer:* Yes, heat and water are included in Maplewood Apartments rent.

**Q3:** How many days does a Westbrook landlord have to return a security deposit after move-out?
*Expected answer:* 30 days, under Westbrook city law.

**Q4:** What bus route serves Riverside Commons and what time does it stop running?
*Expected answer:* Route 7 serves Riverside Commons; it stops running at midnight.

**Q5:** What is the early termination fee at Maplewood Apartments?
*Expected answer:* $1,500 (mentioned in the anonymous review warning about the early termination clause).

---

## Anticipated Challenges

**Challenge 1 — Chunk boundary splits on key facts:**
Some facts span two sentences that bridge a chunk boundary. For example, "The parking situation near the CS building has gotten worse since construction started" followed by "and it now costs $75/month" could be split. The 80-character overlap mitigates but does not eliminate this. I'll monitor for this in evaluation.

**Challenge 2 — Inconsistent source attribution:**
The LLM may attribute an answer to the wrong source document if multiple chunks contain similar content (e.g., both Maplewood and Riverside reviews mention the Route 12 bus). Source attribution is programmatically enforced from retrieved chunk metadata, not left to the model to generate — this reduces but doesn't eliminate the risk of misleading attribution.

**Challenge 3 — Out-of-scope queries returning weak matches:**
A query like "What's the best bar near Westbrook?" has no good match in our documents. ChromaDB will still return the k nearest neighbors even if they're irrelevant. The 0.75 distance threshold helps, but if all chunks exceed the threshold, the system returns an empty list and the fallback message.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WESTBROOK UNOFFICIAL GUIDE                      │
└─────────────────────────────────────────────────────────────────────┘

  /docs/*.txt
      │
      ▼
┌─────────────────┐
│  Document       │  ingest.py
│  Ingestion      │  • Load .txt files
│                 │  • Clean text (normalize, strip decorators)
└────────┬────────┘
         │  raw document text
         ▼
┌─────────────────┐
│  Chunking       │  ingest.py → chunk_text()
│                 │  • 400-char chunks
│                 │  • 80-char overlap
│                 │  • Metadata: source, chunk_id
└────────┬────────┘
         │  list of chunk dicts
         ▼
┌─────────────────┐
│  Embedding +    │  retriever.py
│  Vector Store   │  • all-MiniLM-L6-v2 (sentence-transformers)
│                 │  • ChromaDB (local persistent)
└────────┬────────┘
         │
    ┌────┴────────────────────────────────────┐
    │           AT QUERY TIME                  │
    └────┬────────────────────────────────────┘
         │  user query string
         ▼
┌─────────────────┐
│  Retrieval      │  retriever.py → retrieve()
│                 │  • Embed query (same model)
│                 │  • Cosine similarity search
│                 │  • Return top-5 chunks (dist ≤ 0.75)
└────────┬────────┘
         │  top-k chunks + metadata
         ▼
┌─────────────────┐
│  Generation     │  generator.py → generate_response()
│                 │  • Groq llama-3.3-70b-versatile
│                 │  • Strong grounding system prompt
│                 │  • Source attribution in response
└────────┬────────┘
         │  answer + sources
         ▼
┌─────────────────┐
│  Gradio UI      │  app.py
│                 │  • Question input
│                 │  • Answer display
│                 │  • Sources display
└─────────────────┘
```

---

## AI Tool Plan

| Pipeline Component | What I'll give the AI | What I expect it to produce |
|---|---|---|
| `ingest.py` (chunking) | This Chunking Strategy section + example document text | `chunk_text()` function with sliding window logic using specified chunk_size and overlap |
| `retriever.py` (embedding + ChromaDB) | Retrieval Approach section + pipeline diagram | `embed_and_store()` and `retrieve()` functions using sentence-transformers and ChromaDB API |
| `generator.py` (grounding prompt) | Grounding requirement + example of desired output format | System prompt text and `generate_response()` function structure |
| `app.py` (Gradio UI) | Desired input/output fields, example questions list | Gradio Blocks layout with question input, answer output, sources output |
| `README.md` (evaluation section) | 5 test questions + system outputs | Evaluation table structure (not the content — I'll fill actual results) |
