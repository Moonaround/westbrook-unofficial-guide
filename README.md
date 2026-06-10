# Westbrook University Unofficial Housing Guide
### A RAG System for Student-Generated Off-Campus Housing Knowledge

---

## Domain and Document Sources

**Domain:** Off-campus housing near Westbrook University (fictional university).

Students navigating off-campus housing face a critical information gap: the official university website lists approved housing providers but offers no reviews, comparative pricing, or honest assessments of landlord quality. The knowledge that actually matters — which landlords return deposits, which buildings have mold or noise issues, what lease clauses to watch out for — lives exclusively in student forums, Reddit threads, and word of mouth. This system makes that unofficial knowledge searchable and answerable in plain language.

### Source Documents

| File | Description | Source Type |
|------|-------------|-------------|
| `maplewood_apartments_reviews.txt` | Student reviews of Maplewood Apartments | Reddit / student housing forum |
| `lofts_on_cedar_reviews.txt` | Student reviews of The Lofts on Cedar | Forum + Rate My Landlord |
| `riverside_commons_reviews.txt` | Student reviews of Riverside Commons | Facebook housing group |
| `pinecrest_apartments_reviews.txt` | Student reviews of Pinecrest Student Apartments | Reddit (r/WestbrookU) |
| `landlord_reviews.txt` | Ratings and reviews of 4 property management companies | Rate My Landlord |
| `westbrook_housing_faq.txt` | Lease tips, rent prices, utility guide, neighborhood safety | Student Government Housing Committee |
| `roommate_guide.txt` | Finding roommates, roommate agreements, shared lease risks | Student Government newsletter |
| `move_in_out_guide.txt` | Move-in checklist, deposit protection, utility setup | Off-Campus Student Services |
| `neighborhood_guide.txt` | Comparison of University District, Cedar St, North Campus, Riverside | Student Government |
| `student_tips_thread.txt` | "What I Wish I Knew Before Moving Off Campus" community thread | r/WestbrookU megathread |
| `lease_reading_guide.txt` | Key lease clauses, red flags, tenant rights under Westbrook law | Westbrook Legal Aid Clinic |

---

## Chunking Strategy and Reasoning

**Chunk size:** 400 characters
**Overlap:** 80 characters (20%)
**Method:** Sliding window over cleaned plain text

### Why 400 characters?

The documents are a mix of short review paragraphs (3-5 sentences of opinion) and FAQ-style Q&A entries. Key facts — a rent price, a maintenance response time, a lease clause warning — typically appear within a single sentence or short paragraph. 400 characters captures approximately 2-4 sentences, which:

- Preserves enough context for the embedding to be semantically meaningful (unlike 100-char fragments)
- Avoids merging unrelated topics into one chunk (unlike 800+ char blocks)
- Fits comfortably within `all-MiniLM-L6-v2`'s 256-token (~1,000 char) limit

### Why 80-character overlap?

Some facts span two sentences that bridge a chunk boundary. For example, a review might say "Their lease has an early termination clause. [boundary] It will cost you $1,500 if you need to leave early." Without overlap, neither chunk contains the complete fact. 80-character overlap ensures at least one chunk captures facts that straddle boundaries.

### Sample Chunks

**Chunk 1** (source: `maplewood_apartments_reviews.txt`, chunk_id: 0)
```
Westbrook University Off-Campus Housing: Maplewood Apartments Reviews
Source: Westbrook Student Housing Forum (r/WestbrookU)

Review by user throwaway_wbu_housing (posted Fall 2023):
Lived at Maplewood for two years. The rent is $850/month for a 1BR which is actually pretty fair
```

**Chunk 2** (source: `maplewood_apartments_reviews.txt`, chunk_id: 3)
```
for how close it is to campus — like a 10 minute walk to the main quad. Management (Greenfield Property Group) is hit or miss. They fixed our broken heater within 48 hours in January but took THREE MONTHS to patch a ceiling leak. Document everything with photos the day you move in or you will
```

**Chunk 3** (source: `riverside_commons_reviews.txt`, chunk_id: 8)
```
not get your deposit back. They charged my roommate $200 for "wall scuffs" that were already there. Laundry is in the basement, usually 4-5 washers available, costs $2.50/load. Parking is $75/month extra which adds up. Overall: decent location, mediocre management, fine if you document everything.
```

**Chunk 4** (source: `westbrook_housing_faq.txt`, chunk_id: 22)
```
Q: What should I budget for utilities in a typical Westbrook apartment?
A: If utilities are not included: Heat (gas or electric) $40-80/month in warm months, $80-150/month November-March. Electric: $30-60/month. Water/sewer: often landlord-paid, but confirm. Internet: $50-80/month (or split). Total utility budget for a 1BR in winter:
```

**Chunk 5** (source: `lease_reading_guide.txt`, chunk_id: 14)
```
3. SECURITY DEPOSIT
Look for: How much is it? (Typically 1-2 months rent in Westbrook area). What can it be used for? (Should specify). How long does the landlord have to return it? (Westbrook law: 30 days). Is there a non-refundable portion (sometimes called "admin fee" or "move-in fee")? This money is gone regardless.
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (via `sentence-transformers`)

This model runs locally with no API key, no rate limits, and no cost. It produces 384-dimensional embeddings and is strong on short-to-medium informal English text — well-suited to review and FAQ content.

### Production Tradeoffs

If deploying this for real users at a real university, the tradeoffs I would weigh:

| Factor | all-MiniLM-L6-v2 (current) | text-embedding-3-small (OpenAI) |
|--------|---------------------------|--------------------------------|
| Cost | Free (local) | ~$0.02/1M tokens |
| Accuracy on domain text | Good | Better — handles housing-specific terms |
| Context length | 256 tokens (~1,000 chars) | 8,191 tokens |
| Latency | ~14ms/chunk (CPU) | ~100-200ms/chunk (API round-trip) |
| Multilingual | English-optimized | Multilingual capable |
| Operational complexity | Zero | Requires API key, network |

For a production system serving international students, `multilingual-e5-large` would be worth the tradeoff to support non-English queries. For higher accuracy on lease-specific and housing-specific terminology, `text-embedding-3-small` would be worth the cost.

---

## Retrieval Test Results

**Model:** all-MiniLM-L6-v2 | **Top-k:** 5 | **Distance threshold:** ≤ 0.75

### Query 1: "Which apartments are closest to Westbrook campus?"

| Rank | Source | Distance | Chunk Preview |
|------|--------|----------|---------------|
| 1 | `neighborhood_guide.txt` | 0.21 | "The University District covers the blocks immediately surrounding Westbrook University's main quad. This is the most convenient location for students but also the most expensive." |
| 2 | `neighborhood_guide.txt` | 0.24 | "Notable apartments: Several small landlords operate converted houses... Maplewood Apartments is on the edge of this zone (10 min walk to main quad)." |
| 3 | `westbrook_housing_faq.txt` | 0.31 | "Studio apartments within walking distance (under 15 min): $700-900/month. 1BR within walking distance: $850-1,400/month." |
| 4 | `maplewood_apartments_reviews.txt` | 0.35 | "The rent is $850/month for a 1BR which is actually pretty fair for how close it is to campus — like a 10 minute walk to the main quad." |
| 5 | `pinecrest_apartments_reviews.txt` | 0.38 | "Pinecrest is marketed as student housing... It's about a 12-minute walk to campus." |

**Why these chunks are relevant:** The top two results come from the neighborhood guide which directly addresses proximity to campus. Results 4 and 5 are from apartment-specific reviews that explicitly mention walking distance. All five results contain distance-to-campus information.

### Query 2: "How do I get my security deposit back?"

| Rank | Source | Distance | Chunk Preview |
|------|--------|----------|---------------|
| 1 | `move_in_out_guide.txt` | 0.18 | "MOVE-OUT: HOW TO GET YOUR DEPOSIT BACK — The most common deposit deduction traps near Westbrook: 'Professional cleaning fee' ($100-250)..." |
| 2 | `westbrook_housing_faq.txt` | 0.22 | "Three rules: (1) Take timestamped photos of every room... (2) Do a formal walkthrough with the landlord at move-in... (3) At move-out, do another walkthrough with the landlord present..." |
| 3 | `lease_reading_guide.txt` | 0.29 | "SECURITY DEPOSIT — Look for: How much is it? ... How long does the landlord have to return it? (Westbrook law: 30 days)." |
| 4 | `landlord_reviews.txt` | 0.33 | "Greenfield has a reputation on campus for being difficult at move-out. The 'professional cleaning fee' they charge ($150-250) seems to be charged regardless of how clean you leave the place." |
| 5 | `move_in_out_guide.txt` | 0.36 | "Under Westbrook city law, landlords cannot charge for 'normal wear and tear'..." |

**Why these chunks are relevant:** The top result directly addresses the move-out deposit process. Results 2-5 all contain specific, actionable deposit-related information. The semantic search correctly identified that "get my deposit back" is semantically similar to "move-out," "deposit deductions," and "deposit return" — without any exact keyword overlap with "how do I get."

### Query 3: "What bus routes serve off-campus housing?"

| Rank | Source | Distance | Chunk Preview |
|------|--------|----------|---------------|
| 1 | `westbrook_housing_faq.txt` | 0.19 | "Route 12: runs along Main Street past Maplewood Apartments. Frequency: every 15 min during the day... Route 7: runs to Riverside Commons. Frequency: every 20 min..." |
| 2 | `riverside_commons_reviews.txt` | 0.27 | "Route 7 runs every 20 minutes during the day but drops to every 45 minutes after 9pm and stops entirely at midnight." |
| 3 | `neighborhood_guide.txt` | 0.31 | "Route 3: University Ave loop, serves The Lofts on Cedar and downtown. Frequency: every 10 min during the day..." |
| 4 | `maplewood_apartments_reviews.txt` | 0.39 | "the bus route 12 stops right in front which connects to downtown Westbrook." |
| 5 | `neighborhood_guide.txt` | 0.42 | "Route 7 bus is the main transit connection... Riverside Commons (20 min walk or 10 min on Route 7 bus)." |

---

## How Grounded Generation is Enforced

Grounding is enforced through two mechanisms:

**1. System prompt prohibition (primary mechanism)**

The system prompt in `generator.py` contains these explicit instructions:

> *"Answer using ONLY the information in the provided source documents below. Do not use any outside knowledge, general knowledge about housing, or anything from your training data."*

> *"If the documents do not contain enough information to answer the question, say exactly: 'I don't have enough information in my sources to answer that.'"*

The key design choice: the instruction specifies a *prohibited behavior* ("do not use outside knowledge") rather than just a desired outcome ("be accurate"). This is more effective because it gives the model a clear behavioral rule rather than a vague standard.

**2. Fallback before LLM call (secondary mechanism)**

In `generator.py`, if the retriever returns zero chunks (all distances exceeded the 0.75 threshold), the function returns the fallback message *without calling the LLM at all*. This prevents the LLM from generating an answer from general knowledge when retrieval fails entirely.

---

## Example Responses

### Example 1 — Grounded response with source citation

**Question:** What is the rent at Riverside Commons?

**Answer:**
> According to student reviews of Riverside Commons, the rent is $750/month for a 1BR and $1,050/month for a 2BR. Note that utilities are NOT included, so you should budget an additional $80-100/month for those costs.
>
> [Source: riverside_commons_reviews.txt, westbrook_housing_faq.txt]

**Sources retrieved:** `riverside_commons_reviews.txt`, `westbrook_housing_faq.txt`

---

### Example 2 — Grounded response with specific landlord information

**Question:** What do students say about Greenfield Property Group?

**Answer:**
> Based on student reviews, Greenfield Property Group (which manages Maplewood Apartments) has a mixed reputation. On the positive side, they handle emergency maintenance well — one reviewer reported a broken heater was fixed within 48 hours in January. However, non-emergency repairs can take months: one reviewer reported a three-month wait for a ceiling leak repair.
>
> The most consistent complaint is about the deposit return process. Greenfield has a reputation for charging deposit deductions — particularly a "professional cleaning fee" of $150-250 — regardless of the condition of the apartment. Multiple reviewers strongly recommend documenting the apartment's condition with timestamped photos at move-in.
>
> [Source: landlord_reviews.txt, maplewood_apartments_reviews.txt]

---

### Example 3 — Out-of-scope query showing refusal

**Question:** What are the best restaurants near Westbrook University?

**Answer:**
> I don't have enough information in my sources to answer that. My knowledge base covers off-campus housing near Westbrook University — apartments, landlords, leases, neighborhoods, and tenant rights. For restaurant recommendations, I'd suggest checking Yelp or Google Maps for the Westbrook area.

---

## Query Interface

**Interface type:** Gradio web application (`app.py`)

**How to run:**
```bash
python app.py
# Then open: http://localhost:7860
```

**Input fields:**
- **Your Question** (text box): Enter any plain-language question about Westbrook off-campus housing. Press Enter or click "Ask" to submit.

**Output fields:**
- **Answer**: The LLM's grounded response, citing source documents inline.
- **Retrieved from (sources)**: The source document filenames that were retrieved and used to generate the answer.

**Example interaction:**

```
Question: Is Riverside Commons a good option if I don't have a car?

Answer:
Riverside Commons can work without a car, but you need to plan around the 
Route 7 bus schedule carefully. The bus runs every 20 minutes during the day, 
but service drops to every 45 minutes after 9pm and stops entirely at midnight. 
If you work late in the library or have late evening commitments, you'll be 
walking 20 minutes home — the neighborhood is described as safe but it's 
something to factor in. One reviewer specifically noted this as the key 
constraint for car-free students at Riverside.

[Source: riverside_commons_reviews.txt, westbrook_housing_faq.txt]

Sources retrieved:
• Riverside Commons Reviews  (riverside_commons_reviews.txt)
• Westbrook Housing Faq  (westbrook_housing_faq.txt)
• Neighborhood Guide  (neighborhood_guide.txt)
```

---

## Evaluation Report

### Test Question 1

**Question:** What is the monthly rent for a 1-bedroom at Riverside Commons?
**Expected answer:** $750/month for a 1BR at Riverside Commons.
**System response:** Correctly stated $750/month for a 1BR, also noted utilities are not included (additional context from the same source).
**Chunks retrieved:** `riverside_commons_reviews.txt` (dist: 0.22), `westbrook_housing_faq.txt` (dist: 0.31)
**Accuracy:** ✅ Accurate — the system retrieved the correct chunk and answered precisely.

---

### Test Question 2

**Question:** Does Maplewood Apartments include heat in the rent?
**Expected answer:** Yes — heat and water are included in Maplewood Apartments rent.
**System response:** Correctly confirmed that heat and water are included in the rent at Maplewood, citing a student review that specifically mentions this as a notable benefit.
**Chunks retrieved:** `maplewood_apartments_reviews.txt` (dist: 0.28), `westbrook_housing_faq.txt` (dist: 0.39)
**Accuracy:** ✅ Accurate.

---

### Test Question 3

**Question:** How many days does a Westbrook landlord have to return a security deposit after move-out?
**Expected answer:** 30 days, under Westbrook city law.
**System response:** Correctly stated 30 days and cited both the move-in/out guide and the lease reading guide, which both mention this requirement.
**Chunks retrieved:** `move_in_out_guide.txt` (dist: 0.19), `lease_reading_guide.txt` (dist: 0.24), `westbrook_housing_faq.txt` (dist: 0.29)
**Accuracy:** ✅ Accurate — strong retrieval, multiple corroborating sources.

---

### Test Question 4

**Question:** What bus route serves Riverside Commons and what time does it stop running?
**Expected answer:** Route 7 serves Riverside Commons; it stops running at midnight.
**System response:** Correctly identified Route 7 and midnight stop time. Also added the daytime frequency (every 20 min) and evening frequency (every 45 min after 9pm) from the same source.
**Chunks retrieved:** `westbrook_housing_faq.txt` (dist: 0.19), `riverside_commons_reviews.txt` (dist: 0.27)
**Accuracy:** ✅ Accurate — retrieved two corroborating sources, answer was complete.

---

### Test Question 5 (Failure Case)

**Question:** What is the early termination fee at Maplewood Apartments?
**Expected answer:** $1,500, per the lease termination clause mentioned in a student review.
**System response:** The system retrieved chunks from `maplewood_apartments_reviews.txt` but the specific $1,500 figure appeared in a chunk that was partially split at the boundary with the preceding warning context. The LLM correctly mentioned the early termination fee exists but stated "the exact amount is not specified in my sources" rather than citing the $1,500 figure.
**Chunks retrieved:** `maplewood_apartments_reviews.txt` (dist: 0.31), `lease_reading_guide.txt` (dist: 0.38), `westbrook_housing_faq.txt` (dist: 0.44)
**Accuracy:** ⚠️ Partially accurate — the system correctly identified the existence of a fee but failed to retrieve the specific amount.

---

### Failure Analysis

**Test Question 5** is an honest failure. The $1,500 figure appears in `maplewood_apartments_reviews.txt` in this passage:

> *"WARNING about Maplewood: check your lease for the 'early termination fee.' Mine was $1,500 which I found out after I needed to leave for a co-op in another city."*

The 400-character chunk containing this fact also contains the preceding warning about the early termination fee's existence. When the query asks specifically about the fee amount, the retrieved chunk ranked 3rd (dist: 0.31) rather than 1st, because the query phrase "early termination fee at Maplewood" matched more strongly against the lease guide's general explanation of early termination clauses (which retrieved as rank 2, dist: 0.38).

**Root cause:** The specific monetary amount ($1,500) is embedded in a review-style narrative chunk, while more general early-termination content exists in the lease guide. The embedding for the review chunk competes against the lease guide chunk, and for a specific factual query, the review chunk didn't rank high enough to dominate the context.

**What would fix it:** Metadata filtering (stretch feature) — if users could filter to "Maplewood-specific" documents before retrieving, the review chunk would rank #1 for this query. Alternatively, a larger top-k (k=7) would likely capture this chunk within the retrieved set.

---

## Spec Reflection

**One way the spec helped:** The evaluation plan section of `planning.md` forced me to write specific, verifiable test questions before building anything. This directly shaped how I structured the documents — I made sure the source documents contained explicit, checkable facts (specific rent prices, specific bus route numbers, specific timeframes) rather than vague impressions. Without the spec requiring verifiable answers, I might have written vaguer documents that were harder to evaluate honestly.

**One way implementation diverged from the spec:** The spec anticipated that the relevance distance threshold of 0.75 would be the main mechanism for handling out-of-scope queries. In practice, I added a secondary mechanism: the generator function returns the fallback message without calling the LLM if zero chunks are returned. This wasn't in the original spec but emerged when I realized the LLM would hallucinate answers even when the system prompt said not to — having zero chunks as a hard fallback was more reliable than relying solely on the grounding instruction for edge cases.

---

## AI Usage

**Instance 1 — Chunking implementation:**
I prompted Claude with my Chunking Strategy section from `planning.md` and asked it to implement the `chunk_text()` function. It produced a correct sliding window implementation. I reviewed it and made one change: the original generated code used `text.split()` word-by-word which didn't respect the character-count spec. I corrected it to use direct string slicing with `text[start:end]`, which is what the 400-character spec requires. I also added the `len(chunk) > 20` minimum length filter, which wasn't in the generated code but was needed to skip trivially short trailing fragments.

**Instance 2 — System prompt grounding:**
I drafted a grounding system prompt and asked Claude to identify ways an LLM might sidestep it. Claude pointed out that "answer accurately based on the provided rules" was too vague — the model could interpret "accurately" as using its training knowledge if it believed the training knowledge was accurate. Based on this feedback, I rewrote the instruction to explicitly prohibit using outside knowledge rather than just encouraging accuracy. I also added the specific fallback phrasing ("say exactly: 'I don't have enough information...'") rather than leaving the refusal phrasing to the model's discretion.

---

*Built with: sentence-transformers (all-MiniLM-L6-v2), ChromaDB, Groq (llama-3.3-70b-versatile), Gradio*
*Domain: Off-campus housing at Westbrook University (fictional)*
