# Lunorsoft AI Developer — Round 1 Assignment

## Option 1: Mini AI Knowledge Assistant (RAG)

### Project Requirements Document

---

## 1. Core Requirements (from assignment brief — mandatory)

| # | Requirement | Notes |
| --- | --- | --- |
| 1 | Use a PDF/document or collection of documents as the knowledge source | |
| 2 | Extract and process the document content | |
| 3 | Split the content into appropriate chunks | Implemented via advanced/semantic chunking (see §3) |
| 4 | Generate embeddings for the content | |
| 5 | Store and retrieve relevant information using a vector store | |
| 6 | Accept questions from the user | |
| 7 | Retrieve relevant information and generate answers using an LLM | |
| 8 | Ensure answers are primarily based on the provided knowledge source | Implemented via grounding/faithfulness check (see §3) |
| 9 | Provide a simple and usable interface | |

---

## 2. Bonus Features Selected (from assignment brief — optional)

| Feature | Status |
| --- | --- |
| Source citations | ✅ Include |
| Conversation history | 🕒 Stretch goal (only if time allows) |
| Multiple documents | ✅ Include |
| Improved retrieval / evaluation | ✅ Include (via §3 additions) |
| Deployment | 🕒 Stretch goal (Streamlit Cloud, attempt last) |

---

## 3. Differentiation Features (beyond naive RAG — mandatory for this build)

**A. Hybrid Search**

- Combine BM25 (keyword) retrieval with vector similarity retrieval
- Purpose: catch exact-keyword matches that pure vector search misses

**B. Re-ranking**

- Cross-encoder reranker applied to top-k retrieved chunks before passing to the LLM
- Purpose: improve precision of the final context window

**C. Query Transformation**

- Rewrite/expand the raw user query before retrieval (e.g., paraphrasing, sub-query generation, or HyDE-style hypothetical answer embedding)
- Purpose: improve retrieval quality on vague/poorly-phrased questions

**D. Grounding / Faithfulness Check**

- Post-generation check verifying the answer is supported by the retrieved context (LLM-as-judge or overlap heuristic)
- Purpose: directly satisfies core requirement #8; surface a confidence/faithfulness indicator in the UI

**E. Advanced Chunking Strategy**

- Semantic or structure-aware chunking (e.g., split by headers/sections, or semantic similarity grouping) instead of fixed-size splitting
- Purpose: produce more coherent, context-preserving chunks

---

## 4. Parked / Future Work

- **Formal evaluation harness** (e.g., RAGAS or similar)
  - Decision: revisit only after core pipeline (§1–3) is complete and stable. If not implemented in time, document as "future work" in the README.

---

## 5. Priority Order for Build

**Must ship (in this order):**

1. Core pipeline: extraction → chunking → embeddings → vector store
2. Basic retrieval + generation + Streamlit UI (requirements 1–9)
3. Advanced chunking (E)
4. Hybrid search (A)
5. Re-ranking (B)
6. Query transformation (C)
7. Grounding/faithfulness check (D)
8. Source citations
9. Multiple document support

**Stretch (attempt only if time remains):**
10. Conversation history
11. Deployment (Streamlit Community Cloud)
12. Formal eval harness (RAGAS)

---

## 6. Submission Checklist (per assignment brief, §09)

- [ ] GitHub repository (public/accessible)
- [ ] `requirements.txt` (Python dependencies) and `.env.example` included
- [ ] Working demo/link, if available (or screen recording as fallback)
- [ ] README covering: approach, architecture, why-beyond-naive-RAG rationale, tech stack, setup instructions, AI tools disclosure, before/after example query, limitations/future work
- [ ] AI tools used clearly disclosed
- [ ] All links tested in incognito/private browser
- [ ] Submitted before deadline: **18 September 2026, 12:00 PM IST**

---

> **Note:** This file documents the *project requirements/spec*, not Python package dependencies. Python package dependencies will be listed separately in the project's own `requirements.txt` inside the repo (e.g., `langchain`, `faiss-cpu`, `rank_bm25`, `sentence-transformers`, `streamlit`, etc.) once the tech stack is finalized.
