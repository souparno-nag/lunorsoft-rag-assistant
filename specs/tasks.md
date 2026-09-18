# Lunorsoft AI Developer — Round 1

## Mini AI Knowledge Assistant (RAG) — Task Breakdown

> Derived from `design.md` and `requirements.md`. Tasks are sequenced as a
> **walking skeleton**: a working end-to-end MVP is built first, then each
> differentiation feature is layered on top — so there is always a runnable,
> submittable artifact at every stage.

---

## How to Use This Document

**Status legend:** ☐ not started · ◐ in progress · ☑ done

The **Status** column in each phase table is the single source of truth for
progress. Update a task's status in the same commit that completes it, so the
document never disagrees with the repository.

**Priority tags:**

- **MUST** — required for a strong submission (core + all five differentiators)
- **STRETCH** — attempt only if the MUST set is complete and tested
- **FUTURE** — explicitly deferred (documented as future work if unbuilt)

**Effort:** S (≤30 min) · M (~1–2 hr) · L (2 hr+)

**Columns:** each task lists its ID, status, description, dependencies (other
task IDs), priority, effort, the file(s) it touches, and concrete acceptance
criteria.

**Deadline:** 18 September 2026, 12:00 PM IST. See the **Critical Path** and
**Minimum Viable Submission** sections at the end before starting.

---

## Phase 0 — Project Setup & Scaffolding

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T0.1 | ☑ | Create repo + folder structure exactly as in `design.md §9` | — | MUST | S | full tree |
| T0.2 | ☑ | Set up Python venv; create `requirements.txt` (pip packages) | T0.1 | MUST | S | `requirements.txt` |
| T0.3 | ☑ | Create `.env.example` with `GROQ_API_KEY`, `GEMINI_API_KEY`; add `.env` to `.gitignore` | T0.1 | MUST | S | `.env.example`, `.gitignore` |
| T0.4 | ☑ | Create `config/settings.py` with all tunables (provider flag, models, chunk size/overlap, `k_retrieve`, `k_final`, fusion weights, transform mode, grounding threshold, confidence cutoffs) | T0.1 | MUST | S | `config/settings.py` |
| T0.5 | ☑ | Define data models in `src/models/schemas.py`: `Chunk`, `RetrievalResult`, `AnswerEnvelope` (fields per `design.md §7`) | T0.1 | MUST | S | `src/models/schemas.py` |
| T0.6 | ☑ | Obtain free API keys: Groq (LLM) + Gemini (embeddings); verify each with a one-line smoke test | — | MUST | S | scratch |

**Definition of done (Phase 0):** repo scaffolded, both keys verified working,
config and schema stubs importable without error.

**pip packages to include (T0.2):** `langchain`, `langchain-community`,
`langchain-groq`, `langchain-google-genai`, `sentence-transformers`,
`chromadb`, `rank_bm25`, `pdfplumber`, `pypdf`, `streamlit`, `python-dotenv`,
`tiktoken`. (Local-embeddings path also pulls `torch` via sentence-transformers.)

---

## Phase 1 — Ingestion & Indexing (MVP baseline)

> Goal: turn documents into a searchable index. Start with **basic** chunking;
> it is upgraded to advanced chunking in Phase 3.

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T1.1 | ☑ | Document loader — accept one PDF; return normalized `RawDocument`; carry `doc_id` + `source_file` from the start (enables multi-doc later) | T0.5 | MUST | S | `src/ingest/loader.py` |
| T1.2 | ☑ | Text extraction with `pdfplumber` (PyPDF fallback); capture **page numbers** as metadata | T1.1 | MUST | M | `src/ingest/extract.py` |
| T1.3 | ☑ | Preprocessing: whitespace normalization, de-hyphenation, strip repeated headers/footers | T1.2 | MUST | S | `src/ingest/extract.py` |
| T1.4 | ☑ | **Basic** chunking: `RecursiveCharacterTextSplitter` with size+overlap from config; emit `Chunk` objects with full metadata | T1.2, T0.5 | MUST | S | `src/ingest/chunker.py` |
| T1.5 | ☑ | Embedding module with provider switch: Gemini (`langchain-google-genai`) **and** local `all-MiniLM-L6-v2`; batching + retry/backoff on 429 | T0.4 | MUST | M | `src/index/embeddings.py` |
| T1.6 | ☑ | Vector store wrapper (Chroma default), persisted to `storage/`; store vector + `chunk_id` + metadata payload | T1.4, T1.5 | MUST | M | `src/index/vector_store.py` |

**Definition of done (Phase 1):** running ingestion on a sample PDF populates a
persisted Chroma index; a manual similarity query returns sensible chunks with
correct page/source metadata.

**Note:** Groq has **no embeddings API** — embeddings come only from
Gemini or local MiniLM (per `design.md §8.1`). Do not wire embeddings to Groq.

---

## Phase 2 — Basic Query Pipeline + UI (walking skeleton — MVP end-to-end)

> Goal: a usable app that answers questions from the document using **single
> vector retrieval** + grounded generation. This is the first submittable state
> and satisfies core requirements 1–9 before any differentiators are added.

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T2.1 | ☑ | Grounded generator: Groq LLM (`langchain-groq`); strict prompt = answer only from context, say "not found" if insufficient, cite sources | T1.6 | MUST | M | `src/generate/generator.py` |
| T2.2 | ☑ | Minimal query pipeline: embed query → vector top-k → assemble context (respect token budget) → generate → return `AnswerEnvelope` | T2.1, T1.6 | MUST | M | `src/query/pipeline.py` |
| T2.3 | ☑ | Streamlit UI v1: upload PDF → trigger ingestion → ask question → show answer | T2.2 | MUST | M | `app.py` |
| T2.4 | ☑ | Guard: block querying before any document is indexed; show clear prompt to upload first | T2.3 | MUST | S | `app.py` |
| **T2.5** | ☑ | **CHECKPOINT — commit + tag a working MVP.** Test full flow in an incognito window. | T2.3 | MUST | S | — |

**Definition of done (Phase 2):** upload → ask → grounded answer works end to
end locally. **This is the fallback submission if time runs out.**

---

## Phase 3 — Advanced Chunking *(Differentiation E)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T3.1 | ☑ | Structure-aware split: detect section/heading boundaries (font-size/regex heuristics) so chunks don't straddle sections; capture `section_header` metadata | T1.2 | MUST | M | `src/ingest/chunker.py` |
| T3.2 | ☑ | Semantic split within sections (LangChain `SemanticChunker`); enforce min/max token bounds + small overlap | T3.1, T1.5 | MUST | M | `src/ingest/chunker.py` |
| T3.3 | ☐ | Swap pipeline to advanced chunker; re-index; sanity-check chunk coherence vs basic chunking | T3.2, T1.6 | MUST | S | `src/ingest/chunker.py` |

**Definition of done (Phase 3):** chunks align to topic/section boundaries;
`section_header` populated for citations. Keep a note of the before/after
difference for the README.

---

## Phase 4 — Hybrid Search *(Differentiation A)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T4.1 | ☐ | Build BM25 / keyword index over the same chunk set (`rank_bm25` / `BM25Retriever`), persisted to `storage/` | T3.3 | MUST | M | `src/index/keyword_index.py` |
| T4.2 | ☐ | Hybrid retriever: run dense (vector) + sparse (BM25) in parallel | T4.1, T1.6 | MUST | M | `src/query/hybrid_retriever.py` |
| T4.3 | ☐ | Fuse results with **Reciprocal Rank Fusion** (default); weighted fusion available via config; output `k_retrieve` candidates | T4.2 | MUST | M | `src/query/hybrid_retriever.py` |
| T4.4 | ☐ | Wire hybrid retriever into `pipeline.py` (replaces single vector retrieval) | T4.3, T2.2 | MUST | S | `src/query/pipeline.py` |

**Definition of done (Phase 4):** an exact-keyword query that pure vector search
missed now surfaces the right chunk. Save one such example for the README.

---

## Phase 5 — Re-ranking *(Differentiation B)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T5.1 | ☐ | Cross-encoder reranker (local `ms-marco-MiniLM-L-6-v2`); score each candidate jointly with the query | T4.3 | MUST | M | `src/query/reranker.py` |
| T5.2 | ☐ | Keep top `k_final` after reranking; populate `rerank_score` on `RetrievalResult` | T5.1 | MUST | S | `src/query/reranker.py` |
| T5.3 | ☐ | Insert reranking between hybrid retrieval and context assembly in `pipeline.py` | T5.2, T4.4 | MUST | S | `src/query/pipeline.py` |

**Definition of done (Phase 5):** final context is the reranked top `k_final`;
verify precision improves on a query where raw fusion returned noisy hits.

---

## Phase 6 — Query Transformation *(Differentiation C)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T6.1 | ☐ | Query rewrite (Groq LLM normalizes/clarifies the raw query) | T2.1 | MUST | S | `src/query/transform.py` |
| T6.2 | ☐ | Multi-query: generate 2–3 paraphrases; retrieve per paraphrase; pool + de-duplicate by `chunk_id` | T6.1, T4.3 | MUST | M | `src/query/transform.py` |
| T6.3 | ☐ | HyDE (optional): embed a hypothetical answer for dense search; toggle via config | T6.1 | STRETCH | M | `src/query/transform.py` |
| T6.4 | ☐ | Wire transform stage at the front of `pipeline.py`; record `used_query_transform` in the envelope | T6.2, T4.4 | MUST | S | `src/query/pipeline.py` |

**Definition of done (Phase 6):** a vague question retrieves better with
transformation on than off. Transform mode is config-switchable for the demo.

---

## Phase 7 — Grounding / Faithfulness Check *(Differentiation D)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T7.1 | ☐ | LLM-as-judge (Groq): score whether answer claims are supported by retrieved context → `faithfulness_score` + verdict | T2.1 | MUST | M | `src/generate/grounding.py` |
| T7.2 | ☐ | Overlap-heuristic fallback (cheaper) when judge is unavailable | T7.1 | MUST | S | `src/generate/grounding.py` |
| T7.3 | ☐ | Map score → confidence band (green/amber/red) using config cutoffs; on low score, downgrade/refuse per `design.md §5.6` | T7.1, T0.4 | MUST | S | `src/generate/grounding.py` |
| T7.4 | ☐ | Add grounding step after generation in `pipeline.py`; populate `faithfulness_score` + `confidence` | T7.3, T2.2 | MUST | S | `src/query/pipeline.py` |

**Definition of done (Phase 7):** every answer carries a faithfulness score;
an answer unsupported by the docs is flagged/refused rather than shown as
confident. Directly satisfies core requirement #8.

---

## Phase 8 — Source Citations *(Bonus)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T8.1 | ☐ | Populate `citations[]` (source_file · page · section · snippet) from the chunks used | T5.3 | MUST | S | `src/query/pipeline.py` |
| T8.2 | ☐ | UI: citations expander under each answer showing the exact supporting snippets | T8.1, T2.3 | MUST | S | `app.py` |
| T8.3 | ☐ | UI: confidence badge rendered from `confidence` band | T7.4, T2.3 | MUST | S | `app.py` |

**Definition of done (Phase 8):** each answer shows traceable citations + a
confidence badge.

---

## Phase 9 — Multiple Documents *(Bonus)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T9.1 | ☐ | UI: upload multiple files; list indexed docs; "rebuild index" action | T2.3 | MUST | M | `app.py` |
| T9.2 | ☐ | Confirm indexes + retrieval span all docs; citations disambiguate across files (relies on `doc_id`/`source_file` from T1.1) | T9.1, T4.3 | MUST | S | `src/index/*` |

**Definition of done (Phase 9):** questions answer correctly across a
multi-document corpus with file-attributed citations.

---

## Phase 10 — Conversation History *(Stretch)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T10.1 | ☐ | Rolling message history in Streamlit session state | T2.3 | STRETCH | S | `app.py` |
| T10.2 | ☐ | History-aware query rewrite: condense follow-up + prior turns into a standalone question before retrieval (retrieval still grounds on docs) | T10.1, T6.1 | STRETCH | M | `src/query/transform.py` |

**Definition of done (Phase 10):** follow-up questions resolve correctly without
losing grounding/faithfulness.

---

## Phase 11 — Deployment *(Stretch)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T11.1 | ☐ | Deploy to **Streamlit Community Cloud** (HF Spaces = fallback). **Not Netlify** — it can't host a Python/Streamlit server (`design.md §12.4`) | T2.5 | STRETCH | M | repo |
| T11.2 | ☐ | Set embedding provider = **Gemini** for deploy (remote, memory-safe on ~1 GB tier); keep reranker local | T11.1, T1.5 | STRETCH | S | `config/settings.py` |
| T11.3 | ☐ | Configure secrets (`GROQ_API_KEY`, `GEMINI_API_KEY`) via platform secret store; never commit keys | T11.1 | STRETCH | S | platform |
| T11.4 | ☐ | Handle cold-start storage: ship a small prebuilt sample index or rebuild on start | T11.1 | STRETCH | M | repo |
| T11.5 | ☐ | Test the live link in an incognito window (brief requirement) | T11.1 | STRETCH | S | — |

**Definition of done (Phase 11):** public link loads and answers in a clean
browser session. If unstable near the deadline, fall back to local demo +
screen recording.

---

## Phase 12 — Evaluation Harness *(Future)*

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T12.1 | ☐ | Curate a small Q/A set over the sample docs | T2.5 | FUTURE | M | `eval/` |
| T12.2 | ☐ | Run same questions through naive baseline vs full pipeline; tabulate deltas | T12.1 | FUTURE | M | `eval/` |
| T12.3 | ☐ | RAGAS metrics: faithfulness, answer relevance, context precision/recall | T12.2 | FUTURE | L | `eval/` |

**Definition of done (Phase 12):** quantitative before/after table for the
README. If unbuilt, document as future work (per `requirements.md §4`).

---

## Phase 13 — Documentation & Submission

| ID | Status | Task | Deps | Prio | Eff | Files |
| --- | --- | --- | --- | --- | --- | --- |
| T13.1 | ☐ | Write `README.md`: what it does, architecture (from `design.md`), **why-beyond-naive-RAG** rationale, tech stack, setup/run steps | all MUST | MUST | M | `README.md` |
| T13.2 | ☐ | Include a **before/after example query** (naive vs full pipeline) using the examples saved in T4.4/T5.3/T6.4 | T13.1 | MUST | S | `README.md` |
| T13.3 | ☐ | **AI tools disclosure** (mandatory, brief §02/§10): list tools used + how | T13.1 | MUST | S | `README.md` |
| T13.4 | ☐ | Gemini unpaid-tier privacy note (submitted content may be used to improve Google products) | T13.1 | MUST | S | `README.md` |
| T13.5 | ☐ | Limitations / future work (incl. eval harness if unbuilt) | T13.1 | MUST | S | `README.md` |
| T13.6 | ☐ | Verify repo is public; all links open in incognito; final `requirements.txt` + `.env.example` present | T13.1 | MUST | S | repo |
| T13.7 | ☐ | (If not deployed) record a short screen-recording demo as the live-link substitute | T2.5 | MUST* | S | link |
| T13.8 | ☐ | Submit via official Lunorsoft Round 1 form before deadline | T13.6 | MUST | S | — |

*T13.7 is MUST only if Phase 11 (deployment) is not completed.

**Submission checklist (brief §09):**

- [ ] Assignment complete for AI Developer / Option 1
- [ ] GitHub repo public & accessible
- [ ] `requirements.txt` + `.env.example` included
- [ ] Working demo link **or** screen recording
- [ ] README with approach + architecture
- [ ] Technologies/tools listed
- [ ] AI tools disclosed
- [ ] Links tested in incognito
- [ ] Submitted before 18 Sept 2026, 12:00 PM IST

---

## Critical Path

The shortest route to a **complete, differentiated** submission:

```md
T0.* → T1.* → T2.* (MVP CHECKPOINT)
     → T3.* → T4.* → T5.* → T6.(1,2,4) → T7.*
     → T8.* → T9.*
     → T13.* (docs + submit)
```

Everything in `T6.3` (HyDE), Phase 10 (history), Phase 11 (deploy), and
Phase 12 (eval) is off the critical path — attempt only after the above is
done and committed.

---

## Minimum Viable Submission (if time runs short)

Ship in this order of value; stop wherever the clock forces you, but always
leave a **committed, runnable** state:

1. **Floor:** Phases 0–2 (MVP: upload → grounded answer + Streamlit UI) → this
   alone satisfies all nine core requirements and is submittable.
2. **+Differentiators (biggest standout gain):** add Phases 3–7 in order. Even
   getting hybrid search (4) + reranking (5) + grounding (7) done clears the
   naive-RAG bar decisively.
3. **+Polish:** Phases 8–9 (citations, multi-doc).
4. **Always:** Phase 13 (README with AI-tools disclosure + a before/after
   example, and a screen recording if not deployed). A clean, well-explained
   MVP beats a broken ambitious build — and the brief (§11) says exactly that.

---

## Traceability (task phases → design/requirements)

| Feature | Phase | Design ref |
| --- | --- | --- |
| Core pipeline (req 1–9) | 1–2 | §4–§5, §11 |
| Advanced chunking (E) | 3 | §4.3 |
| Hybrid search (A) | 4 | §4.6, §5.2 |
| Re-ranking (B) | 5 | §5.3 |
| Query transformation (C) | 6 | §5.1 |
| Grounding/faithfulness (D) | 7 | §5.6 |
| Source citations (bonus) | 8 | §5.7, §12.1 |
| Multiple documents (bonus) | 9 | §12.2 |
| Conversation history (bonus) | 10 | §12.3 |
| Deployment (bonus) | 11 | §12.4 |
| Evaluation harness (future) | 12 | §13 |
| Docs & submission | 13 | §09 (brief) |

---

*End of tasks document.*
