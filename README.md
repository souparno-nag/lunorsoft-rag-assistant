# Lunorsoft RAG Assistant

An AI knowledge assistant that answers questions from your own documents, and
shows its work: every answer carries the excerpts it was drawn from and a
confidence badge saying how much of it was verified against those excerpts.

**Live demo:** <https://lunorsoft-rag-assistant-8j4k2p2azhkqyvg3enabfb.streamlit.app/>

> The demo sleeps when idle. A cold start takes around 25 seconds to wake, and
> the first session after that also downloads the re-ranking model and indexes
> the sample paper — so give the first load a minute before concluding it is
> broken. It opens with *Attention Is All You Need* already indexed; ask a
> question straight away, or upload your own documents.

---

## What it does

Upload one or more PDFs, text or Markdown files. Ask questions in plain
language. The assistant retrieves the passages that actually bear on the
question, answers from those passages only, checks its own answer against them,
and shows you the evidence.

It handles follow-up questions — ask *"why is it scaled?"* after a question
about attention and it resolves what "it" refers to before searching.

When the documents do not contain the answer, it says so rather than
improvising.

---

## Why this is not naive RAG

A naive pipeline chunks by character count, embeds, retrieves the top *k* by
vector similarity, and asks a model to answer. Each of those steps has a
characteristic way of failing. This project addresses five of them, and the
[Results](#results-measured-not-asserted) section shows the measured effect of
each.

| Naive failure | What goes wrong | Remedy here |
| --- | --- | --- |
| Fixed-size chunks | A chunk spans the end of one section and the start of another; retrieved, it answers with two half-topics and cannot be cited | **Structure-aware + semantic chunking** — split at the document's own headings, then at topic shifts within them |
| Vector-only search | The embedding understands the question and returns passages on the *topic* instead of the one containing the literal term asked for | **Hybrid retrieval** — BM25 alongside vectors, fused by Reciprocal Rank Fusion |
| Top-*k* by similarity | The top 5 are chunks *about* the subject, not chunks that *answer* the question | **Cross-encoder re-ranking** — query and chunk scored together, not as two independent vectors |
| Question taken literally | A terse or colloquial question gives retrieval almost nothing to match on | **Query transformation** — several phrasings retrieved in parallel, or a hypothetical answer used as the search probe |
| Model asked to behave | "Only use the context" is a request, not a guarantee | **Grounding check** — a second model rules on each claim; unsupported answers are withdrawn |

---

## Architecture

Two pipelines: documents are indexed once, questions are served against those
indexes.

```
INGESTION                                QUERY
─────────                                ─────
PDF / TXT / MD                           question
    │                                       │
    ▼                                       ▼
extract (pdfplumber → pypdf)            condense follow-up against history
    │                                       │
    ▼                                       ▼
preprocess                              transform (multi-query / HyDE / rewrite)
  de-hyphenate, strip running                │
  heads and footers                          ├──────────────┐
    │                                        ▼              ▼
    ▼                                   vector search    BM25 search
chunk                                        └──────┬───────┘
  split at headings, then at                        ▼
  topic shifts; 64–512 tokens              fuse (RRF) → 20 candidates
    │                                              │
    ├──────────────┬──────────────┐                ▼
    ▼              ▼              ▼         cross-encoder re-rank → top 5
 embeddings    BM25 index   chunks.jsonl            │
    │              │         (metadata)             ▼
    ▼              ▼                          assemble context (token budget)
 Chroma        rank_bm25                            │
                                                    ▼
                                             generate (grounded prompt)
                                                    │
                                                    ▼
                                             grounding check → score + band
                                                    │
                                                    ▼
                                      answer + citations + confidence badge
```

Retrieval is deliberately wide and selection deliberately narrow: 20 candidates
are gathered so recall is the retriever's problem, and 5 survive re-ranking so
precision is the re-ranker's.

The transformed queries are used for retrieval **only**. Re-ranking and
generation work from the question as actually asked, and the conversation
history never becomes context for an answer — it resolves what the question
*means*, never where the answer may come from.

See [`specs/design.md`](specs/design.md) for the full design and
[`specs/requirements.md`](specs/requirements.md) for scope.

---

## Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| UI | Streamlit | Fastest route to a usable interface; chat, file upload and state in one file |
| Orchestration | LangChain | Retriever and LLM abstractions, text splitters |
| Extraction | pdfplumber, with pypdf fallback | Layout fidelity for heading detection; pypdf rescues pages pdfplumber mangles |
| Chunking | Custom structure-aware + semantic splitter | Hand-rolled rather than `SemanticChunker` so character offsets stay exact — page citations depend on them |
| Embeddings | `gemini-embedding-001` (768d) deployed · `all-MiniLM-L6-v2` (384d) local | Remote embedding keeps the deployed container small; local costs nothing and needs no network |
| Vector store | Chroma | Local, persistent, no server to run |
| Keyword index | `rank_bm25` | Exact-term matching that vectors miss |
| Fusion | Reciprocal Rank Fusion (`k=60`) | Needs no score normalisation — cosine similarity and BM25 scores are not comparable |
| Re-ranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | The largest single precision gain; small enough to run locally on the deploy tier |
| LLM | Groq `gpt-oss-120b`; `gpt-oss-20b` for auxiliary calls | Fast, generous free tier. Groq has **no** embeddings API, which is why embeddings come from elsewhere |
| Grounding | LLM-as-judge, with a token-overlap fallback | Verifies the answer instead of trusting the prompt |

Everything runs on free tiers.

---

## Running it locally

Requires **Python 3.12** (pinned in `.python-version`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then add your keys
streamlit run app.py
```

Opens at <http://localhost:8501>.

**Keys.** `GROQ_API_KEY` is required — it drives generation, query
transformation and the grounding judge. `GEMINI_API_KEY` is only needed if you
set `EMBEDDING_PROVIDER=gemini`; the default local path needs no key. Both have
free tiers with no card required ([Groq](https://console.groq.com/keys),
[Gemini](https://aistudio.google.com/app/apikey)).

**First run** downloads two models, about 90 MB each — the local embedding
model and the cross-encoder re-ranker.

**Sample corpus.** `data/` ships with *Attention Is All You Need*, and the app
indexes whatever is in `data/` the first time it starts with an empty index.

Behaviour is tuned in [`config/settings.py`](config/settings.py) — chunk bounds,
`k` values, fusion method, query-transform mode, confidence cutoffs. Switching
`EMBEDDING_PROVIDER` changes the vector width, so the index must be rebuilt;
the app detects the mismatch and says so rather than failing mid-query.
