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

---

## Results, measured not asserted

Every differentiator was measured against the pipeline without it. The corpus
for the retrieval numbers is 672 chunks from four structurally different
documents: *Attention Is All You Need*, the RAG paper, NIST SP 800-63-3 (76
pages, with a table of contents and front matter) and RFC 2616 (8,989 lines of
paginated plain text).

### The headline example — hybrid search

> **"What does the 402 status code mean?"**

**Vector search alone** returns five chunks *about status codes*: `6.1.1 Status
Code and Reason Phrase`, `10 Status Code Definitions`, `10.2 Successful 2xx`,
`10.4 Client Error 4xx`. Not one of them contains the string `402`. The correct
chunk is not in the top 20 at all.

**With hybrid search**, RFC 2616's status code registry — the table reading
`"402" ; Section 10.4.3: Payment Required` — is retrieved at rank 10 with
`dense_score = None` and `bm25_score = 11.99`. The dense retriever never found
it; BM25 did.

**With re-ranking**, that chunk moves to **rank 3**, inside the five the
generator actually sees. Without re-ranking it sits at rank 10 and never
reaches the model.

That is the whole argument in one question: the embedding model understood the
question perfectly and answered the topic instead of the question.

### Chunking — chunks that respect section boundaries

A chunk spanning two sections answers with the tail of one topic and the head
of another, and cannot be cited with a section at all.

| Document | Naive: chunks straddling a section | This pipeline |
| --- | --- | --- |
| Attention paper | 18 / 47 (38%) | **0 / 49** |
| RAG paper | 21 / 81 (26%) | **0 / 74** |
| NIST SP 800-63-3 | 39 / 163 (24%) | **0 / 125** |

Section headers are also populated for 96%, 96% and 86% of chunks
respectively, which is what lets a citation say *"p. 8 · 5.4 Regularization"*
instead of just a page number.

### Re-ranking — what reaches the model

Ten literal-term queries with verifiable ground truth, asking whether a correct
chunk is among the five chunks passed to the generator:

| Selection | Correct chunk in final context |
| --- | --- |
| Fused retrieval order | 9 / 10 |
| Cross-encoder order | **10 / 10** |

The baseline is already high, so this is not a dramatic aggregate shift — the
value is in *which* chunk occupies a five-slot context, which the 402 example
above shows concretely.

### Query transformation — vague questions

Fourteen deliberately vague or colloquial questions, three repeats each, scored
by the rank of the best correct chunk after re-ranking:

| Mode | In top 5 | MRR |
| --- | --- | --- |
| No transformation | 13.0 / 14 | 0.792 |
| Rewrite | 12.0 / 14 | **0.720** |
| Multi-query | **14.0 / 14** | **0.857** |
| HyDE | **14.0 / 14** | 0.856 |

Worth reporting the negative result too: **rewriting is worse than doing
nothing**, consistently across all three runs. Asked not to add information it
was not given, it mostly adds a question mark — it cannot bridge *"whats that
code for when you gotta pay"* to *"Payment Required"*, because it has never
seen the document. Multi-query and HyDE can, and do. Multi-query is the
default.

### Grounding — catching an answer that is not supported

Three answers to *"What label smoothing value was used during training?"*,
scored by the judge:

| Answer | Score |
| --- | --- |
| Faithful — "label smoothing of 0.1" | 1.00 (1/1 claims) |
| Mixed — correct value, invented attribution | 0.50 (1/2) |
| Fabricated — wrong value, invented corpus and authors | **0.00 (0/3)** |

End to end, a fabricated answer forced through the pipeline — *"label smoothing
was set to 0.35 … ablations on a held-out Portuguese corpus at Stanford"* —
scored 0/3 and was **withdrawn** in favour of an honest "I couldn't find enough
support", rather than served with a confident badge.

---

## AI tools disclosure

**Claude (Anthropic), used through the Claude Code CLI**, assisted in building
this project. It was used for implementation, debugging, measurement and
documentation — including this README.

How the work was divided:

- **Specification and direction were human.** The project is spec-driven:
  [`specs/requirements.md`](specs/requirements.md) fixes the scope,
  [`specs/design.md`](specs/design.md) the architecture, and
  [`specs/tasks.md`](specs/tasks.md) breaks it into tasks with a status column.
  The order tasks were tackled in, and the decisions at each fork — which
  fusion method, whether to disable re-ranking for deployment, whether
  conversation history needed a database — were made by the developer, with the
  assistant presenting measurements and a recommendation.
- **Implementation was assisted.** Each task was written, run and measured
  against real documents rather than accepted as written. Several designs were
  revised because the measurement contradicted them; the "Results" section
  above reports those outcomes, including the ones that did not favour the more
  elaborate approach.
- **Every claim in this README is from a run that happened**, not an estimate.
  The comparison tables come from scripts executed against the corpora named.

No AI attribution appears in commit messages, by preference — the commit
history is the developer's record of the work, and every commit was reviewed
before it was made.

**Distinct from the above**, the assistant *is* the product: this application
calls Groq (`gpt-oss-120b` for answers, `gpt-oss-20b` for query transformation
and the grounding judge), Google Gemini for embeddings when deployed, and two
local `sentence-transformers` models. Those are the system's runtime
dependencies, listed under [Tech stack](#tech-stack), not tools used to write
it.

---

## Privacy note — Gemini's unpaid tier

The deployed app embeds documents with **Gemini on Google's free tier**, where
**content submitted may be used to improve Google's products**. Paid tiers
carry different terms.

For this project that is an acceptable trade: the sample corpus is a set of
public documents — a published paper, a NIST standard, an RFC — and nothing
confidential passes through it.

It is not acceptable for everyone. **Do not upload confidential, personal or
proprietary documents to the public demo.** If you need to ask questions of
private material, run it locally with the default `EMBEDDING_PROVIDER=local`:
embedding then happens on your own machine with the MiniLM model and no
document text leaves it.

Note that this applies to the embedding provider specifically. Answer
generation always calls Groq, so the *question* and the *retrieved excerpts*
are sent there on every query, including locally. Only the embedding step is
avoidable by switching provider.
