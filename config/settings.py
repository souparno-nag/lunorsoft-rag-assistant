"""Central tunables for the RAG pipeline (see specs/design.md §10)."""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
CHROMA_COLLECTION = "chunks"
# Records which embedding provider built the index on disk. Switching provider
# changes the vector width, and Chroma cannot mix widths in one collection, so
# the mismatch has to be caught and reported rather than discovered as a
# dimension error halfway through a query.
INDEX_META_PATH = STORAGE_DIR / "index_meta.json"
# Every indexed chunk, in full. Serves as both the chunk metadata store of
# design.md §3 and the corpus the BM25 index is rebuilt from. The BM25
# structure itself is not persisted: pickling a rank_bm25 object would tie the
# files on disk to an installed library version, and rebuilding it is a regex
# pass over the corpus.
CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"

# Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Providers
# Switching provider changes the vector dimensionality, so the index must be
# rebuilt from scratch after a change — Chroma cannot mix dimensions.
EmbeddingProvider = Literal["gemini", "local"]

_provider = os.getenv("EMBEDDING_PROVIDER", "local")
if _provider not in ("gemini", "local"):
    raise ValueError(
        f"EMBEDDING_PROVIDER must be 'gemini' or 'local', got {_provider!r}"
    )
EMBEDDING_PROVIDER: EmbeddingProvider = _provider

# Groq is chat-only; embeddings never route here. Llama models 404 on the free
# tier — they are enterprise-gated — so generation uses GPT-OSS. The judge runs
# on the smaller variant to keep the second LLM call off the rate-limit budget;
# raise it to 120b if faithfulness scoring proves unreliable.
# Do not switch these to groq/compound: it has built-in web search and would
# answer from outside the documents, breaking grounding.
GROQ_MODEL = "openai/gpt-oss-120b"
# Auxiliary calls run on the smaller model. Both of them — rewriting a query
# and judging an answer — are short and mechanical, and neither is worth the
# free tier's token budget: that budget is 8000 tokens per minute against the
# 120b model and it is shared with generation. It is reached in practice, not
# in theory. Note that max_tokens counts toward the tokens a request asks for,
# so a generous cap on a three-line paraphrase is charged whether or not it is
# used — which is why the auxiliary calls set their own, much smaller limit.
GROQ_SMALL_MODEL = "openai/gpt-oss-20b"
GROQ_JUDGE_MODEL = GROQ_SMALL_MODEL
GROQ_TRANSFORM_MODEL = GROQ_SMALL_MODEL
LLM_TEMPERATURE = 0.0
# GPT-OSS emits reasoning tokens before the answer and they count against this
# budget, so a tight cap returns empty content rather than a truncated answer.
LLM_MAX_TOKENS = 1024
# A rewritten query, three paraphrases or a HyDE passage all fit comfortably.
# Reasoning tokens count against this too, hence the headroom.
TRANSFORM_MAX_TOKENS = 512

# gemini-embedding-001 over gemini-embedding-2: it still accepts task_type
# (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY), which the LangChain integration sets
# for us. The newer model expects those hints inlined into the text instead.
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_EMBED_DIM = 768
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# MiniLM-L6 is a 384-dimension model. Declared rather than probed so the
# vector store can detect a provider switch without spending an API call.
LOCAL_EMBED_DIM = 384
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBED_BATCH_SIZE = 64
EMBED_MAX_RETRIES = 5
EMBED_BACKOFF_SECONDS = 2.0

# PDF extraction
# pdfplumber is primary because it preserves layout information that the
# structure-aware chunker needs in Phase 3. Its default x_tolerance of 3 merges
# words on LaTeX-produced PDFs whose inter-word gaps are tight (the sample paper
# comes out as "Providedproperattribution"); 2.0 restores the spaces without
# splitting words apart.
PDF_X_TOLERANCE = 2.0
# A page whose alphabetic tokens are mostly very long has lost its spaces, and
# is re-extracted with pypdf. Correctly spaced prose sits near 0.00 by this
# measure and a space-collapsed page near 0.20, so 0.08 separates them safely.
PDF_RUNON_MIN_WORD_LEN = 15
PDF_RUNON_MAX_FRACTION = 0.08

# Repeated header/footer removal. A line is furniture only if it recurs on this
# fraction of the document's pages AND sits in an unbroken run inward from the
# top or bottom of its page, so body text is never at risk. Short documents are
# left alone because a handful of pages cannot establish that a line recurs.
HEADER_FOOTER_MIN_PAGES = 3
HEADER_FOOTER_MIN_FRACTION = 0.6
# Catastrophe guard, not a tuning knob: if furniture removal would take more
# than this share of the document's characters, the detection is assumed to
# have gone wrong and the document is left untouched. Set high on purpose. The
# case worth defending against is a systematic extraction quirk that makes
# every page look identical and would silently delete the entire corpus; a
# document that is merely mostly furniture — a sparse slide deck whose few
# words compete with a footer on every slide — should still be cleaned.
# Measured documents land between 0.1% and 7%.
# Applied once to the whole document rather than per page, so every page is
# cleaned by the same rule — a per-page budget made the outcome depend on how
# much body text a page happened to have, and left furniture behind on the
# shorter pages of the same document.
HEADER_FOOTER_MAX_DOCUMENT_FRACTION = 0.9

# Token counting. o200k_harmony is GPT-OSS's own encoding, so counts match what
# the Groq model actually sees and the context budget in §5.4 is honest rather
# than an estimate from a different tokenizer.
TOKENIZER_ENCODING = "o200k_harmony"

# Chunking. CHUNK_SIZE and CHUNK_OVERLAP are in characters; the first is used
# only by the fixed-size fallback, the second as the overlap carried between
# adjacent semantic chunks. CHUNK_MIN/MAX_TOKENS are the guardrails that keep
# semantic boundaries from producing a one-line chunk or a runaway one.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_MIN_TOKENS = 64
CHUNK_MAX_TOKENS = 512

# Semantic chunking. Sentences are embedded and a chunk boundary is placed
# wherever consecutive sentences are unusually dissimilar, so boundaries land
# at topic shifts instead of at character counts. Turning this off falls back
# to fixed-size splitting *within* each detected section, which is still
# structure-aware — useful for the before/after comparison in the README.
SEMANTIC_CHUNKING = True
# The breakpoint is a percentile of the section's own distances rather than an
# absolute distance, because how far apart "adjacent sentences" sit varies by
# document: prose runs smooth, a specification's clauses jump. Taking the top
# decile adapts the cut-off per section instead of imposing one number on every
# document. Higher = fewer, larger chunks.
SEMANTIC_BREAKPOINT_PERCENTILE = 90
# Chunking always embeds with the local model, whatever EMBEDDING_PROVIDER is
# set to for retrieval. Chunk boundaries never have to share a vector space
# with the index — they are thrown away once the text is split — so sending a
# document's every sentence to Gemini would spend the free-tier quota that
# matters for indexing and querying, and add network latency to ingestion, for
# no gain in boundary quality.
CHUNK_EMBED_PROVIDER = "local"

# Retrieval
K_RETRIEVE = 20
K_FINAL = 5

# Re-ranking. The cross-encoder is the single biggest precision win over top-k
# vector search, and also the slowest stage per candidate, so it runs on the
# K_RETRIEVE fused candidates only and hands K_FINAL to the generator.
# Switching it off falls back to fused-retrieval order, which is what the
# README's before/after comparison needs.
RERANKING = True
RERANK_BATCH_SIZE = 32

FusionMethod = Literal["rrf", "weighted"]
FUSION_METHOD: FusionMethod = "rrf"
RRF_K = 60
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

# Query transformation
QueryTransformMode = Literal["none", "rewrite", "multi_query", "hyde"]
QUERY_TRANSFORM_MODE: QueryTransformMode = "multi_query"
# How many *additional* phrasings multi-query generates. The original question
# is always retrieved for as well, so 3 here means four rankings per retriever.
MULTI_QUERY_COUNT = 3

# Generation
MAX_CONTEXT_TOKENS = 6000

# Grounding
# The judge answers one line per claim, so its budget scales with how many
# claims an answer makes; reasoning tokens count against it too.
JUDGE_MAX_TOKENS = 512
GROUNDING_THRESHOLD = 0.50
CONFIDENCE_HIGH_CUTOFF = 0.75
CONFIDENCE_MEDIUM_CUTOFF = 0.50
