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
BM25_PATH = STORAGE_DIR / "bm25.pkl"
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
GROQ_JUDGE_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.0
# GPT-OSS emits reasoning tokens before the answer and they count against this
# budget, so a tight cap returns empty content rather than a truncated answer.
LLM_MAX_TOKENS = 1024

# gemini-embedding-001 over gemini-embedding-2: it still accepts task_type
# (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY), which the LangChain integration sets
# for us. The newer model expects those hints inlined into the text instead.
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_EMBED_DIM = 768
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBED_BATCH_SIZE = 64
EMBED_MAX_RETRIES = 5
EMBED_BACKOFF_SECONDS = 2.0

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNK_MIN_TOKENS = 64
CHUNK_MAX_TOKENS = 512

# Retrieval
K_RETRIEVE = 20
K_FINAL = 5

FusionMethod = Literal["rrf", "weighted"]
FUSION_METHOD: FusionMethod = "rrf"
RRF_K = 60
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5

# Query transformation
QueryTransformMode = Literal["none", "rewrite", "multi_query", "hyde"]
QUERY_TRANSFORM_MODE: QueryTransformMode = "multi_query"
MULTI_QUERY_COUNT = 3

# Generation
MAX_CONTEXT_TOKENS = 6000

# Grounding
GROUNDING_THRESHOLD = 0.50
CONFIDENCE_HIGH_CUTOFF = 0.75
CONFIDENCE_MEDIUM_CUTOFF = 0.50
