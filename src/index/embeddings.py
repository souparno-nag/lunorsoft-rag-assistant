"""Embedding generation (see specs/design.md §4.4 and §8.1).

One switch chooses between two providers that exist for different reasons:

- **gemini** — runs on Google's servers, so it costs the Streamlit container
  almost no memory. That is what makes the deployed app fit the free tier's
  ~1 GB ceiling alongside a local cross-encoder reranker.
- **local** — `all-MiniLM-L6-v2` on CPU, no API key, no rate limit, no network.
  The offline path for development and the demo recording.

Groq is deliberately absent: it serves chat models only and has no embeddings
endpoint at all, so embeddings can never route there.

Both providers are wrapped so that callers get the same guarantees whichever is
selected — batching, retry with backoff, and unit-length vectors.
"""

import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)

# Substrings that mark an error as worth retrying. Matched against the
# exception text because the two providers raise unrelated exception types, and
# the google-genai SDK's own classes have moved between releases — matching on
# the message keeps this working across both without importing either.
_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "too many requests",
    "500",
    "502",
    "503",
    "504",
    "internal error",
    "unavailable",
    "deadline exceeded",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
)


class EmbeddingError(RuntimeError):
    """Raised when embedding fails and retrying will not help."""


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Return the configured embedding model, ready to hand to the vector store.

    The result satisfies LangChain's `Embeddings` interface, so Chroma can use
    it directly.
    """
    return _build(provider or settings.EMBEDDING_PROVIDER)


def embedding_dimension(provider: str | None = None) -> int:
    """Vector width of a provider, without spending an API call to find out.

    The vector store needs this to notice that an index on disk was built with
    a different provider: Chroma cannot mix dimensionalities, so such an index
    has to be rebuilt rather than appended to.
    """
    provider = provider or settings.EMBEDDING_PROVIDER
    if provider == "gemini":
        return settings.GEMINI_EMBED_DIM
    return settings.LOCAL_EMBED_DIM


@lru_cache(maxsize=2)
def _build(provider: str) -> Embeddings:
    """Construct a provider once and reuse it.

    Cached because the local path loads a sentence-transformers model from
    disk, which takes seconds and would otherwise repeat on every Streamlit
    rerun.
    """
    if provider == "gemini":
        inner = _build_gemini()
    elif provider == "local":
        inner = _build_local()
    else:
        raise EmbeddingError(
            f"Unknown embedding provider {provider!r}; expected 'gemini' or 'local'"
        )

    logger.info("Embedding provider: %s (%d dimensions)", provider, embedding_dimension(provider))
    return _ResilientEmbeddings(inner, provider)


def _build_gemini() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not settings.GEMINI_API_KEY:
        raise EmbeddingError(
            "GEMINI_API_KEY is not set, which the 'gemini' embedding provider "
            "requires. Add it to .env, or set EMBEDDING_PROVIDER=local to run "
            "with the offline MiniLM model instead."
        )

    # The integration picks the task type per call — RETRIEVAL_DOCUMENT when
    # embedding chunks, RETRIEVAL_QUERY when embedding a question — which is
    # why settings pins gemini-embedding-001 rather than the newer model that
    # wants those hints written into the text.
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBED_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        output_dimensionality=settings.GEMINI_EMBED_DIM,
    )


def _build_local() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    # Downloads ~90 MB on first use and is cached by huggingface_hub afterwards.
    return HuggingFaceEmbeddings(model_name=settings.LOCAL_EMBED_MODEL)


class _ResilientEmbeddings(Embeddings):
    """Adds batching, retry with backoff and normalization to any provider.

    Ingestion embeds hundreds of chunks in one run, so a single rate-limited
    request must not lose the work already done: texts go out in batches and
    each batch is retried on its own.
    """

    def __init__(self, inner: Embeddings, provider: str) -> None:
        self._inner = inner
        self._provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batches = list(_batched(texts, settings.EMBED_BATCH_SIZE))
        vectors: list[list[float]] = []
        for number, batch in enumerate(batches, start=1):
            if len(batches) > 1:
                logger.info("Embedding batch %d/%d (%d chunks)", number, len(batches), len(batch))
            vectors.extend(
                _with_retry(
                    lambda: self._inner.embed_documents(list(batch)),
                    what=f"embedding batch {number}/{len(batches)}",
                )
            )
        return [_normalize(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = _with_retry(
            lambda: self._inner.embed_query(text), what="embedding the query"
        )
        return _normalize(vector)


def _batched(items: Sequence[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _with_retry[T](call: Callable[[], T], *, what: str) -> T:
    """Run `call`, retrying transient failures with exponential backoff.

    Rate limiting is the expected failure on a free tier, and it is temporary,
    so it is worth waiting out. Anything else — a bad key, a wrong model name —
    fails immediately rather than sleeping through five doomed attempts.
    """
    for attempt in range(settings.EMBED_MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:
            if not _is_transient(exc):
                raise EmbeddingError(f"Failed {what}: {exc}") from exc
            if attempt == settings.EMBED_MAX_RETRIES:
                raise EmbeddingError(
                    f"Failed {what} after {settings.EMBED_MAX_RETRIES} retries: {exc}"
                ) from exc

            # Jittered exponential backoff, so parallel callers that were rate
            # limited together do not all come back at the same instant.
            delay = settings.EMBED_BACKOFF_SECONDS * 2**attempt
            delay += random.uniform(0, delay * 0.1)
            logger.warning(
                "%s failed (%s); retrying in %.1fs [%d/%d]",
                what.capitalize(),
                exc,
                delay,
                attempt + 1,
                settings.EMBED_MAX_RETRIES,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def _is_transient(exc: Exception) -> bool:
    """Whether an error is worth retrying rather than reporting."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True

    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length.

    Necessary rather than cosmetic for Gemini: `gemini-embedding-001` returns
    normalized vectors only at its native 3072 dimensions, and settings ask for
    768, so those come back unnormalized and cosine similarity over them would
    be distorted. Normalizing both providers also makes the vector store's
    distance metric a non-issue, since Euclidean distance between unit vectors
    ranks identically to cosine similarity.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
