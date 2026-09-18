"""Cross-encoder re-ranking (see specs/design.md §5.3).

Retrieval so far has scored the query and each chunk *separately* — the dense
side compares two vectors that were computed without knowledge of each other,
and BM25 counts term overlap. Both are cheap enough to run over a whole corpus,
and both pay for it in precision: they can tell that a chunk is about the right
subject, not that it answers the question.

A cross-encoder reads the query and the chunk **together**, as one input, and
scores how well that specific chunk answers that specific query. It cannot be
used for retrieval — there is no index to build, it would have to run against
every chunk in the corpus — but over the twenty candidates hybrid retrieval
has already narrowed to, it is affordable and it is the largest single
precision gain in the pipeline.

The model is local (`ms-marco-MiniLM-L-6-v2`, ~90 MB) rather than an API. That
is the memory-budget half of specs/design.md §8.1: Gemini embeddings run off
the deployed container so the reranker can have the RAM instead.
"""

import logging
import math
from functools import lru_cache

from config import settings
from src.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """Raised when the cross-encoder model cannot be loaded."""


def score(query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Score every candidate against the query and return them best first.

    Populates `rerank_score` on each `RetrievalResult` in place — including
    the candidates that will be cut, so a caller that wants to inspect the
    whole ranking can.
    """
    if not results:
        return []

    model = _model()
    pairs = [[query, result.chunk.chunk_text] for result in results]
    raw_scores = model.predict(pairs, batch_size=settings.RERANK_BATCH_SIZE)

    for result, raw in zip(results, raw_scores):
        result.rerank_score = _to_probability(float(raw))

    logger.info(
        "Re-ranked %d candidate(s); best %.3f, worst %.3f",
        len(results),
        max(r.rerank_score or 0.0 for r in results),
        min(r.rerank_score or 0.0 for r in results),
    )
    return sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)


@lru_cache(maxsize=1)
def _model():
    """Load the cross-encoder once and reuse it.

    Cached for the same reason the embedding model is: Streamlit re-runs the
    script on every interaction, and loading this from disk takes seconds.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RerankerUnavailable(f"sentence-transformers is not installed: {exc}") from exc

    try:
        model = CrossEncoder(settings.RERANKER_MODEL)
    except Exception as exc:
        # Downloads ~90 MB on first use, so this is most often a machine with
        # no network and a cold huggingface cache.
        raise RerankerUnavailable(
            f"could not load {settings.RERANKER_MODEL}: {exc}"
        ) from exc

    logger.info("Re-ranker loaded: %s", settings.RERANKER_MODEL)
    return model


def _to_probability(raw: float) -> float:
    """Squash a cross-encoder logit into (0, 1).

    `ms-marco-MiniLM` is trained for relevance classification and emits an
    unbounded logit, roughly -11 to +11. The sigmoid is monotonic, so it
    changes no ordering whatsoever — it exists so the number can be shown to a
    user and compared against the other scores in `RetrievalResult`, which are
    all already on a 0-to-1 scale.
    """
    return 1.0 / (1.0 + math.exp(-raw))
