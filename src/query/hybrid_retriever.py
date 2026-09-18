"""Hybrid retrieval — dense and sparse together (see specs/design.md §5.2).

Dense vector search and BM25 fail in opposite directions. An embedding model
finds a passage that *means* what the question means, and misses the one that
merely contains the exact token asked for; BM25 does the reverse. Running both
and combining them is what stops either failure from reaching the answer.

This module runs the two searches, reconciles their results into one
candidate set, and fuses the two rankings into a single order.
"""

import logging
from collections import defaultdict

from config import settings
from src.index.keyword_index import KeywordIndex
from src.index.vector_store import VectorStore
from src.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves from the vector store and the keyword index together."""

    def __init__(self, vectors: VectorStore, keywords: KeywordIndex) -> None:
        self._vectors = vectors
        self._keywords = keywords

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        """Return up to `k` candidate chunks drawn from both retrievers.

        Both sides are asked for `k` of their own, so a chunk that only one of
        them can find still reaches the candidate set; the union is then cut
        back to `k`.

        The two searches run one after the other rather than concurrently.
        specs/design.md §5.2 describes them as running "in parallel", but
        threading them buys nothing measurable here: the total is
        `max(dense, sparse)` instead of `dense + sparse`, and the sparse side
        is a regex and an array multiply — around a millisecond — against a
        dense side that is either a local model or a network round trip. The
        saving is the smaller of the two, so it is the millisecond, and it is
        not worth the failure modes that sharing these objects across threads
        would introduce.
        """
        k = k or settings.K_RETRIEVE

        dense = self._vectors.similarity_search(query, k=k)
        sparse = self._keywords.search(query, k=k)
        fused = fuse(dense, sparse)
        logger.info(
            "Hybrid retrieval for %r: %d dense + %d sparse -> %d fused candidate(s) (%s)",
            query,
            len(dense),
            len(sparse),
            len(fused),
            settings.FUSION_METHOD,
        )
        return fused[:k]


def _merge(
    dense: list[RetrievalResult], sparse: list[RetrievalResult]
) -> list[RetrievalResult]:
    """Combine both result lists into one, de-duplicated by `chunk_id`.

    A chunk found by both retrievers must arrive as a single candidate
    carrying both of its scores, not as two rival entries — otherwise it takes
    two slots in the candidate set and, worse, gets scored twice by whatever
    ranks them next.

    Ordering here is dense hits first, then the chunks only BM25 found; it
    carries no meaning, because `fuse` immediately re-orders by fused score.
    It does decide ties, which is why it is deterministic.
    """
    merged: dict[str, RetrievalResult] = {}
    for result in dense:
        merged[result.chunk_id] = RetrievalResult(
            chunk=result.chunk, dense_score=result.dense_score
        )
    for result in sparse:
        if existing := merged.get(result.chunk_id):
            existing.bm25_score = result.bm25_score
        else:
            merged[result.chunk_id] = RetrievalResult(
                chunk=result.chunk, bm25_score=result.bm25_score
            )
    return list(merged.values())


# --- Fusion (T4.3) ----------------------------------------------------------


class UnknownFusionMethod(ValueError):
    """Raised for a FUSION_METHOD that is neither 'rrf' nor 'weighted'."""


def fuse(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    method: str | None = None,
) -> list[RetrievalResult]:
    """Combine two rankings into one, best first.

    The method is `settings.FUSION_METHOD`; `method` overrides it, which is
    what lets the two be compared on the same query.
    """
    method = method or settings.FUSION_METHOD
    if method == "rrf":
        scores = _reciprocal_rank_fusion(dense, sparse)
    elif method == "weighted":
        scores = _weighted_fusion(dense, sparse)
    else:
        raise UnknownFusionMethod(
            f"FUSION_METHOD must be 'rrf' or 'weighted', got {method!r}"
        )

    merged = _merge(dense, sparse)
    for result in merged:
        result.fused_score = scores.get(result.chunk_id, 0.0)
    # Python's sort is stable, so candidates that fuse to the same score keep
    # the deterministic order `_merge` gave them.
    return sorted(merged, key=lambda result: result.fused_score or 0.0, reverse=True)


def _reciprocal_rank_fusion(*rankings: list[RetrievalResult]) -> dict[str, float]:
    """Score each chunk by where it placed, not by what it scored.

    This is the default because it needs no score normalization, and the two
    retrievers' scores are not on speaking terms: cosine similarity is bounded
    in [0, 1] and means one thing, a BM25 score is unbounded and means
    another. Comparing their *positions* sidesteps the question entirely.

    RRF_K flattens the curve near the top. Without it, first place would be
    worth double second place and a single retriever could dominate the fused
    order; with it at 60, the first few ranks are worth almost the same, so
    agreement between the two retrievers matters more than either one's
    confidence about its own winner.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] += 1.0 / (settings.RRF_K + rank)
    return dict(scores)


def _weighted_fusion(
    dense: list[RetrievalResult], sparse: list[RetrievalResult]
) -> dict[str, float]:
    """Score by normalized score rather than by rank, weighted per retriever.

    The alternative to RRF, available for tuning: it keeps the *margin*
    between hits, which rank-based fusion throws away, so a chunk that a
    retriever found far better than its runner-up can say so.

    The price is that each list has to be squeezed into [0, 1] against its own
    minimum and maximum, which makes a chunk's score depend on the company it
    happens to keep — and the weakest hit in each list is normalized to zero,
    discarding its evidence entirely. That is the trade RRF avoids, and why
    RRF is the default.
    """
    dense_scores = _min_max({r.chunk_id: r.dense_score or 0.0 for r in dense})
    sparse_scores = _min_max({r.chunk_id: r.bm25_score or 0.0 for r in sparse})
    return {
        chunk_id: settings.DENSE_WEIGHT * dense_scores.get(chunk_id, 0.0)
        + settings.SPARSE_WEIGHT * sparse_scores.get(chunk_id, 0.0)
        for chunk_id in dense_scores.keys() | sparse_scores.keys()
    }


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores into [0, 1]. A list whose scores all agree maps to 1.0."""
    if not scores:
        return {}
    lowest, highest = min(scores.values()), max(scores.values())
    if highest == lowest:
        return {chunk_id: 1.0 for chunk_id in scores}
    span = highest - lowest
    return {chunk_id: (score - lowest) / span for chunk_id, score in scores.items()}
