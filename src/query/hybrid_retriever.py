"""Hybrid retrieval — dense and sparse together (see specs/design.md §5.2).

Dense vector search and BM25 fail in opposite directions. An embedding model
finds a passage that *means* what the question means, and misses the one that
merely contains the exact token asked for; BM25 does the reverse. Running both
and combining them is what stops either failure from reaching the answer.

This module runs the searches, reconciles their results into one candidate
set, and fuses the rankings into a single order. It takes *lists* of queries
rather than one, because query transformation (Phase 6) may hand it several
phrasings of the same question, and because dense and sparse retrieval do not
always want the same text to search with.
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
        """Return up to `k` candidate chunks for one query."""
        return self.retrieve_pooled([query], [query], k=k)

    def retrieve_pooled(
        self,
        dense_queries: list[str],
        sparse_queries: list[str],
        k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve for several queries at once and pool the results.

        This is what multi-query transformation needs: each phrasing is
        retrieved for separately and every ranking is fused together, so a
        chunk that only one phrasing could reach still competes, and a chunk
        several phrasings agree on is rewarded for that agreement.

        Dense and sparse take separate query lists because they do not always
        want the same input — HyDE searches the vector store with a
        hypothetical answer while BM25 keeps the literal question.

        Both sides are asked for `k` of their own per query, so a chunk that
        only one of them can find still reaches the candidate set; the union is
        then cut back to `k`.

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

        dense_rankings = [
            self._vectors.similarity_search(query, k=k) for query in dense_queries
        ]
        sparse_rankings = [
            self._keywords.search(query, k=k) for query in sparse_queries
        ]
        fused = fuse_many(dense_rankings, sparse_rankings)
        logger.info(
            "Hybrid retrieval over %d dense + %d sparse quer(y/ies): "
            "%d + %d hits -> %d fused candidate(s) (%s)",
            len(dense_queries),
            len(sparse_queries),
            sum(len(r) for r in dense_rankings),
            sum(len(r) for r in sparse_rankings),
            len(fused),
            settings.FUSION_METHOD,
        )
        return fused[:k]


def _merge(
    dense: list[RetrievalResult], sparse: list[RetrievalResult]
) -> list[RetrievalResult]:
    """Combine one dense and one sparse ranking. See `_merge_many`."""
    return _merge_many([dense], [sparse])


def _merge_many(
    dense_rankings: list[list[RetrievalResult]],
    sparse_rankings: list[list[RetrievalResult]],
) -> list[RetrievalResult]:
    """Combine every ranking into one candidate list, de-duplicated by id.

    A chunk found by both retrievers, or by several phrasings of the question,
    must arrive as a single candidate carrying its best score from each side —
    otherwise it takes several slots in the candidate set and gets scored
    repeatedly by whatever ranks them next.

    "Best" is the maximum across phrasings rather than the mean: a chunk that
    one phrasing matched strongly and three matched weakly is a chunk that one
    phrasing found, and averaging would punish it for the phrasings that
    happened to miss. How many rankings agreed is already accounted for by
    RRF, which adds a contribution per ranking the chunk appears in.

    Ordering is dense hits first, then sparse-only ones; it carries no meaning
    because `fuse_many` immediately re-orders by fused score. It does decide
    ties, which is why it is deterministic.
    """
    merged: dict[str, RetrievalResult] = {}

    for ranking in dense_rankings:
        for result in ranking:
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = RetrievalResult(
                    chunk=result.chunk, dense_score=result.dense_score
                )
            elif result.dense_score is not None:
                existing.dense_score = max(
                    existing.dense_score or 0.0, result.dense_score
                )

    for ranking in sparse_rankings:
        for result in ranking:
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = RetrievalResult(
                    chunk=result.chunk, bm25_score=result.bm25_score
                )
            elif result.bm25_score is not None:
                existing.bm25_score = max(
                    existing.bm25_score or 0.0, result.bm25_score
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
    """Combine one dense and one sparse ranking, best first."""
    return fuse_many([dense], [sparse], method=method)


def fuse_many(
    dense_rankings: list[list[RetrievalResult]],
    sparse_rankings: list[list[RetrievalResult]],
    method: str | None = None,
) -> list[RetrievalResult]:
    """Combine any number of rankings into one order, best first.

    The method is `settings.FUSION_METHOD`; `method` overrides it, which is
    what lets the two be compared on the same query.
    """
    method = method or settings.FUSION_METHOD
    if method == "rrf":
        scores = _reciprocal_rank_fusion(*dense_rankings, *sparse_rankings)
    elif method == "weighted":
        scores = _weighted_fusion(dense_rankings, sparse_rankings)
    else:
        raise UnknownFusionMethod(
            f"FUSION_METHOD must be 'rrf' or 'weighted', got {method!r}"
        )

    merged = _merge_many(dense_rankings, sparse_rankings)
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
    dense_rankings: list[list[RetrievalResult]],
    sparse_rankings: list[list[RetrievalResult]],
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
    dense_scores = _min_max(_best_scores(dense_rankings, "dense_score"))
    sparse_scores = _min_max(_best_scores(sparse_rankings, "bm25_score"))
    return {
        chunk_id: settings.DENSE_WEIGHT * dense_scores.get(chunk_id, 0.0)
        + settings.SPARSE_WEIGHT * sparse_scores.get(chunk_id, 0.0)
        for chunk_id in dense_scores.keys() | sparse_scores.keys()
    }


def _best_scores(
    rankings: list[list[RetrievalResult]], attribute: str
) -> dict[str, float]:
    """Each chunk's best score across every ranking it appears in."""
    best: dict[str, float] = {}
    for ranking in rankings:
        for result in ranking:
            value = getattr(result, attribute)
            if value is not None:
                best[result.chunk_id] = max(best.get(result.chunk_id, 0.0), value)
    return best


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores into [0, 1]. A list whose scores all agree maps to 1.0."""
    if not scores:
        return {}
    lowest, highest = min(scores.values()), max(scores.values())
    if highest == lowest:
        return {chunk_id: 1.0 for chunk_id in scores}
    span = highest - lowest
    return {chunk_id: (score - lowest) / span for chunk_id, score in scores.items()}
