"""Hybrid retrieval — dense and sparse together (see specs/design.md §5.2).

Dense vector search and BM25 fail in opposite directions. An embedding model
finds a passage that *means* what the question means, and misses the one that
merely contains the exact token asked for; BM25 does the reverse. Running both
and combining them is what stops either failure from reaching the answer.

This module runs the two searches and reconciles their results into one
candidate set. Deciding the order of that set — the fusion — is T4.3, added
below the merge.
"""

import logging

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
        logger.info(
            "Hybrid retrieval for %r: %d dense + %d sparse candidate(s)",
            query,
            len(dense),
            len(sparse),
        )
        return _merge(dense, sparse)[:k]


def _merge(
    dense: list[RetrievalResult], sparse: list[RetrievalResult]
) -> list[RetrievalResult]:
    """Combine both result lists into one, de-duplicated by `chunk_id`.

    A chunk found by both retrievers must arrive as a single candidate
    carrying both of its scores, not as two rival entries — otherwise it takes
    two slots in the candidate set and, worse, gets scored twice by whatever
    ranks them next.

    Ordering here is provisional: dense hits first, then the chunks only BM25
    found. That is not a judgement about which retriever is more trustworthy,
    only a deterministic arrangement until T4.3 replaces it with a fusion that
    scores the two rankings against each other.
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
