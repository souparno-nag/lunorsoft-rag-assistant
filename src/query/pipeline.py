"""Query pipeline — orchestrates the full query flow (see specs/design.md §5, §9).

The shape as of Phase 6: transform the question into the queries worth
searching for, retrieve with dense and sparse search fused together, re-rank
the candidates with a cross-encoder, assemble a token-budgeted context from the
survivors, generate a grounded answer. The grounding check (Phase 7) slots in
after generation without this module's shape changing.

Retrieval is deliberately wide and selection deliberately narrow: K_RETRIEVE
candidates are gathered so that recall is somebody else's problem, and K_FINAL
survive re-ranking so that precision is.

The transformed queries are used for retrieval and *only* for retrieval.
Re-ranking and generation both work from the question as it was actually
asked, because a paraphrase is a tool for finding candidates, not a better
statement of what the user wants — and under HyDE the "query" is an invented
passage that must never reach either stage.
"""

import logging

from config import settings
from src.generate.generator import generate_answer
from src.index.indexer import Indexer
from src.models.schemas import AnswerEnvelope, Chunk
from src.query.hybrid_retriever import HybridRetriever
from src.query.reranker import rerank
from src.query.transform import transform_query

logger = logging.getLogger(__name__)


def answer_question(query: str, indexer: Indexer | None = None) -> AnswerEnvelope:
    """Answer `query` against the indexed documents, grounded in retrieved chunks.

    `indexer` is injectable for testing; by default one is opened with the
    configured embedding provider. An empty or unindexed corpus is not an error
    here — retrieval returns no results and `generate_answer` turns that into
    an honest "not found" answer rather than raising, per the fail-safe design
    principle (specs/design.md §2, §14). Blocking the query up front when
    nothing is indexed is `app.py`'s job (T2.4), not this function's.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    indexer = indexer or Indexer()
    retriever = HybridRetriever(indexer.vectors, indexer.keywords)

    transformed = transform_query(query)
    results = retriever.retrieve_pooled(
        transformed.dense_queries,
        transformed.sparse_queries,
        k=settings.K_RETRIEVE,
    )
    retrieved_k = len(results)

    selected = rerank(query, results, k=settings.K_FINAL)
    context = _assemble_context(
        [result.chunk for result in selected], settings.MAX_CONTEXT_TOKENS
    )
    answer = generate_answer(query, context)

    return AnswerEnvelope(
        answer=answer,
        used_query_transform=transformed.mode,
        retrieved_k=retrieved_k,
        final_k=len(context),
    )


def _assemble_context(chunks: list[Chunk], max_tokens: int) -> list[Chunk]:
    """Take chunks in ranked order until the token budget would be exceeded.

    `chunks` must already be ordered best-first, as the re-ranker returns
    them; this only decides where to cut, dropping the lowest-ranked chunks
    once the budget is spent (specs/design.md §5.4, §14 "Oversized context").

    In practice the budget rarely binds now that re-ranking hands over only
    K_FINAL chunks — five chunks capped at CHUNK_MAX_TOKENS cannot approach
    MAX_CONTEXT_TOKENS. It stays because it is the guard that holds when those
    numbers are tuned, and because an over-long chunk (see the character-split
    fallback in the chunker) can still arrive larger than expected.

    The first chunk is always kept even if it alone exceeds `max_tokens`: a
    context that slightly overruns the budget is still useful to the LLM, an
    empty one is not.
    """
    selected: list[Chunk] = []
    used_tokens = 0
    for chunk in chunks:
        if selected and used_tokens + chunk.token_count > max_tokens:
            break
        selected.append(chunk)
        used_tokens += chunk.token_count
    return selected
