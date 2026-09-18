"""Query pipeline — orchestrates the full query flow (see specs/design.md §5, §9).

The shape as of Phase 4: retrieve with dense and sparse search fused together,
assemble a token-budgeted context, generate a grounded answer. Re-ranking
(Phase 5), query transformation (Phase 6) and the grounding check (Phase 7)
each slot in without this module's shape changing — re-ranking between
retrieval and context assembly, transformation in front of retrieval,
grounding after generation.
"""

import logging

from config import settings
from src.generate.generator import generate_answer
from src.index.indexer import Indexer
from src.models.schemas import AnswerEnvelope, Chunk
from src.query.hybrid_retriever import HybridRetriever

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
    results = retriever.retrieve(query, k=settings.K_RETRIEVE)
    retrieved_k = len(results)

    context = _assemble_context(
        [result.chunk for result in results], settings.MAX_CONTEXT_TOKENS
    )
    answer = generate_answer(query, context)

    return AnswerEnvelope(
        answer=answer,
        retrieved_k=retrieved_k,
        final_k=len(context),
    )


def _assemble_context(chunks: list[Chunk], max_tokens: int) -> list[Chunk]:
    """Take chunks in ranked order until the token budget would be exceeded.

    `chunks` must already be ordered best-first, as the retriever returns
    them; this only decides where to cut, dropping the lowest-ranked chunks
    once the budget is spent (specs/design.md §5.4, §14 "Oversized context").
    There is no re-ranking yet to order by, so the cut is by fused-retrieval
    rank — the same trimming rule, applied to whatever ranking the current
    pipeline stage has produced.

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
