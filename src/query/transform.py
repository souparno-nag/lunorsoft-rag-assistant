"""Query transformation (see specs/design.md §5.1).

Retrieval can only find what the question gives it something to match on, and
a question as typed is often a poor search key: it is terse, it is phrased in
the asker's words rather than the document's, or it buries the searchable term
in conversational scaffolding. Every stage downstream — dense search, BM25,
fusion, re-ranking — inherits whatever this one produces.

The modes are chosen by `settings.QUERY_TRANSFORM_MODE` and each hands back a
`TransformedQuery`, which carries the texts to search with rather than a single
rewritten string. Dense and sparse retrieval are listed separately because they
do not always want the same input — see the `hyde` mode.

Every mode degrades to the untransformed query if the LLM call fails. A
transformation is an optimisation; being unable to paraphrase a question is
not a reason to refuse to answer it (specs/design.md §2, "fail safe").
"""

import logging
import re
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings

logger = logging.getLogger(__name__)

# Leading list markers an LLM adds when asked for several lines: "1. ", "- ",
# "* ", "2) ".
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

# Rewriting must not "clean up" the literal tokens a sparse retriever depends
# on. An error code, a header name or an identifier is the most valuable thing
# in the query, and a model asked to clarify will happily turn `Max-Forwards`
# into "the maximum forwards header" — which is more readable and retrieves
# worse. The rewrite and multi-query prompts both say so explicitly. HyDE does
# not use this, and should not: it writes a passage rather than a query, and
# pinning its vocabulary would defeat the point of writing one.
_PRESERVE = (
    "Preserve every specific term exactly as written — names, identifiers, "
    "error codes, header names, numbers, acronyms and any word in code or "
    "quotes. Do not expand, translate or tidy them."
)

_MULTI_QUERY_PROMPT = f"""You generate alternative phrasings of a user's question, so that a document retrieval system has several ways to find the answer.

{_PRESERVE}

Beyond those fixed terms, vary the wording as much as you can: use synonyms, and use the technical vocabulary a document on this subject would be likely to use, even when the asker did not. A phrasing that reaches for the document's own words is the most useful one you can write.

Write one phrasing per line. Do not number them, do not answer the question, and write nothing else."""

_HYDE_PROMPT = """Write a short passage that answers the user's question the way a technical reference document would.

Write two or three sentences of plain declarative prose, in the register of a specification or a research paper. State it directly. Do not hedge, do not say you are uncertain, do not mention that you are guessing, and do not address the reader. Write the passage alone and nothing else."""

_REWRITE_PROMPT = f"""You rewrite a user's question into a single clear, self-contained search query for a document retrieval system.

{_PRESERVE}

Keep the question's meaning and keep it phrased as a question. Do not reduce it to a bare keyword: the rewritten query is embedded for semantic search, and a lone term carries far less meaning than the question it came from. Remove only conversational filler and ambiguity.

Do not answer the question. Do not add information that is not in it. Reply with the rewritten query alone, on one line, with no preamble, quotes or explanation."""


@dataclass(frozen=True)
class TransformedQuery:
    """What retrieval should actually search for.

    `mode` records what ran, not what was requested — a mode whose LLM call
    failed reports itself as the fallback that replaced it, so the envelope
    shown to the user never claims a transformation that did not happen.
    """

    original: str
    mode: str
    dense_queries: list[str] = field(default_factory=list)
    sparse_queries: list[str] = field(default_factory=list)

    @classmethod
    def untransformed(cls, query: str, mode: str = "none") -> "TransformedQuery":
        """Search for exactly what was asked."""
        return cls(
            original=query,
            mode=mode,
            dense_queries=[query],
            sparse_queries=[query],
        )


def transform_query(
    query: str, mode: str | None = None, llm: BaseChatModel | None = None
) -> TransformedQuery:
    """Turn a raw question into the queries retrieval should run.

    `llm` is injectable for testing; by default the configured Groq model is
    used. An unrecognised mode falls back to no transformation rather than
    raising, so a typo in config costs retrieval quality instead of breaking
    the app.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    mode = mode or settings.QUERY_TRANSFORM_MODE
    if mode == "none":
        return TransformedQuery.untransformed(query)
    if mode == "rewrite":
        return _rewrite(query, llm)
    if mode == "multi_query":
        return _multi_query(query, llm)
    if mode == "hyde":
        return _hyde(query, llm)

    logger.warning(
        "Unknown QUERY_TRANSFORM_MODE %r; retrieving with the original query", mode
    )
    return TransformedQuery.untransformed(query)


def _rewrite(query: str, llm: BaseChatModel | None) -> TransformedQuery:
    """Normalize the question into one cleaner search query."""
    lines = _ask(_REWRITE_PROMPT, query, llm)
    if not lines:
        return TransformedQuery.untransformed(query)

    rewritten = lines[0]
    logger.info("Rewrote query %r -> %r", query, rewritten)
    return TransformedQuery(
        original=query,
        mode="rewrite",
        dense_queries=[rewritten],
        sparse_queries=[rewritten],
    )


def _multi_query(query: str, llm: BaseChatModel | None) -> TransformedQuery:
    """Retrieve for several phrasings of the question and pool the results.

    The point is vocabulary, not variety for its own sake. A reader asks
    "whats that code for when you gotta pay"; the document says "Payment
    Required". Rewriting cannot bridge that, because it is forbidden from
    adding what it was not given — but generating several phrasings *invites*
    the model to spend what it knows about the subject, and one of them is
    likely to reach for the term the document actually uses.

    The original question is always kept alongside the paraphrases. It is the
    only phrasing guaranteed to contain the asker's exact words, and pooling
    means an extra ranking can only add candidates, never remove them.
    """
    paraphrases = _ask(_MULTI_QUERY_PROMPT, query, llm)[: settings.MULTI_QUERY_COUNT]
    if not paraphrases:
        return TransformedQuery.untransformed(query)

    queries = [query, *paraphrases]
    logger.info("Multi-query: %d phrasing(s) from %r: %s", len(paraphrases), query, paraphrases)
    return TransformedQuery(
        original=query,
        mode="multi_query",
        dense_queries=queries,
        sparse_queries=queries,
    )


def _hyde(query: str, llm: BaseChatModel | None) -> TransformedQuery:
    """Search the vector store with a hypothetical answer instead of the question.

    A question and its answer are different kinds of text, and an embedding
    model knows it: "What is the maximum path length for self-attention?" and
    "The maximum path length between any two positions is O(1)" do not sit as
    close together as two passages of prose about path length would. HyDE
    exploits that by writing the passage the answer would be, and searching
    with it — comparing a document-shaped probe against documents.

    **The passage may be factually wrong, and that is acceptable.** It is never
    shown to anyone, never enters the generation context, and contributes no
    words to the answer. It is used once, to produce an embedding, and then
    discarded. What it has to get right is register and vocabulary — sounding
    like the document — not fact.

    Only the dense side uses it. BM25 keeps the literal question, because
    matching invented prose term-for-term would retrieve on words the model
    made up, which is the one way a wrong hypothetical could do real damage.
    """
    lines = _ask(_HYDE_PROMPT, query, llm)
    if not lines:
        return TransformedQuery.untransformed(query)

    passage = " ".join(lines)
    logger.info("HyDE passage for %r: %r", query, passage)
    return TransformedQuery(
        original=query,
        mode="hyde",
        dense_queries=[passage],
        sparse_queries=[query],
    )


def _ask(system_prompt: str, query: str, llm: BaseChatModel | None) -> list[str]:
    """Run one transformation call and return its non-empty lines.

    Returns an empty list on any failure, which every caller reads as "use the
    original query". Catching broadly is deliberate: this is an optional
    pre-processing step, and there is no failure of it that should cost the
    user their answer.
    """
    from src.generate.generator import get_llm

    try:
        llm = llm or get_llm(
            settings.GROQ_TRANSFORM_MODEL, settings.TRANSFORM_MAX_TOKENS
        )
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        )
    except Exception as exc:
        logger.warning(
            "Query transformation failed (%s); retrieving with the original query", exc
        )
        return []

    return _clean_lines(str(response.content or ""))


def _clean_lines(text: str) -> list[str]:
    """Strip list markers and quotes from a model's multi-line reply."""
    lines = []
    for raw in text.splitlines():
        line = _LIST_MARKER.sub("", raw).strip().strip('"').strip()
        if line:
            lines.append(line)
    return lines
