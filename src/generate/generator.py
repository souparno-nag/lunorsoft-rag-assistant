"""Grounded answer generation (see specs/design.md §5.5).

The model is instructed to answer only from the retrieved context, to cite
which excerpt supports each claim, and to say plainly when the context does
not contain the answer — this is what turns core requirement #8 ("answers are
primarily based on the provided knowledge source") from a hope into something
the prompt actually enforces. The grounding check added in Phase 7 verifies
that it held; this module only asks for it.

Unlike `src/index/embeddings.py`, no custom retry/backoff is layered on here.
There is exactly one provider — Groq is chat-only, so there is no
cross-provider exception-matching problem to solve — and `ChatGroq` sits on
top of the official `groq` SDK, which already retries 429/5xx with backoff
internally before an exception would ever reach this module.
"""

import logging
import re
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.models.schemas import Chunk

logger = logging.getLogger(__name__)

# Returned verbatim whenever the context does not support an answer, so
# callers (the grounding check in Phase 7, the UI) can detect a refusal by an
# exact string match instead of guessing at the model's phrasing.
NOT_FOUND_MESSAGE = "I could not find this in the provided documents."

_SYSTEM_PROMPT = f"""You are a knowledge assistant. You answer questions using ONLY the numbered source excerpts given to you. Follow these rules strictly:

1. Answer using only information contained in the provided sources. Never use outside knowledge, even if you are confident it is correct.
2. After every claim, cite the excerpt(s) it came from using the exact ASCII bracketed tag shown before that excerpt, e.g. [1] or [2, 3]. Use plain square brackets only — never full-width or other bracket characters.
3. If the sources do not contain enough information to answer the question, respond with exactly this sentence and nothing else: "{NOT_FOUND_MESSAGE}"
4. Do not fabricate, guess, or fill gaps with speculation. A partial, honestly-cited answer is better than a complete-sounding one without support.
5. Be concise and answer the question directly."""


# A citation marker written with full-width or CJK brackets instead of ASCII
# ones. GPT-OSS emits these regularly despite _SYSTEM_PROMPT rule 2 forbidding
# them — instructing a model not to do something is not a guarantee that it
# will not, and Phase 8 parses these markers to build the citation list, so the
# answer text has to be normalized rather than merely requested.
#
# Only digits, commas and spaces are allowed between the brackets. That is what
# makes this safe on a corpus that legitimately contains CJK text: 【1, 3】 is
# a citation and is rewritten, 【重要】 is content and is left alone.
_NON_ASCII_CITATION = re.compile(r"[【〔［]\s*([\d\s,]+?)\s*[】〕］]")
# The same marker once normalized, for reading back which excerpts were used.
_ASCII_CITATION = re.compile(r"\[\s*([\d\s,]+?)\s*\]")
_NUMBER = re.compile(r"\d+")


def normalize_citation_markers(text: str) -> str:
    """Rewrite non-ASCII citation brackets as plain square brackets."""
    return _NON_ASCII_CITATION.sub(lambda m: f"[{' '.join(m.group(1).split())}]", text)


def cited_markers(answer: str) -> set[int]:
    """The excerpt numbers an answer refers to, from its [1] / [2, 3] markers.

    Reads back what `_format_context` wrote. This is only reliable because
    `normalize_citation_markers` has already run over the answer: the model
    emits full-width brackets often enough that parsing the raw reply would
    miss most citations on some questions entirely.
    """
    markers: set[int] = set()
    for group in _ASCII_CITATION.findall(answer):
        markers.update(int(number) for number in _NUMBER.findall(group))
    return markers


class GenerationError(RuntimeError):
    """Raised when the LLM cannot be reached at all (bad key, bad model, ...)."""


def generate_answer(
    query: str, chunks: list[Chunk], *, llm: BaseChatModel | None = None
) -> str:
    """Generate an answer to `query`, grounded only in `chunks`.

    `chunks` should already be the final, budget-trimmed context in the order
    they are to be presented — this function does not re-rank, re-order or
    trim them; that is `src/query/pipeline.py`'s job. An empty list is a valid
    input and always yields `NOT_FOUND_MESSAGE`, since there is nothing to
    answer from.
    """
    if not chunks:
        return NOT_FOUND_MESSAGE

    llm = llm or get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Sources:\n\n{_format_context(chunks)}\n\nQuestion: {query}"),
    ]

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        raise GenerationError(f"Failed to generate an answer: {exc}") from exc

    text = normalize_citation_markers(str(response.content or "").strip())
    if not text:
        # GPT-OSS emits reasoning tokens before the answer, and they count
        # against LLM_MAX_TOKENS (see config/settings.py), so a tight budget
        # can exhaust itself on those and return no visible content at all.
        logger.warning("LLM returned an empty response for query: %r", query)
        return NOT_FOUND_MESSAGE
    return text


def get_llm(
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    """Return a configured Groq chat model.

    Defaults to the generation model. The auxiliary stages — query
    transformation, and the grounding judge in Phase 7 — pass the smaller
    model and a tighter token cap, because they share one free-tier budget
    with generation and neither needs the larger model to do its job.
    """
    return _build(
        model or settings.GROQ_MODEL,
        max_tokens or settings.LLM_MAX_TOKENS,
        reasoning_effort,
    )


@lru_cache(maxsize=4)
def _build(
    model: str, max_tokens: int, reasoning_effort: str | None
) -> BaseChatModel:
    from langchain_groq import ChatGroq

    if not settings.GROQ_API_KEY:
        raise GenerationError(
            "GROQ_API_KEY is not set. Add it to .env — see .env.example."
        )

    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return ChatGroq(
        model=model,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=max_tokens,
        **extra,
    )


def _format_context(chunks: list[Chunk]) -> str:
    """Render chunks as numbered, source-tagged excerpts for the prompt.

    The tag numbers are what the model is asked to cite by (see
    `_SYSTEM_PROMPT`). The full source metadata (filename, page, section) is
    shown here so the model can ground its citation in something concrete, even
    though it only needs to echo back the bracket number — `pipeline.py` is
    what turns the chunks actually used into the structured `citations[]` list
    for the UI.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        tag = f"[{i}] {chunk.source_file}"
        if chunk.page_number is not None:
            tag += f", p.{chunk.page_number}"
        if chunk.section_header:
            tag += f', "{chunk.section_header}"'
        parts.append(f"{tag}\n{chunk.chunk_text}")
    return "\n\n".join(parts)
