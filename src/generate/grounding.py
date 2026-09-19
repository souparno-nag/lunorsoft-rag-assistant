"""Grounding / faithfulness check (see specs/design.md §5.6).

The generator is *instructed* to answer only from the retrieved context. This
module checks whether it did. That distinction is the whole point: a prompt is
a request, and core requirement #8 — that answers are based on the provided
knowledge source — is not satisfied by having asked politely.

An answer is scored by having a second model read the excerpts and the answer
together and rule on each claim the answer makes.
"""

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.models.schemas import Chunk

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You check whether an answer is supported by the source excerpts it was drawn from.

Break the answer into its individual factual claims and rule on each one separately.

Rules:
- Judge only against the excerpts. Your own knowledge of the subject is irrelevant here: a claim that is true in the world but absent from the excerpts is UNSUPPORTED.
- A claim is SUPPORTED only if the excerpts actually state it or directly entail it. Being merely consistent with the excerpts is not enough.
- Ignore citation markers such as [1] or [2, 3]; they are not claims.
- Ignore hedging, restatements of the question and connecting phrases. Rule on substance only.

Write one line per claim, in exactly this form, and nothing else:

SUPPORTED: <the claim in a few words>
UNSUPPORTED: <the claim in a few words>"""


class GroundingUnavailable(RuntimeError):
    """Raised when the judge cannot be reached or returned nothing usable."""


@dataclass(frozen=True)
class Grounding:
    """The verdict on one answer.

    `method` records how the score was arrived at, because the scores are not
    equally trustworthy and later stages need to know which one they have.
    """

    score: float
    method: str
    supported_claims: int = 0
    unsupported_claims: int = 0

    @property
    def supported(self) -> bool:
        """Whether the answer clears the configured grounding threshold."""
        return self.score >= settings.GROUNDING_THRESHOLD


def judge(
    answer: str, chunks: list[Chunk], *, llm: BaseChatModel | None = None
) -> Grounding:
    """Score an answer by ruling on each of its claims against the context.

    The judge returns a verdict per claim rather than a number, and the score
    is computed from those verdicts here. That is deliberate: language models
    are unreliable at emitting a calibrated score — asked for "a faithfulness
    score between 0 and 1" they reach for 0.8 and 0.9 almost regardless — but
    they are reasonably good at the much smaller question of whether one
    specific claim appears in one specific passage. Aggregating many small
    binary judgements produces a better number than asking for the number.

    It also makes parsing robust and the result explainable: the counts say
    how much of the answer was verified, not merely how confident something
    felt.
    """
    from src.generate.generator import _format_context, get_llm

    if not chunks:
        raise GroundingUnavailable("no context to check the answer against")

    llm = llm or get_llm(settings.GROQ_JUDGE_MODEL, settings.JUDGE_MAX_TOKENS)
    message = (
        f"Source excerpts:\n\n{_format_context(chunks)}\n\n"
        f"Answer to check:\n\n{answer}"
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=_JUDGE_PROMPT), HumanMessage(content=message)]
        )
    except Exception as exc:
        raise GroundingUnavailable(f"the judge could not be reached: {exc}") from exc

    supported, unsupported = _count_verdicts(str(response.content or ""))
    total = supported + unsupported
    if total == 0:
        raise GroundingUnavailable("the judge returned no claim verdicts")

    score = supported / total
    logger.info(
        "Grounding judge: %d/%d claim(s) supported (score %.2f)",
        supported,
        total,
        score,
    )
    return Grounding(
        score=score,
        method="judge",
        supported_claims=supported,
        unsupported_claims=unsupported,
    )


def _count_verdicts(text: str) -> tuple[int, int]:
    """Count SUPPORTED and UNSUPPORTED verdict lines in the judge's reply.

    Matched by prefix rather than by parsing the whole line, so a judge that
    adds a stray heading or trailing remark costs nothing. UNSUPPORTED is
    tested first because it contains SUPPORTED as a substring.
    """
    supported = unsupported = 0
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*• ").upper()
        if stripped.startswith("UNSUPPORTED"):
            unsupported += 1
        elif stripped.startswith("SUPPORTED"):
            supported += 1
    return supported, unsupported
