"""Grounding / faithfulness check (see specs/design.md §5.6).

The generator is *instructed* to answer only from the retrieved context. This
module checks whether it did. That distinction is the whole point: a prompt is
a request, and core requirement #8 — that answers are based on the provided
knowledge source — is not satisfied by having asked politely.

An answer is scored by having a second model read the excerpts and the answer
together and rule on each claim the answer makes. When that model cannot be
reached, a much cruder token-overlap heuristic stands in, so an answer is never
served with no grounding signal at all.
"""

import logging
import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.models.schemas import Chunk, Confidence

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


# Citation markers are not claims, and their numbers would otherwise count as
# content words that the context happens not to contain.
_CITATION_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

# Enough function words to stop "the", "of" and "is" from inflating an overlap
# score towards 1 for any answer whatsoever. Deliberately short: the heuristic
# is a floor, and a longer list would imply a precision it does not have.
_FUNCTION_WORDS = frozenset(
    """a an and are as at be been but by for from had has have in into is it its
    of on or that the this to was were which with not no than then they their
    these those there here when where who whom how what why can could may might
    must shall should will would do does did done using used use also such each
    any all both more most other some only own same so too very s t don now""".split()
)


# Replaces an answer the judge found largely unsupported. Phrased as the
# system's own failure to find support rather than as a claim about the
# documents, because "not supported by the retrieved excerpts" and "not in the
# documents" are different statements and only the first one is known.
LOW_SUPPORT_MESSAGE = (
    "I couldn't find enough support in the documents for a confident answer."
)


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

    @property
    def confidence(self) -> Confidence:
        """The band shown to the user, from the configured cutoffs.

        An overlap score can never earn the top band. The measure is not
        capable of justifying it: T7.2 measured it awarding a perfect 1.00 to
        an answer whose one factual claim was wrong, because the wrong value's
        digits appeared elsewhere in the excerpts. A badge is a statement to a
        user about how far the answer was checked, and "high" would overstate
        what a word-overlap count establishes.
        """
        band = confidence_for(self.score)
        if self.method != "judge" and band == "high":
            return "medium"
        return band

    @property
    def should_downgrade(self) -> bool:
        """Whether the answer should be replaced by LOW_SUPPORT_MESSAGE.

        Only a judge verdict can trigger this. Withdrawing an answer is a
        destructive act — the user loses a reply that may well have been
        correct — and the overlap heuristic is too blunt to justify it: it
        scores on vocabulary, so a correct answer phrased in the asker's words
        rather than the document's can score low without being wrong. A weak
        overlap score lowers the badge and says so; it does not censor.
        """
        return self.method == "judge" and not self.supported


def check_grounding(
    answer: str, chunks: list[Chunk], *, llm: BaseChatModel | None = None
) -> Grounding:
    """Score an answer, preferring the judge and falling back to overlap.

    The fallback exists because the judge is an LLM call on a shared free-tier
    budget — Phase 6 hit that limit for real — and an answer served with no
    grounding signal at all would quietly defeat the purpose of having one.
    """
    try:
        return judge(answer, chunks, llm=llm)
    except GroundingUnavailable as exc:
        logger.warning("Grounding judge unavailable (%s); falling back to overlap", exc)
        return overlap(answer, chunks)


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


def overlap(answer: str, chunks: list[Chunk]) -> Grounding:
    """Score an answer by how much of its vocabulary appears in the context.

    The cheap fallback: no model, no network, no tokens. It asks a much weaker
    question than the judge does — *are the words of this answer drawn from
    the sources* rather than *are its claims supported by them* — and the gap
    between those two questions is where its limits live.

    What it catches reliably is the failure that matters most: an answer that
    has wandered off the documents entirely, into the model's own knowledge,
    brings vocabulary with it that the context does not contain.

    What it cannot catch is a false claim assembled entirely from words that
    are present. "Label smoothing of 0.3" scores no worse than "0.1" when both
    numbers appear somewhere in the excerpts, and a claim with its subject and
    object swapped scores perfectly. It is a floor, not a substitute, and
    `method` records which of the two produced a score so that nothing
    downstream treats them as interchangeable.
    """
    from src.index.keyword_index import tokenize

    answer_words = _content_words(answer)
    if not answer_words:
        # Nothing assertive to check — a refusal, or a one-word reply. There is
        # no evidence either way, and inventing a low score would be a worse
        # answer than admitting the measure does not apply.
        return Grounding(score=1.0, method="overlap")

    context_words = set()
    for chunk in chunks:
        context_words.update(tokenize(chunk.chunk_text))

    grounded = answer_words & context_words
    score = len(grounded) / len(answer_words)
    logger.info(
        "Grounding overlap: %d/%d content word(s) found in the context (score %.2f)",
        len(grounded),
        len(answer_words),
        score,
    )
    return Grounding(score=score, method="overlap")


def _content_words(text: str) -> set[str]:
    """The words of `text` that carry meaning, for overlap purposes."""
    from src.index.keyword_index import tokenize

    return {
        word
        for word in tokenize(_CITATION_MARKER.sub(" ", text))
        if word not in _FUNCTION_WORDS
    }


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


def confidence_for(score: float) -> Confidence:
    """Map a faithfulness score onto a confidence band (specs/design.md §5.6)."""
    if score >= settings.CONFIDENCE_HIGH_CUTOFF:
        return "high"
    if score >= settings.CONFIDENCE_MEDIUM_CUTOFF:
        return "medium"
    return "low"
