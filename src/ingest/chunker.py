"""Chunking (see specs/design.md §4.3).

Two stages, deliberately not fixed-size:

1. **Structure-aware** — the document is first cut at its own section
   boundaries, so a chunk never straddles two unrelated sections and every
   chunk can be cited with the heading it came from.
2. **Semantic** — within a section, sentences are embedded and a boundary is
   placed wherever consecutive sentences are unusually dissimilar, so chunks
   break at topic shifts rather than at arbitrary character counts. Min/max
   token guardrails then keep the result from degenerating into one-line
   chunks or runaway ones.

Splitting runs over the whole document rather than page by page. A paragraph
that continues across a page break stays in one chunk that way, where per-page
splitting would cut it in half and leave both halves harder to retrieve. The
cost is that a chunk's character offsets no longer imply its page, so pages are
recovered by mapping offsets back through `_PageIndex`.

Headings are detected from the *shape of the text* alone — numbering, length,
capitalization — and never from font sizes. Font metrics are unavailable on the
pypdf fallback path and meaningless for Markdown and plain text, so a
text-shape rule is the only one that behaves identically on every input the
loader accepts. The known cost is documented on `_heading_candidates`.
"""

import logging
import math
import re
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.models.schemas import Chunk, Page, RawDocument

logger = logging.getLogger(__name__)

# Must match how RawDocument.text joins its pages, or every offset below is
# wrong by the number of preceding pages.
PAGE_SEPARATOR = "\n"


def chunk_documents(documents: Iterable[RawDocument]) -> list[Chunk]:
    """Chunk several documents into one flat list, ready for indexing."""
    return [chunk for document in documents for chunk in chunk_document(document)]


def chunk_document(document: RawDocument) -> list[Chunk]:
    """Split one document into chunks that respect its section structure.

    Every chunk carries the metadata a citation needs: which file it came from,
    which page it starts on, which section it belongs to, and where it sits in
    the document text.
    """
    text = document.text
    if not text.strip():
        return []

    pages = _PageIndex.build(document.pages)
    sections = split_into_sections(text)
    logger.info(
        "%s: %d section(s) detected (%d with a heading)",
        document.source_file,
        len(sections),
        sum(1 for section in sections if section.header),
    )

    spans_by_section = _plan_spans(text, sections)

    chunks: list[Chunk] = []
    for section, spans in zip(sections, spans_by_section):
        for start, end in spans:
            body = text[start:end].strip()
            if not body:
                continue
            page_number = pages.page_for_span(start, end)
            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document.doc_id}_p{page_number}"
                        f"_s{section.index}_{len(chunks):04d}"
                    ),
                    doc_id=document.doc_id,
                    source_file=document.source_file,
                    chunk_text=body,
                    char_start=start,
                    char_end=end,
                    token_count=count_tokens(body),
                    page_number=page_number,
                    section_header=section.header,
                )
            )
    return chunks


# --- Structure-aware sectioning (T3.1) --------------------------------------


@dataclass(frozen=True)
class Section:
    """A span of the document text running from one heading to the next.

    `start` points at the heading line itself, not past it, so the heading
    travels with the text it introduces — it is useful context for the reader
    of a citation and useful signal for both retrievers.
    """

    index: int
    header: str | None
    start: int
    end: int


@dataclass(frozen=True)
class _Candidate:
    """A line that might be a heading, before the outline check rules on it."""

    start: int
    text: str
    kind: str
    # The section number as a tuple — (3, 2, 1) for "3.2.1" — or None for a
    # heading that carries no number.
    number: tuple[int, ...] | None


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
# "3", "3.2", "3.2.1" — optionally followed by a dot — then a title.
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(\S.*)$")
_ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 \-—:,'/&()]{2,}$")
# A whole word of three or more letters. The lookarounds matter: without them
# "AAL1 AAL2 AAL3" reads as three words and a table header is mistaken for a
# heading.
_ALPHABETIC_WORD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{3,}(?![A-Za-z0-9])")
# A table-of-contents entry, by its two universal tells: dot leaders, or a
# title that ends in the page number it points at.
_CONTENTS_LINE = re.compile(r"\.{3,}|\s\d+\s*$")

_HEADING_MAX_WORDS = 12
_HEADING_MAX_CHARS = 90
# A document does not have a section 10062; a bibliography entry does start
# that way. Bounds the top-level section number to something a real outline
# could reach.
_HEADING_MAX_SECTION_NUMBER = 99
# Words no heading ends on — these mark a line that has been cut off mid-clause
# rather than a title.
_HEADING_BAD_TAIL = frozenset(
    {"and", "or", "the", "of", "a", "an", "with", "to", "for", "is", "are"}
)


def split_into_sections(text: str) -> list[Section]:
    """Cut `text` at its detected headings.

    A document with no detectable outline — one that leans on indentation or
    font size to mark its headings — comes back as a single unheaded section.
    That is the intended degradation, not a failure: the semantic splitter
    still finds topic boundaries within it, and the BM25 index added in Phase 4
    covers exact-term lookup regardless of structure.
    """
    headings = _outline(_heading_candidates(text))

    if not headings:
        return [Section(index=0, header=None, start=0, end=len(text))]

    sections: list[Section] = []
    # Front matter — an abstract, a title block — precedes the first heading
    # and would otherwise be dropped entirely.
    if headings[0].start > 0:
        sections.append(
            Section(index=0, header=None, start=0, end=headings[0].start)
        )

    for position, heading in enumerate(headings):
        end = (
            headings[position + 1].start
            if position + 1 < len(headings)
            else len(text)
        )
        sections.append(
            Section(
                index=len(sections),
                header=heading.text,
                start=heading.start,
                end=end,
            )
        )
    return sections


def _heading_candidates(text: str) -> list[_Candidate]:
    """Every line whose shape could make it a heading.

    Deliberately generous — the outline check below is what turns this into a
    decision. Three families are recognized: Markdown ATX headings, numbered
    sections, and ALL-CAPS lines.

    **Known limitation.** A heading that is neither numbered nor capitalized,
    and announces itself only by sitting unindented in a larger font — an
    RFC's "Abstract", a book's chapter title — is not detectable here, because
    `extract.preprocess_pages` strips the layout whitespace that would be the
    only remaining evidence. Measured on RFC 2616, this misses a handful of
    front-matter headings while finding all 194 numbered sections.
    """
    candidates: list[_Candidate] = []
    for start, line in _iter_lines(text):
        stripped = line.strip()
        if not stripped or "http" in stripped.lower():
            continue
        if _CONTENTS_LINE.search(stripped):
            continue

        if match := _MARKDOWN_HEADING.match(stripped):
            # The hash count is the level, so a Markdown outline orders the
            # same way a numbered one does.
            candidates.append(
                _Candidate(start, match.group(2).strip(), "markdown", None)
            )
            continue

        if len(stripped) > _HEADING_MAX_CHARS:
            continue
        words = stripped.split()
        if len(words) > _HEADING_MAX_WORDS:
            continue
        if stripped.endswith((".", ",", ";")) or words[-1].lower() in _HEADING_BAD_TAIL:
            continue

        if match := _NUMBERED_HEADING.match(stripped):
            number = tuple(int(part) for part in match.group(1).split("."))
            title = match.group(2)
            if (
                number[0] <= _HEADING_MAX_SECTION_NUMBER
                and title[:1].isupper()
                and _ALPHABETIC_WORD.search(title)
                # Figure and table captions are numbered and titled exactly
                # like sections are.
                and not title.lower().startswith(("figure", "table", "eq"))
            ):
                candidates.append(_Candidate(start, stripped, "numbered", number))
                continue

        if (
            _ALL_CAPS_HEADING.match(stripped)
            and ":" not in stripped
            and len(_ALPHABETIC_WORD.findall(stripped)) >= 2
        ):
            candidates.append(_Candidate(start, stripped, "caps", None))

    return candidates


def _outline(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep only the candidates that form one coherent outline.

    This is what separates a document's real section headings from the things
    that merely look like them, and it does so without any document-specific
    tuning: a real outline is a long monotone chain of section numbers, while
    a table of contents, an enumerated list inside a paragraph, a chart axis
    and a bibliography entry are each a short chain or none at all. Keeping the
    longest chain therefore keeps the body and discards the imitations.

    Two real documents motivated this. NIST SP 800-63-3 opens with a contents
    table that duplicates its outline eight entries deep, and RFC 2616's
    contents table is *longer* than its detectable body outline — so neither
    "prefer the first" nor "prefer the longest" alone is right, and the dot
    leaders that `_CONTENTS_LINE` rejects are what separate them.

    Unnumbered headings cannot join or break a chain, so they are kept as they
    are; only numbered ones are filtered.
    """
    numbered = [c for c in candidates if c.number is not None]
    if not numbered:
        return candidates

    # Longest chain ending at each candidate, by the usual O(n²) DP. n is the
    # number of numbered lines in a document, a few hundred at most.
    best = [1] * len(numbered)
    previous = [-1] * len(numbered)
    for i in range(len(numbered)):
        for j in range(i):
            if _follows(numbered[j].number, numbered[i].number) and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                previous[i] = j

    end = max(range(len(numbered)), key=lambda i: best[i])
    chain: set[int] = set()
    while end != -1:
        chain.add(numbered[end].start)
        end = previous[end]

    return [c for c in candidates if c.number is None or c.start in chain]


def _follows(earlier: tuple[int, ...], later: tuple[int, ...]) -> bool:
    """Whether `later` can directly follow `earlier` in one outline."""
    if later <= earlier:
        return False
    # Going a level deeper has to start at 1: 3.2 may be followed by 3.2.1,
    # never by 3.2.7. This is what stops a page number or a measurement from
    # attaching itself to the chain.
    if len(later) > len(earlier):
        return (
            later[: len(earlier)] == earlier
            and later[len(earlier) :] == (1,) * (len(later) - len(earlier))
        )
    return True


def _iter_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield `(offset, line)` for each line, offsets into `text`."""
    offset = 0
    for line in text.split("\n"):
        yield offset + len(line) - len(line.lstrip()), line
        offset += len(line) + 1


# --- Splitting within a section (T3.2) --------------------------------------
#
# Boundaries come from the text's own topic structure rather than from a
# character count: cut the section into sentences, embed them, break wherever
# consecutive sentences are unusually dissimilar, then enforce the token
# guardrails and carry a small overlap between neighbours.
#
# Everything is carried as (start, end) offsets into the document text rather
# than as strings, so a chunk's recorded span always reproduces its text
# exactly and `_PageIndex` can still resolve which page to cite.

# A sentence ends at .!? — plus any closing quote or bracket — followed by
# whitespace, or at a blank line. Abbreviations ("e.g.", "et al.") do cut a
# sentence early, but the min-token guardrail rejoins the fragment, so the
# error corrects itself rather than reaching a chunk.
_SENTENCE_BOUNDARY = re.compile(r"""(?<=[.!?])["')\]]*\s+|\n{2,}""")


def _plan_spans(text: str, sections: list[Section]) -> list[list[tuple[int, int]]]:
    """Decide every chunk boundary in the document, section by section.

    Sentence embedding happens here, once for the whole document, rather than
    inside each section: a section is often only a handful of sentences, and
    paying the model's per-call overhead once per section — thirty-nine times
    over on NIST SP 800-63-3 — costs far more than the embedding itself.
    """
    sentences_by_section = [_sentence_spans(text, section) for section in sections]

    if not settings.SEMANTIC_CHUNKING:
        return [_fixed_size_spans(text, section) for section in sections]

    flat = [span for spans in sentences_by_section for span in spans]
    # A lone sentence cannot be dissimilar to anything, so there is nothing to
    # embed and nothing to decide.
    vectors = _embed_sentences(text, flat) if len(flat) > 1 else []

    planned: list[list[tuple[int, int]]] = []
    cursor = 0
    for section, sentences in zip(sections, sentences_by_section):
        section_vectors = vectors[cursor : cursor + len(sentences)]
        cursor += len(sentences)
        planned.append(_split_section(text, section, sentences, section_vectors))
    return planned


def _split_section(
    text: str,
    section: Section,
    sentences: list[tuple[int, int]],
    vectors: list[list[float]],
) -> list[tuple[int, int]]:
    """Split one section into chunk spans."""
    if not sentences:
        return []
    # A section holding nothing but its own heading has no retrievable content
    # of its own; the heading still reaches citations as metadata on the
    # sections nested under it.
    if section.header and text[section.start : section.end].strip() == section.header:
        return []

    groups = _group_by_topic(sentences, vectors) if vectors else [sentences]
    groups = _enforce_token_bounds(text, groups)
    spans = [(group[0][0], group[-1][1]) for group in groups if group]
    return _with_overlap(spans, section)


def _sentence_spans(text: str, section: Section) -> list[tuple[int, int]]:
    """Cut a section into sentence spans, as offsets into the document text."""
    spans: list[tuple[int, int]] = []
    cursor = section.start
    for match in _SENTENCE_BOUNDARY.finditer(text, section.start, section.end):
        if match.end() > cursor:
            spans.append((cursor, match.end()))
            cursor = match.end()
    if cursor < section.end:
        spans.append((cursor, section.end))
    return [span for span in spans if text[span[0] : span[1]].strip()]


def _embed_sentences(text: str, spans: list[tuple[int, int]]) -> list[list[float]]:
    """Embed every sentence in the document with the local model.

    Deliberately not the configured retrieval provider — see
    `settings.CHUNK_EMBED_PROVIDER`. Failure here is not fatal: an empty result
    makes the caller treat each section as one group, which the token
    guardrails then cut to size, so an unavailable model degrades chunk quality
    instead of refusing the document.
    """
    from src.index.embeddings import get_embeddings

    sentences = [text[start:end].strip() for start, end in spans]
    try:
        return get_embeddings(settings.CHUNK_EMBED_PROVIDER).embed_documents(sentences)
    except Exception as exc:
        logger.warning(
            "Semantic chunking unavailable (%s); falling back to token-bounded "
            "splitting within each section",
            exc,
        )
        return []


def _group_by_topic(
    sentences: list[tuple[int, int]], vectors: list[list[float]]
) -> list[list[tuple[int, int]]]:
    """Group consecutive sentences, breaking where the topic shifts.

    The distance between neighbouring sentences is judged against a percentile
    of the distances in this same section, so the cut-off adapts to how much
    the section's own prose moves around rather than imposing one number on
    every document.
    """
    if len(sentences) < 2 or len(vectors) != len(sentences):
        return [sentences]

    distances = [1.0 - _dot(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = _percentile(distances, settings.SEMANTIC_BREAKPOINT_PERCENTILE)

    groups: list[list[tuple[int, int]]] = [[sentences[0]]]
    for i, distance in enumerate(distances):
        if distance > threshold:
            groups.append([])
        groups[-1].append(sentences[i + 1])
    return groups


def _enforce_token_bounds(
    text: str, groups: list[list[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    """Merge groups that are too small and split ones that are too large.

    Semantic boundaries alone produce both: a one-sentence chunk too sparse to
    retrieve on, and a long undifferentiated passage that would crowd the
    context window. Merging runs first, so a fragment left behind by an
    abbreviation is rejoined before anything is measured for splitting.
    """
    merged: list[list[tuple[int, int]]] = []
    for group in groups:
        if merged and _tokens(text, merged[-1]) < settings.CHUNK_MIN_TOKENS:
            merged[-1].extend(group)
        else:
            merged.append(list(group))
    # The final group can still be under the minimum with nothing after it to
    # absorb it, so it folds back into its predecessor instead.
    if len(merged) > 1 and _tokens(text, merged[-1]) < settings.CHUNK_MIN_TOKENS:
        merged[-2].extend(merged.pop())

    bounded: list[list[tuple[int, int]]] = []
    for group in merged:
        if _tokens(text, group) <= settings.CHUNK_MAX_TOKENS:
            bounded.append(group)
            continue
        # Cut at sentence boundaries rather than mid-sentence.
        current: list[tuple[int, int]] = []
        for sentence in group:
            # One "sentence" can exceed the ceiling by itself where the text
            # offers no punctuation to cut at — an ASCII table, a BNF grammar
            # block, a run-on line from a bad extraction. Character-splitting
            # it is what keeps the ceiling a real bound instead of an
            # aspiration: without this, RFC 2616 produced 796-token chunks.
            if _tokens(text, [sentence]) > settings.CHUNK_MAX_TOKENS:
                if current:
                    bounded.append(current)
                    current = []
                bounded.extend([piece] for piece in _hard_split(text, sentence))
                continue
            if current and _tokens(text, current + [sentence]) > settings.CHUNK_MAX_TOKENS:
                bounded.append(current)
                current = []
            current.append(sentence)
        if current:
            bounded.append(current)
    return bounded


def _hard_split(text: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    """Split one over-long sentence by character count — the last resort.

    Reached only when a span has no sentence boundary to cut at, so there is
    no meaning-preserving place to break and an arbitrary one is the honest
    choice. Overlap is left at zero here because `_with_overlap` applies it
    afterwards to every chunk alike.
    """
    start, end = span
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=0,
        add_start_index=True,
    )
    pieces = [
        (
            start + piece.metadata["start_index"],
            start + piece.metadata["start_index"] + len(piece.page_content),
        )
        for piece in splitter.create_documents([text[start:end]])
    ]
    return pieces or [span]


def _with_overlap(
    spans: list[tuple[int, int]], section: Section
) -> list[tuple[int, int]]:
    """Extend each chunk backwards into its predecessor.

    A claim that depends on the sentence before it still retrieves when that
    sentence is carried along. Extending backwards rather than forwards keeps
    the recorded span exact — the chunk text stays `text[start:end]` — and the
    first chunk of a section is left alone, since nothing before it belongs to
    the same section.

    This can push a chunk slightly past CHUNK_MAX_TOKENS. Accepted: the
    overlap is bounded by CHUNK_OVERLAP characters, and it is the context
    budget that trims by rank (specs/design.md §5.4) which actually protects
    the prompt.
    """
    return [
        (start if i == 0 else max(section.start, start - settings.CHUNK_OVERLAP), end)
        for i, (start, end) in enumerate(spans)
    ]


def _fixed_size_spans(text: str, section: Section) -> list[tuple[int, int]]:
    """Fixed-size fallback, used when SEMANTIC_CHUNKING is off.

    Still structure-aware, since it runs inside a single section. That makes it
    the honest "before" half of the comparison the README needs: same
    sectioning, boundaries chosen by character count instead of by meaning.
    """
    body = text[section.start : section.end]
    if not body.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,
    )
    spans: list[tuple[int, int]] = []
    for piece in splitter.create_documents([body]):
        start = section.start + piece.metadata["start_index"]
        spans.append((start, start + len(piece.page_content)))
    return spans


def _tokens(text: str, group: list[tuple[int, int]]) -> int:
    return count_tokens(text[group[0][0] : group[-1][1]])


def _dot(a: list[float], b: list[float]) -> float:
    """Cosine similarity — vectors arrive unit length (see index/embeddings)."""
    return sum(x * y for x, y in zip(a, b))


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile, so numpy is not needed for one number."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


# --- Tokens ------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Count tokens the way the generation model will count them."""
    return len(_encoding().encode(text, allowed_special=set()))


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Load the tokenizer once; building one costs more than using it."""
    return tiktoken.get_encoding(settings.TOKENIZER_ENCODING)


# --- Page mapping ------------------------------------------------------------


class _PageIndex:
    """Maps a character offset in the document text back to its page number.

    Built once per document, so resolving a chunk's page is a binary search
    rather than a scan over the pages.
    """

    def __init__(self, starts: list[int], ends: list[int], numbers: list[int]) -> None:
        self._starts = starts
        self._ends = ends
        self._numbers = numbers

    @classmethod
    def build(cls, pages: list[Page]) -> "_PageIndex":
        starts: list[int] = []
        ends: list[int] = []
        numbers: list[int] = []
        cursor = 0
        for page in pages:
            starts.append(cursor)
            ends.append(cursor + len(page.text))
            numbers.append(page.page_number)
            cursor += len(page.text) + len(PAGE_SEPARATOR)
        return cls(starts, ends, numbers)

    def page_for_span(self, start: int, end: int) -> int | None:
        """The page a chunk spanning `[start, end)` should be cited as.

        A chunk that straddles a page break belongs to whichever page holds
        most of its text, not to the page it happens to begin on. The
        difference is not academic: a chunk can open with the last eight
        characters of one page and spend the remaining nine hundred on the
        next, and citing the first page sends the reader somewhere the answer
        demonstrably is not.
        """
        if not self._starts:
            return None

        index = max(0, bisect_right(self._starts, start) - 1)
        best_page = self._numbers[index]
        best_overlap = -1
        while index < len(self._starts) and self._starts[index] < end:
            overlap = min(end, self._ends[index]) - max(start, self._starts[index])
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = self._numbers[index]
            index += 1
        return best_page
