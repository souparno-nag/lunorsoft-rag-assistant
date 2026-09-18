"""PDF text extraction (see specs/design.md §4.2).

pdfplumber is the primary extractor because it preserves the layout detail the
structure-aware chunker will need in Phase 3. pypdf is the fallback, and it is
triggered two ways: when pdfplumber cannot open the file at all, and — per page
— when pdfplumber returns text whose words have run together. That second case
is not hypothetical; it is exactly what the sample paper does, and an
exception-only fallback would never catch it because the degraded text extracts
"successfully".
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from config import settings
from src.models.schemas import Page

logger = logging.getLogger(__name__)

_WORDS = re.compile(r"[A-Za-z]+")


class ExtractionError(ValueError):
    """Raised when no extractor can read the PDF.

    Subclasses `ValueError` so `loader.load_documents` skips the file and
    carries on with the rest of the batch.
    """


def extract_pages(path: str | Path) -> list[Page]:
    """Extract one `Page` per PDF page, numbered from 1.

    Page numbers are 1-based so they match what a reader sees in a PDF viewer,
    since they are carried through chunk metadata into citations.
    """
    path = Path(path)

    try:
        pages = _extract_with_pdfplumber(path)
    except Exception as exc:  # pdfminer raises a wide range of parse errors
        logger.warning("pdfplumber could not read %s (%s); using pypdf", path.name, exc)
        return _extract_with_pypdf(path)

    return _repair_run_together_pages(path, pages)


def _extract_with_pdfplumber(path: Path) -> list[Page]:
    with pdfplumber.open(path) as pdf:
        return [
            Page(
                page_number=number,
                text=page.extract_text(x_tolerance=settings.PDF_X_TOLERANCE) or "",
            )
            for number, page in enumerate(pdf.pages, start=1)
        ]


def _extract_with_pypdf(path: Path) -> list[Page]:
    try:
        reader = PdfReader(str(path))
        return [
            Page(page_number=number, text=page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        raise ExtractionError(f"{path.name} could not be read by pdfplumber or pypdf: {exc}") from exc


def _repair_run_together_pages(path: Path, pages: list[Page]) -> list[Page]:
    """Re-extract with pypdf any page whose words have lost their spaces.

    Done per page rather than per document so a single bad page does not cost
    the layout fidelity of the rest.
    """
    damaged = [i for i, page in enumerate(pages) if _words_ran_together(page.text)]
    if not damaged:
        return pages

    try:
        fallback = _extract_with_pypdf(path)
    except ExtractionError as exc:
        logger.warning("%s: pypdf fallback unavailable (%s); keeping pdfplumber text", path.name, exc)
        return pages

    repaired = 0
    for i in damaged:
        if i < len(fallback) and not _words_ran_together(fallback[i].text):
            pages[i] = fallback[i]
            repaired += 1

    logger.info(
        "%s: %d/%d run-together page(s) re-extracted with pypdf",
        path.name,
        repaired,
        len(damaged),
    )
    return pages


def _words_ran_together(text: str) -> bool:
    """Detect text whose inter-word spaces were dropped during extraction.

    Measured as the share of alphabetic tokens longer than a plausible English
    word. Real prose barely registers; a page of concatenated words is unmistakable.
    """
    words = _WORDS.findall(text)
    if not words:
        return False
    long_words = sum(1 for word in words if len(word) > settings.PDF_RUNON_MIN_WORD_LEN)
    return long_words / len(words) > settings.PDF_RUNON_MAX_FRACTION


# --- Preprocessing (see specs/design.md §4.2) -------------------------------
#
# Applied to every document, PDF or plain text, before chunking. The aim is to
# remove artefacts of the page medium that would otherwise end up embedded,
# retrieved and quoted back in a citation.

# Non-breaking and typographic spaces, which PDFs emit freely and which
# would otherwise survive every "collapse the spaces" pass below.
_UNICODE_SPACES = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]"
)
_HORIZONTAL_RUNS = re.compile(r"[ \t]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
# A hyphen at end of line followed by a lowercase word. Requiring lowercase
# skips the clear-cut compounds such as "Dot-\nProduct"; everything that does
# match is ambiguous and is resolved against _Vocabulary below.
_HYPHEN_BREAK = re.compile(r"([A-Za-z]+)-\n([a-z][A-Za-z]*)")
# A word standing on its own, i.e. not part of a hyphenated compound.
_STANDALONE_WORD = re.compile(r"(?<![\w-])([A-Za-z]{2,})(?![\w-])")
# A compound written with its hyphen intact on one line. The character
# class excludes newlines, so line-broken hyphens cannot match here.
_INLINE_COMPOUND = re.compile(r"(?<![\w-])([A-Za-z]+)-([A-Za-z]+)(?![\w-])")
_DIGITS = re.compile(r"\d+")
_ROMAN_NUMERAL = re.compile(
    r"(?=[ivxlcdm])m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})"
)


def preprocess_pages(pages: list[Page], *, from_layout: bool = True) -> list[Page]:
    """Normalize whitespace, rejoin hyphenated words and drop page furniture.

    `from_layout` says whether the text came out of a paginated layout (a PDF)
    or was written by a human (Markdown, plain text). It matters because the
    two need opposite treatment: indentation in a PDF is an artefact of where
    glyphs sat on the page and should go, whereas indentation in Markdown is
    the author's and carries meaning — flattening it turns a nested code block
    into syntactically wrong code and a table into a row of pipes. Likewise a
    line-final hyphen is typesetting in a PDF and deliberate in authored text.
    """
    pages = [
        Page(
            page_number=page.page_number,
            text=normalize_whitespace(page.text, collapse_layout=from_layout),
        )
        for page in pages
    ]
    pages = strip_repeated_headers_footers(pages)
    if not from_layout:
        return pages

    # Built once from the whole document so that a hyphen broken across a page
    # boundary is judged against the same evidence as the rest, and so the cost
    # is paid once rather than per page.
    evidence = _Vocabulary.from_text("\n".join(page.text for page in pages))
    return [
        Page(
            page_number=page.page_number,
            text=normalize_whitespace(dehyphenate(page.text, evidence)),
        )
        for page in pages
    ]


def normalize_whitespace(text: str, *, collapse_layout: bool = True) -> str:
    """Collapse the whitespace noise that PDF extraction leaves behind.

    With `collapse_layout` off, only unambiguous noise is touched — line
    endings, exotic space characters, trailing spaces and runs of blank lines —
    and the indentation the author wrote is left alone.

    Idempotent either way, so it is safe to run again after other passes have
    edited the text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _UNICODE_SPACES.sub(" ", text)
    if collapse_layout:
        text = _HORIZONTAL_RUNS.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
    else:
        text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def dehyphenate(text: str, evidence: "_Vocabulary | None" = None) -> str:
    """Rejoin words split across a line break by hyphenation.

    Without this, "repre-\\nsentation" is embedded as two fragments and never
    matches a search for "representation".

    The hard part is that a line-final hyphen is ambiguous: it may be a word
    broken by the typesetter ("convolu-\\ntional") or a genuine compound that
    happened to break at its own hyphen ("position-\\nwise"). Joining blindly
    corrupts the second kind — on the sample paper it produced "positionwise",
    "attentionbased" and "sequencealigned", destroying the very terms a reader
    would search for.

    Rather than guess, each case is decided against evidence from the rest of
    the document; see `_Vocabulary.should_join`. `evidence` is normally built
    once per document by `preprocess_pages`, and is derived from `text` itself
    when the function is called on its own.
    """
    evidence = evidence or _Vocabulary.from_text(text)
    return _HYPHEN_BREAK.sub(
        lambda m: m.group(1) + m.group(2)
        if evidence.should_join(m.group(1), m.group(2))
        else f"{m.group(1)}-{m.group(2)}",
        text,
    )


@dataclass(frozen=True)
class _Vocabulary:
    """What the document itself says about how its words are spelled.

    A technical document repeats its own terminology, so the surrounding text
    is a more reliable authority on whether "position-wise" is one word or two
    than any general rule could be — and it needs no dictionary to ship.
    """

    words: frozenset[str]
    compounds: frozenset[tuple[str, str]]

    @classmethod
    def from_text(cls, text: str) -> "_Vocabulary":
        return cls(
            words=frozenset(w.lower() for w in _STANDALONE_WORD.findall(text)),
            compounds=frozenset(
                (left.lower(), right.lower())
                for left, right in _INLINE_COMPOUND.findall(text)
            ),
        )

    def should_join(self, left: str, right: str) -> bool:
        """Decide whether a line-broken hyphen should be dropped or kept."""
        # The strongest evidence: the document writes the joined form
        # elsewhere, so the hyphen was the typesetter's.
        if (left + right).lower() in self.words:
            return True
        # Equally strong in reverse: the document writes the compound with its
        # hyphen elsewhere on a single line.
        if (left.lower(), right.lower()) in self.compounds:
            return False
        # No direct evidence. Two words that each stand alone in the document
        # ("source" and "target") are far more likely to be a compound than a
        # word cut in half, whereas fragments like "convolu" and "tional" are
        # not words anywhere.
        if left.lower() in self.words and right.lower() in self.words:
            return False
        return True


def strip_repeated_headers_footers(pages: list[Page]) -> list[Page]:
    """Drop running heads, footers and sidebars — whatever recurs on most pages.

    Two conditions must both hold, which is what keeps this safe on documents
    it was never tuned against:

    1. The line recurs on most pages, compared with digit runs and roman
       numerals masked, so a footer that is merely the page number ("2", "3",
       ... or "ii", "iii", ...) is recognized as one pattern rather than as
       dozens of unique lines.
    2. The line belongs to an unbroken run inward from the top or bottom of its
       page. The run stops at the first line that is not furniture, so body
       text that happens to repeat can never be reached.

    Walking a run, rather than scanning a fixed number of lines, is what lets
    this handle multi-line furniture: a rotated "available free of charge from"
    sidebar extracts as nine consecutive one-word lines, which no two-line
    window would ever catch.

    Two known limitations, both bounded:

    - Digit masking cannot tell a page number from another short numbered line,
      so a heading like "Problem 1" sitting at a page edge on most pages is
      removed along with the furniture. Masking is still worth it, because it
      is what collapses "2", "3", "4", ... into a single pattern.
    - A running head that varies with the chapter ("3 Scheduling", then
      "4 Memory") never recurs often enough to reach the threshold, so it
      survives.
    """
    if len(pages) < settings.HEADER_FOOTER_MIN_PAGES:
        return pages

    lines_per_page = [page.text.split("\n") for page in pages]

    # Counted once per page, so a line appearing twice on one page does not
    # look like evidence of recurrence across the document.
    counts: dict[str, int] = {}
    for lines in lines_per_page:
        for key in {_mask_digits(line) for line in lines if line.strip()}:
            counts[key] = counts.get(key, 0) + 1

    threshold = max(2, round(settings.HEADER_FOOTER_MIN_FRACTION * len(pages)))
    furniture = {key for key, count in counts.items() if count >= threshold}
    if not furniture:
        return pages

    cleaned: list[Page] = []
    removed = 0
    for page, lines in zip(pages, lines_per_page):
        drop = _furniture_runs(lines, furniture)
        removed += len(drop)
        kept = [line for i, line in enumerate(lines) if i not in drop]
        cleaned.append(Page(page_number=page.page_number, text="\n".join(kept)))

    before = sum(len(page.text) for page in pages)
    after = sum(len(page.text) for page in cleaned)
    share = (before - after) / before if before else 0.0
    if share > settings.HEADER_FOOTER_MAX_DOCUMENT_FRACTION:
        logger.warning(
            "Header/footer removal would drop %.0f%% of the document; leaving "
            "it untouched, as that much recurring text is more likely to be "
            "real content than page furniture",
            100 * share,
        )
        return pages

    logger.info(
        "Removed %d header/footer line(s) matching %d recurring pattern(s), "
        "%.1f%% of the text",
        removed,
        len(furniture),
        100 * share,
    )
    return cleaned


def _furniture_runs(lines: list[str], furniture: set[str]) -> set[int]:
    """Indices of the furniture runs at the top and bottom of one page.

    Each run stops at the first line that is not furniture, which is what
    protects body text. Blank lines are skipped rather than treated as a stop,
    so a blank line between a running head and the text does not end the run
    early. A page whose every line recurs is emptied, and the loader then drops
    it — that page was boilerplate.
    """
    filled = [i for i, line in enumerate(lines) if line.strip()]

    drop: set[int] = set()
    for order in (filled, list(reversed(filled))):
        for i in order:
            if i in drop or _mask_digits(lines[i]) not in furniture:
                break
            drop.add(i)
    return drop


def _mask_digits(line: str) -> str:
    """Normalize a line so that page-number variants collapse to one pattern.

    Internal whitespace is collapsed as well as masked digits, because a
    fixed-width document pads its furniture to keep the page number
    right-aligned: RFC 2616's footer reads "... [Page 9]" on one page and
    "... [Page 10]" on the next, with one space fewer before the bracket. Those
    are the same running footer, and without collapsing they hash to different
    patterns, each too rare to reach the recurrence threshold — which is why
    that footer survived into the text and was being retrieved as content.
    """
    line = _HORIZONTAL_RUNS.sub(" ", line.strip().lower())
    # A line that is nothing but a roman numeral is a page number. Matching the
    # whole line only, so that ordinary words are never mistaken for numerals.
    if _ROMAN_NUMERAL.fullmatch(line):
        return "#"
    return _DIGITS.sub("#", line)
