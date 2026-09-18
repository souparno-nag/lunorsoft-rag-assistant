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
# A hyphen at end of line followed by a lowercase letter is line-breaking
# hyphenation ("repre-\nsentation"), not a real compound. Requiring lowercase
# leaves genuine hyphenates such as "Dot-\nProduct" intact.
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n([a-z])")
_DIGITS = re.compile(r"\d+")


def preprocess_pages(pages: list[Page]) -> list[Page]:
    """Normalize whitespace, rejoin hyphenated words and drop page furniture."""
    pages = [
        Page(page_number=page.page_number, text=normalize_whitespace(page.text))
        for page in pages
    ]
    pages = strip_repeated_headers_footers(pages)
    return [
        Page(page_number=page.page_number, text=normalize_whitespace(dehyphenate(page.text)))
        for page in pages
    ]


def normalize_whitespace(text: str) -> str:
    """Collapse the whitespace noise that PDF extraction leaves behind.

    Idempotent, so it is safe to run again after other passes have edited the
    text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _UNICODE_SPACES.sub(" ", text)
    text = _HORIZONTAL_RUNS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by hyphenation.

    Without this, "repre-\\nsentation" is embedded as two fragments and never
    matches a search for "representation".
    """
    return _LINE_BREAK_HYPHEN.sub(r"\1\2", text)


def strip_repeated_headers_footers(pages: list[Page]) -> list[Page]:
    """Drop running heads and footers — the lines that recur on most pages.

    Candidates are taken only from the top and bottom few lines of each page,
    and are compared with digit runs masked, so that a footer which is just the
    page number ("2", "3", "4", ...) is recognized as one recurring line rather
    than fifteen unique ones.
    """
    if len(pages) < settings.HEADER_FOOTER_MIN_PAGES:
        return pages

    lines_per_page = [page.text.split("\n") for page in pages]
    edges_per_page = [_edge_indices(lines) for lines in lines_per_page]

    # Counted once per page, so a document whose header and footer happen to be
    # identical does not count as two sightings on a single page.
    counts: dict[str, int] = {}
    for lines, edges in zip(lines_per_page, edges_per_page):
        for key in {_mask_digits(lines[i]) for i in edges}:
            counts[key] = counts.get(key, 0) + 1

    threshold = max(2, round(settings.HEADER_FOOTER_MIN_FRACTION * len(pages)))
    furniture = {key for key, count in counts.items() if count >= threshold}
    if not furniture:
        return pages

    cleaned: list[Page] = []
    removed = 0
    for page, lines, edges in zip(pages, lines_per_page, edges_per_page):
        # Only the edge positions are dropped — body text that happens to match
        # a running head stays where it is.
        drop = {i for i in edges if _mask_digits(lines[i]) in furniture}
        removed += len(drop)
        kept = [line for i, line in enumerate(lines) if i not in drop]
        cleaned.append(Page(page_number=page.page_number, text="\n".join(kept)))

    logger.info(
        "Removed %d header/footer line(s) matching %d recurring pattern(s)",
        removed,
        len(furniture),
    )
    return cleaned


def _edge_indices(lines: list[str]) -> list[int]:
    """Indices of the first and last few non-empty lines — the only candidates."""
    filled = [i for i, line in enumerate(lines) if line.strip()]
    scan = settings.HEADER_FOOTER_SCAN_LINES
    if len(filled) <= 2 * scan:
        return filled
    return filled[:scan] + filled[-scan:]


def _mask_digits(line: str) -> str:
    return _DIGITS.sub("#", line.strip().lower())
