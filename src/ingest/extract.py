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
