"""PDF text extraction (see specs/design.md §4.2).

Minimal pypdf implementation for now; T1.2 replaces it with pdfplumber as the
primary extractor, keeping pypdf as the fallback for problem files.
"""

from pathlib import Path

from pypdf import PdfReader

from src.models.schemas import Page


def extract_pages(path: str | Path) -> list[Page]:
    """Extract one `Page` per PDF page, numbered from 1 for citations."""
    reader = PdfReader(str(path))
    return [
        Page(page_number=number, text=page.extract_text() or "")
        for number, page in enumerate(reader.pages, start=1)
    ]
