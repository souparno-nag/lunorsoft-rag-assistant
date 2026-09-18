"""Document loading (see specs/design.md §4.1).

Turns a file — on disk or uploaded as bytes — into a `RawDocument` whose pages
are already normalized, so every downstream stage is format-agnostic. Multiple
documents are supported from the start: each one carries a `doc_id` and
`source_file` that follow its chunks all the way into citations.
"""

import hashlib
import re
import tempfile
from pathlib import Path

from src.ingest.extract import extract_pages
from src.models.schemas import Page, RawDocument

PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES

_SLUG_TRIM = re.compile(r"[^a-z0-9]+")


class UnsupportedDocumentError(ValueError):
    """Raised for a file whose extension we have no extractor for."""


class EmptyDocumentError(ValueError):
    """Raised when a document yields no extractable text at all.

    Usually a scanned PDF with no text layer, which would need OCR.
    """


def make_doc_id(filename: str, content: bytes) -> str:
    """Build a stable, readable `doc_id` such as `attention-is-all-you-need-3f2a9c11`.

    The hash is over the file's bytes rather than its name, so re-ingesting the
    same file produces the same id (and therefore the same chunk ids, keeping
    re-indexing idempotent), while two different files that happen to share a
    name stay distinct.
    """
    slug = _SLUG_TRIM.sub("-", Path(filename).stem.lower()).strip("-")[:40]
    digest = hashlib.sha256(content).hexdigest()[:8]
    return f"{slug}-{digest}" if slug else f"doc-{digest}"


def load_document(path: str | Path) -> RawDocument:
    """Load one document from disk into a `RawDocument`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such document: {path}")
    return _build(path, path.name, path.read_bytes())


def load_bytes(content: bytes, filename: str) -> RawDocument:
    """Load a document held in memory, as Streamlit's uploader hands it over.

    pdfplumber and pypdf both want a real path, so the bytes are staged in a
    temporary file that is removed once the pages have been extracted.
    """
    suffix = Path(filename).suffix.lower()
    _check_supported(suffix, filename)
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        return _build(Path(tmp.name), filename, content)


def load_documents(
    paths: list[str | Path],
) -> tuple[list[RawDocument], list[tuple[str, str]]]:
    """Load several documents, skipping any that cannot be read.

    Per specs/design.md §14 a corrupt or unreadable file must not abort the
    whole ingestion run, so failures come back as `(filename, message)` pairs
    alongside the documents that did load, for the UI to surface as warnings.
    """
    documents: list[RawDocument] = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            documents.append(load_document(path))
        except (OSError, ValueError) as exc:
            failures.append((Path(path).name, str(exc)))
    return documents, failures


def _build(path: Path, display_name: str, content: bytes) -> RawDocument:
    suffix = path.suffix.lower()
    _check_supported(suffix, display_name)

    if suffix in PDF_SUFFIXES:
        pages = extract_pages(path)
    else:
        # Plain text has no pagination; it becomes a single page so that
        # citations still have a page slot to fill.
        pages = [Page(page_number=1, text=content.decode("utf-8", errors="replace"))]

    pages = [page for page in pages if page.text.strip()]
    if not pages:
        raise EmptyDocumentError(
            f"{display_name} has no extractable text — if it is a scanned PDF "
            "it would need OCR, which is out of scope."
        )

    return RawDocument(
        doc_id=make_doc_id(display_name, content),
        source_file=display_name,
        pages=pages,
    )


def _check_supported(suffix: str, filename: str) -> None:
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise UnsupportedDocumentError(
            f"{filename}: {suffix or 'no extension'} is not supported "
            f"(supported: {supported})"
        )
