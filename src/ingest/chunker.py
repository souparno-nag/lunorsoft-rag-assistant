"""Chunking (see specs/design.md §4.3).

Basic fixed-size chunking for the Phase 1 walking skeleton. Phase 3 replaces it
with the structure-aware and semantic strategy that is the actual
differentiator; keeping this one first means there is a working index to
compare that against.

Splitting runs over the whole document rather than page by page. A paragraph
that continues across a page break stays in one chunk that way, where per-page
splitting would cut it in half and leave both halves harder to retrieve. The
cost is that a chunk's character offsets no longer imply its page, so pages are
recovered by mapping offsets back through `_PageIndex`.
"""

from bisect import bisect_right
from collections.abc import Iterable
from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.models.schemas import Chunk, Page, RawDocument

# Must match how RawDocument.text joins its pages, or every offset below is
# wrong by the number of preceding pages.
PAGE_SEPARATOR = "\n"


def chunk_documents(documents: Iterable[RawDocument]) -> list[Chunk]:
    """Chunk several documents into one flat list, ready for indexing."""
    return [chunk for document in documents for chunk in chunk_document(document)]


def chunk_document(document: RawDocument) -> list[Chunk]:
    """Split one document into overlapping fixed-size chunks.

    Every chunk carries the metadata a citation needs: which file it came from,
    which page it starts on, and where it sits in the document text.
    """
    text = document.text
    if not text.strip():
        return []

    pages = _PageIndex.build(document.pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks: list[Chunk] = []
    for piece in splitter.create_documents([text]):
        body = piece.page_content
        if not body.strip():
            continue

        start = piece.metadata["start_index"]
        page_number = pages.page_for_span(start, start + len(body))
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_p{page_number}_{len(chunks):04d}",
                doc_id=document.doc_id,
                source_file=document.source_file,
                chunk_text=body,
                char_start=start,
                char_end=start + len(body),
                token_count=count_tokens(body),
                page_number=page_number,
                # Populated by the structure-aware chunker in Phase 3 (T3.1).
                section_header=None,
            )
        )
    return chunks


def count_tokens(text: str) -> int:
    """Count tokens the way the generation model will count them."""
    return len(_encoding().encode(text, allowed_special=set()))


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Load the tokenizer once; building one costs more than using it."""
    return tiktoken.get_encoding(settings.TOKENIZER_ENCODING)


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
