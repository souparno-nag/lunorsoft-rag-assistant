"""Ingestion facade over both indexes (see specs/design.md §3, §4).

From Phase 4 there are two indexes over the same chunks — Chroma for dense
retrieval, BM25 for sparse — and they are only useful if they hold the same
thing. Every write has to reach both, every deletion has to reach both, and a
reset has to clear both. Left to each caller, that invariant survives exactly
as long as nobody forgets a line.

This is the one place that knows a document becomes two index entries, so
`app.py` asks it to index a document and never handles the halves itself.
"""

import logging

from src.index.keyword_index import KeywordIndex
from src.index.vector_store import VectorStore
from src.ingest.chunker import chunk_document, chunk_documents
from src.models.schemas import Chunk, RawDocument

logger = logging.getLogger(__name__)


class Indexer:
    """Keeps the vector store and the keyword index in step."""

    def __init__(self, provider: str | None = None) -> None:
        self.vectors = VectorStore(provider=provider)
        self.keywords = KeywordIndex()
        if self.needs_rebuild:
            logger.warning(
                "The two indexes disagree: %d vector(s) against %d keyword "
                "chunk(s). Re-index the documents to bring them back in step.",
                self.vectors.count(),
                self.keywords.count(),
            )

    # --- writing ----------------------------------------------------------

    def add_document(self, document: RawDocument) -> int:
        """Chunk one document and write it to both indexes."""
        return self.add_chunks(chunk_document(document))

    def add_documents(self, documents: list[RawDocument]) -> int:
        """Chunk several documents and write them to both indexes."""
        return self.add_chunks(chunk_documents(documents))

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Write chunks to both indexes, returning how many were indexed.

        The vector store goes first because it is the side that can fail —
        embedding calls a model, and on the Gemini path a network. If it
        raises, the keyword index is left untouched rather than holding chunks
        whose vectors were never stored, and the caller can retry cleanly.
        """
        if not chunks:
            return 0
        self.vectors.add_chunks(chunks)
        self.keywords.add_chunks(chunks)
        return len(chunks)

    # --- reading ----------------------------------------------------------

    def count(self) -> int:
        """How many chunks are indexed, as the vector store sees it."""
        return self.vectors.count()

    def indexed_documents(self) -> dict[str, str]:
        """Map of `doc_id` to source filename for everything indexed."""
        return self.vectors.indexed_documents()

    def chunk_counts(self) -> dict[str, int]:
        """How many chunks each document contributed, keyed by `doc_id`.

        Read from the keyword side, which holds every chunk in memory, rather
        than from Chroma, which would have to fetch them all back to count. The
        two agree whenever `needs_rebuild` is False.
        """
        return self.keywords.chunk_counts()

    @property
    def needs_rebuild(self) -> bool:
        """Whether the two indexes hold different numbers of chunks.

        True for an index built before the keyword half existed, which has
        vectors and no `chunks.jsonl`: dense retrieval still works, sparse
        silently returns nothing, and hybrid search quietly degrades to the
        Phase 2 behaviour. Worth telling the user rather than letting them
        wonder why keyword queries stopped landing.
        """
        return self.vectors.count() != self.keywords.count()

    # --- lifecycle --------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """Remove one document from both indexes."""
        self.vectors.delete_document(doc_id)
        self.keywords.delete_document(doc_id)

    def reset(self) -> None:
        """Clear both indexes — the 'rebuild index' action."""
        self.vectors.reset()
        self.keywords.reset()
        logger.info("Both indexes reset")
