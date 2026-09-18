"""Chroma vector store wrapper (see specs/design.md §4.5).

Persists to `storage/` so an index survives a restart and the app does not
re-embed the same documents on every launch. Chroma was chosen for exactly
that: local, on disk, no server to run.

The wrapper's job is to keep the rest of the pipeline working in `Chunk` and
`RetrievalResult` objects rather than in Chroma's dictionaries, and to hold the
three pieces of care that a bare client would leave to each caller: metadata
that survives a round trip, similarity scores rather than raw distances, and a
guard against querying an index built by a different embedding provider.
"""

import json
import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma

from config import settings
from src.index.embeddings import embedding_dimension, get_embeddings
from src.models.schemas import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

# Chroma stores only str, int, float and bool, so optional fields are dropped
# when absent and restored as None on the way back out.
_OPTIONAL_FIELDS = ("page_number", "section_header")


class IndexProviderMismatch(RuntimeError):
    """Raised when the index on disk was built by a different embedding provider.

    Not recoverable by retrying: the stored vectors have a different width and
    a different meaning, so the index has to be rebuilt.
    """


class VectorStore:
    """A persisted Chroma collection of embedded chunks."""

    def __init__(self, provider: str | None = None, persist_dir: Path | None = None) -> None:
        self._provider = provider or settings.EMBEDDING_PROVIDER
        self._persist_dir = persist_dir or settings.CHROMA_DIR
        self._meta_path = (
            settings.INDEX_META_PATH
            if persist_dir is None
            else Path(persist_dir).parent / settings.INDEX_META_PATH.name
        )

        self._check_provider_matches_index()
        self._store = self._open()

    def _open(self) -> Chroma:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=get_embeddings(self._provider),
            persist_directory=str(self._persist_dir),
            # Vectors are unit length by the time they arrive (see
            # embeddings._normalize), so cosine is the natural metric and its
            # distance converts to a similarity by simple subtraction.
            collection_metadata={"hnsw:space": "cosine"},
        )

    # --- writing ----------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store chunks, returning how many were written.

        Ids are the chunk ids, and Chroma upserts by id, so re-ingesting an
        unchanged file replaces its chunks instead of duplicating them. That
        works because a `doc_id` is derived from the file's content, so the
        same file always produces the same ids.
        """
        if not chunks:
            return 0

        self._store.add_texts(
            texts=[chunk.chunk_text for chunk in chunks],
            metadatas=[_to_metadata(chunk) for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
        )
        self._write_index_meta()
        logger.info(
            "Indexed %d chunks from %d document(s); collection now holds %d",
            len(chunks),
            len({chunk.doc_id for chunk in chunks}),
            self.count(),
        )
        return len(chunks)

    # --- reading ----------------------------------------------------------

    def similarity_search(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        """Return the `k` chunks closest to `query`, most similar first."""
        k = k or settings.K_RETRIEVE
        if self.count() == 0:
            return []

        hits = self._store.similarity_search_with_score(query, k=k)
        return [
            RetrievalResult(chunk=_to_chunk(document), dense_score=_to_similarity(distance))
            for document, distance in hits
        ]

    def count(self) -> int:
        """How many chunks are currently indexed."""
        return self._store._collection.count()

    def indexed_documents(self) -> dict[str, str]:
        """Map of `doc_id` to source filename for everything in the index.

        Lets the UI list what has been ingested without re-reading the files.
        """
        records = self._store.get(include=["metadatas"])
        return {
            metadata["doc_id"]: metadata["source_file"]
            for metadata in records.get("metadatas") or []
        }

    # --- lifecycle --------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk belonging to one document."""
        self._store.delete(where={"doc_id": doc_id})
        logger.info("Removed document %s from the index", doc_id)

    def reset(self) -> None:
        """Delete the whole index, including the files on disk.

        The 'rebuild index' action, and the way out of a provider mismatch.
        The store stays usable afterwards, holding a fresh empty collection.
        """
        self._store.delete_collection()
        _release_client_cache()
        shutil.rmtree(self._persist_dir, ignore_errors=True)
        self._meta_path.unlink(missing_ok=True)
        self._store = self._open()
        logger.info("Index reset; %s rebuilt empty", self._persist_dir)

    # --- provider bookkeeping --------------------------------------------

    def _check_provider_matches_index(self) -> None:
        """Refuse to open an index whose vectors came from another provider.

        Without this the failure surfaces much later, as a dimension mismatch
        from deep inside Chroma during a query, or — worse, when widths happen
        to agree — as silently meaningless similarity scores.
        """
        if not self._meta_path.exists():
            return

        try:
            meta = json.loads(self._meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; assuming the index matches", self._meta_path)
            return

        stored = meta.get("provider")
        if stored and stored != self._provider:
            raise IndexProviderMismatch(
                f"The index in {self._persist_dir} was built with the "
                f"{stored!r} embedding provider ({meta.get('dimension')} "
                f"dimensions) but the current provider is {self._provider!r} "
                f"({embedding_dimension(self._provider)} dimensions). Chroma "
                "cannot mix vector widths, so rebuild the index or switch "
                f"EMBEDDING_PROVIDER back to {stored!r}."
            )

    def _write_index_meta(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(
                {
                    "provider": self._provider,
                    "dimension": embedding_dimension(self._provider),
                    "model": (
                        settings.GEMINI_EMBED_MODEL
                        if self._provider == "gemini"
                        else settings.LOCAL_EMBED_MODEL
                    ),
                    "collection": settings.CHROMA_COLLECTION,
                },
                indent=2,
            )
        )


def _release_client_cache() -> None:
    """Drop chromadb's process-wide client cache for the persist directory.

    chromadb keeps one client per directory per process. Deleting the files
    underneath a live client leaves that cached handle pointing at a database
    that no longer exists, and the next write fails with "attempt to write a
    readonly database" — which is what a 'rebuild index' button would hit.
    Clearing the cache forces the next open to build a fresh client.
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        # A private API, so tolerate it moving: the worst case is that a reset
        # within one process leaves a stale handle, which a restart clears.
        logger.debug("Could not clear the chromadb client cache", exc_info=True)


def _to_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Flatten a chunk into Chroma's metadata, dropping absent optional fields."""
    metadata: dict[str, str | int] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_file": chunk.source_file,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "token_count": chunk.token_count,
    }
    for field in _OPTIONAL_FIELDS:
        value = getattr(chunk, field)
        if value is not None:
            metadata[field] = value
    return metadata


def _to_chunk(document) -> Chunk:
    """Rebuild a `Chunk` from a stored document, inverting `_to_metadata`."""
    metadata = document.metadata
    return Chunk(
        chunk_id=metadata["chunk_id"],
        doc_id=metadata["doc_id"],
        source_file=metadata["source_file"],
        chunk_text=document.page_content,
        char_start=metadata["char_start"],
        char_end=metadata["char_end"],
        token_count=metadata["token_count"],
        page_number=metadata.get("page_number"),
        section_header=metadata.get("section_header"),
    )


def _to_similarity(distance: float) -> float:
    """Convert Chroma's cosine distance into a similarity in [0, 1].

    Chroma reports distance, where smaller is better, while every other stage
    of the pipeline ranks by score, where larger is better. Converting here
    keeps that inconsistency from leaking into the fusion and reranking code
    that Phases 4 and 5 add.
    """
    return max(0.0, min(1.0, 1.0 - distance))
