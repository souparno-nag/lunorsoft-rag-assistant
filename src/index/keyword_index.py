"""BM25 keyword index (see specs/design.md §4.6).

The sparse half of hybrid retrieval. Dense vectors are good at paraphrase and
bad at literal tokens: a query for `d_k`, `BLEU`, `ms-marco-MiniLM` or an error
code is asking for a string that appears in the text, and an embedding model
will happily return something about the same *topic* instead. BM25 does the
opposite, which is exactly why the two are fused rather than chosen between.

This index also serves as the **chunk metadata store** of specs/design.md §3:
`storage/chunks.jsonl` holds every indexed chunk in full, which is what makes
the BM25 index rebuildable and gives the rest of the system somewhere to read
chunks back from without going through Chroma.
"""

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import settings
from src.models.schemas import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

# Words for BM25: runs of letters, digits and underscores, lowercased.
#
# Deliberately no stemming and no stopword list. Stemming would defeat the
# point of having a sparse retriever at all — it is here to match literal
# tokens, and conflating "encoding" with "encode" is the dense retriever's job.
# Stopwords need no list either, because BM25's IDF term already drives the
# weight of a word appearing in every chunk towards zero.
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Split text into the terms BM25 matches on.

    Known bound, from the sample corpus: PDF extraction flattens subscripts, so
    "Attention Is All You Need" reads `dk` and `dmodel` where the typeset paper
    shows d_k and d_model. A reader who types `d_k` matches nothing here,
    because the string genuinely is not in the extracted text. That is an
    extraction artefact rather than an indexing one, and it is precisely the
    case the dense retriever covers, which is the argument for fusing the two
    rather than picking one.
    """
    return _TOKEN.findall(text.lower())


class KeywordIndex:
    """A persisted BM25 index over the same chunks as the vector store.

    The BM25 structure itself is rebuilt in memory when the index is opened
    rather than pickled to disk. Pickling a third-party object couples the
    files on disk to the installed version of `rank_bm25` — a dependency bump
    would silently invalidate a user's index — while rebuilding costs a regex
    pass over the corpus, which is milliseconds at any size this project will
    see. `storage/chunks.jsonl` is therefore the real persisted artifact.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or settings.CHUNKS_PATH
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    # --- writing ----------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add or replace chunks, returning how many were written.

        Keyed by `chunk_id` exactly as the vector store is, so re-ingesting an
        unchanged file replaces its chunks in both indexes instead of
        duplicating them in either.
        """
        if not chunks:
            return 0

        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._save()
        self._rebuild()
        logger.info(
            "Keyword index: wrote %d chunk(s); now holds %d", len(chunks), self.count()
        )
        return len(chunks)

    # --- reading ----------------------------------------------------------

    def search(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        """Return the `k` best-matching chunks, strongest first.

        Chunks scoring zero are dropped rather than returned to fill the quota.
        A zero means the query and the chunk share no term at all, so the chunk
        carries no keyword evidence — and passing it on would hand it a rank in
        the fusion below, where rank is the whole currency.
        """
        k = k or settings.K_RETRIEVE
        if not self._bm25 or not self._ids:
            return []

        terms = tokenize(query)
        if not terms:
            return []

        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            ((float(score), chunk_id) for score, chunk_id in zip(scores, self._ids) if score > 0),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            RetrievalResult(chunk=self._chunks[chunk_id], bm25_score=score)
            for score, chunk_id in ranked[:k]
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        """Look one chunk up by id."""
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        """How many chunks are currently indexed."""
        return len(self._chunks)

    def indexed_documents(self) -> dict[str, str]:
        """Map of `doc_id` to source filename for everything in the index."""
        return {chunk.doc_id: chunk.source_file for chunk in self._chunks.values()}

    def chunk_counts(self) -> dict[str, int]:
        """How many chunks each document contributed, keyed by `doc_id`."""
        counts: dict[str, int] = {}
        for chunk in self._chunks.values():
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return counts

    # --- lifecycle --------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk belonging to one document."""
        removed = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for chunk_id in removed:
            del self._chunks[chunk_id]
        if removed:
            self._save()
            self._rebuild()
        logger.info("Keyword index: removed %d chunk(s) for %s", len(removed), doc_id)

    def reset(self) -> None:
        """Drop the whole index, on disk and in memory."""
        self._chunks.clear()
        self._ids.clear()
        self._bm25 = None
        self._path.unlink(missing_ok=True)
        logger.info("Keyword index reset; %s removed", self._path)

    # --- persistence ------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        chunk = Chunk(**json.loads(line))
                        self._chunks[chunk.chunk_id] = chunk
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            # A truncated or stale-schema file must not make the app
            # unopenable: the vector store still holds the same chunks, and
            # re-indexing rewrites this from scratch.
            logger.warning(
                "Could not read %s (%s); starting with an empty keyword index. "
                "Re-index the documents to rebuild it.",
                self._path,
                exc,
            )
            self._chunks.clear()
            return
        self._rebuild()
        logger.info("Keyword index: loaded %d chunk(s) from %s", self.count(), self._path)

    def _save(self) -> None:
        """Write the corpus out, replacing whatever was there.

        Rewritten whole rather than appended to, because `add_chunks` replaces
        chunks by id and an append-only file would keep the superseded copies.
        Written to a temporary file and moved into place so an interrupted
        write cannot leave a half-written index behind.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks.values():
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        temporary.replace(self._path)

    def _rebuild(self) -> None:
        """Rebuild the BM25 structure from the chunks held in memory."""
        self._ids = list(self._chunks)
        corpus = [tokenize(self._chunks[chunk_id].chunk_text) for chunk_id in self._ids]
        # BM25Okapi divides by the corpus average document length, so an empty
        # corpus is not merely useless, it raises.
        self._bm25 = BM25Okapi(corpus) if corpus else None
