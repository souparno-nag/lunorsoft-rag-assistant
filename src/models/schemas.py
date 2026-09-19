"""Data models passed between pipeline stages (see specs/design.md §7)."""

from dataclasses import dataclass, field
from typing import Literal

Confidence = Literal["high", "medium", "low"]


@dataclass
class Page:
    """One extracted page of a source document. Page numbers are 1-based so
    they match what a reader sees in a PDF viewer, since they end up in
    citations."""

    page_number: int
    text: str


@dataclass
class RawDocument:
    """A loaded document, normalized so downstream stages never need to know
    which file format it came from."""

    doc_id: str
    source_file: str
    pages: list[Page] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_text: str
    char_start: int
    char_end: int
    token_count: int
    page_number: int | None = None
    section_header: str | None = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class Citation:
    """One source excerpt that was placed in the answer's context.

    `marker` is the bracket number the excerpt was presented to the model
    under, so a `[2]` in the answer text and the citation numbered 2 refer to
    the same passage. `cited` records whether the answer actually referred to
    it: every excerpt the model was shown is listed, because what retrieval
    put in front of it is worth seeing, but the ones it drew on are the ones
    that support the claims.
    """

    source_file: str
    snippet: str
    marker: int
    # Two uploaded files can carry the same name and different contents, and
    # `doc_id` is what tells them apart — it is derived from the bytes, not the
    # filename. Carried here so a citation can be disambiguated when a corpus
    # holds two documents called "report.pdf".
    doc_id: str = ""
    page: int | None = None
    section: str | None = None
    cited: bool = False


@dataclass
class Turn:
    """One exchange in a conversation.

    The whole envelope is kept, not just the answer text, so a past turn can
    be redrawn with the citations and confidence badge it was originally
    served with — Streamlit re-runs the script on every interaction, and an
    answer that lost its evidence on the next keystroke would be worse than
    no history at all.
    """

    question: str
    envelope: "AnswerEnvelope"


@dataclass
class AnswerEnvelope:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    faithfulness_score: float | None = None
    confidence: Confidence | None = None
    # How the faithfulness score was arrived at — "judge" or "overlap". The
    # two are not equally trustworthy (see src/generate/grounding.py), so the
    # UI has to be able to say which one it is showing.
    grounding_method: str | None = None
    used_query_transform: str | None = None
    retrieved_k: int = 0
    final_k: int = 0
