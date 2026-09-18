"""Data models passed between pipeline stages (see specs/design.md §7)."""

from dataclasses import dataclass, field
from typing import Literal

Confidence = Literal["high", "medium", "low"]


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
    source_file: str
    snippet: str
    page: int | None = None
    section: str | None = None


@dataclass
class AnswerEnvelope:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    faithfulness_score: float | None = None
    confidence: Confidence | None = None
    used_query_transform: str | None = None
    retrieved_k: int = 0
    final_k: int = 0
