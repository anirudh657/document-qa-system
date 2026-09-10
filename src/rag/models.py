"""Core domain models shared across ingestion, retrieval and generation."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    source_path: str
    title: str | None = None
    doc_type: str
    content_hash: str
    ingested_at: datetime
    page_count: int | None = None


class Chunk(BaseModel):
    """A retrievable unit of text.

    `contextual_header` is prepended before embedding but excluded from the text
    shown to the user. `parent_id` supports small-to-big retrieval: embed this
    chunk for precision, send the parent to the model for context.
    """

    id: str
    document_id: str
    text: str
    contextual_header: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    parent_id: str | None = None
    token_count: int = 0

    def embedding_text(self) -> str:
        return f"{self.contextual_header}\n\n{self.text}".strip()


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        for score in (self.rerank_score, self.fusion_score, self.dense_score, self.sparse_score):
            if score is not None:
                return score
        return 0.0


class Citation(BaseModel):
    chunk_id: str
    source_path: str
    section: str | None = None
    page_number: int | None = None
    quote: str


class Answer(BaseModel):
    query_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    answered: bool = True
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
