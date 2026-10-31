"""API request and response schemas.

These are deliberately separate from the domain models in rag.models. The wire format
is a contract with callers and should change only when the API changes; the internal
models change whenever the pipeline does. Coupling them would make every refactor a
breaking API change.
"""

from typing import Any

from pydantic import BaseModel, Field

from rag.models import Citation


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None
    stream: bool = False


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    answered: bool
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    page_number: int | None = None
    source: str = ""


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class IngestRequest(BaseModel):
    """Ingest a path the SERVER can already see.

    Not what a browser upload needs — the file does not exist server-side yet. Upload
    goes through `POST /v1/documents` as multipart instead, and this stays for the CLI
    and for re-indexing a directory already on disk.
    """

    path: str
    recursive: bool = True


class IngestResponse(BaseModel):
    job_id: str
    documents_queued: int


class DocumentSummary(BaseModel):
    """One indexed document, as the index itself describes it.

    Derived from the chunks rather than from a separate table, so it cannot drift out
    of step with what is actually searchable. If a document is listed here, questions
    about it can be answered.
    """

    source: str
    chunks: int
    pages: int
    """Distinct pages covered, counting the full span of every chunk."""


class DocumentList(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int


class UploadResponse(BaseModel):
    source: str
    chunks: int
    pages: int
    replaced: bool
    """True when a document of this name was already indexed and has been rebuilt."""

    warning: str = ""
    """Set when the upload succeeded but something about it needs saying."""


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, str]
    index_fingerprint: str
