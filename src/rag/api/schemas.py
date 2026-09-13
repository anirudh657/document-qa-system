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
    path: str
    recursive: bool = True


class IngestResponse(BaseModel):
    job_id: str
    documents_queued: int


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, str]
    index_fingerprint: str
