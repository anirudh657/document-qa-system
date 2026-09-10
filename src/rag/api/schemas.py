"""API request and response schemas."""

from pydantic import BaseModel, Field

from rag.models import Citation


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict | None = None
    stream: bool = False


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    answered: bool
    latency_ms: int
    cache_hit: bool


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
