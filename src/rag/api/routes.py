"""API routes.

TODO(phase-1): /v1/query, /v1/health
TODO(phase-4): /v1/ingest, /v1/query/{id}/trace, auth, rate limiting
"""

from fastapi import APIRouter

from rag.api.schemas import HealthResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/v1")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    raise NotImplementedError("TODO(phase-1)")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    raise NotImplementedError("TODO(phase-1)")
