"""API routes.

Errors are mapped to status codes that say what the caller should do: 503 when a
dependency is down and retrying later may work, 502 when the upstream model failed,
409 when the index is stale and needs rebuilding. A blanket 500 would tell the caller
nothing and hide operational problems behind a generic failure.

TODO(phase-4): /v1/ingest, /v1/query/{id}/trace, auth, rate limiting, streaming
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from rag.api.deps import Components, get_components
from rag.api.schemas import HealthResponse, QueryResponse, SearchHit, SearchResponse
from rag.api.schemas import QueryRequest as QueryRequestSchema
from rag.generation.llm import LLMError
from rag.retrieval.store import IndexFingerprintMismatchError

router = APIRouter(prefix="/v1")

Deps = Annotated[Components, Depends(get_components)]


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequestSchema,
    components: Deps,
) -> QueryResponse:
    """Answer a question from the indexed corpus, with validated citations."""
    if components.generator is None:
        raise HTTPException(status_code=503, detail=components.llm_error)

    try:
        passages = components.store.search(
            components.embedder.embed_query(request.question),
            top_k=request.top_k,
            filters=request.filters,
        )
    except IndexFingerprintMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Vector store unavailable: {type(error).__name__}"
        ) from error

    try:
        answer = await components.generator.generate(request.question, passages)
    except LLMError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return QueryResponse(
        query_id=answer.query_id,
        answer=answer.text,
        citations=answer.citations,
        answered=answer.answered,
        latency_ms=answer.latency_ms,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cache_hit=answer.cache_hit,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: QueryRequestSchema,
    components: Deps,
) -> SearchResponse:
    """Retrieve passages without generating an answer.

    Useful on its own: when an answer is wrong the first question is always whether
    retrieval found the right passages, and this answers that without an LLM call.
    """
    try:
        hits = components.store.search(
            components.embedder.embed_query(request.question),
            top_k=request.top_k,
            filters=request.filters,
        )
    except IndexFingerprintMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Vector store unavailable: {type(error).__name__}"
        ) from error

    return SearchResponse(
        query=request.question,
        hits=[
            SearchHit(
                chunk_id=hit.chunk.id,
                text=hit.chunk.text,
                score=hit.final_score,
                page_number=hit.chunk.page_number,
                source=hit.chunk.contextual_header or hit.chunk.document_id,
            )
            for hit in hits
        ],
    )


@router.get("/health", response_model=HealthResponse)
async def health(components: Deps) -> HealthResponse:
    """Liveness plus per-dependency status.

    Always returns 200 so a monitor can read the body and see which dependency is
    degraded — a non-200 would collapse that detail into "down".
    """
    dependencies: dict[str, str] = {}

    try:
        dependencies["qdrant"] = (
            f"ok ({components.store.count()} chunks)"
            if components.store.exists()
            else "ok (no collection)"
        )
    except Exception as error:
        dependencies["qdrant"] = f"unreachable: {type(error).__name__}"

    dependencies["llm"] = (
        f"ok ({components.settings.llm_model})"
        if components.generator is not None
        else f"unconfigured: {components.llm_error}"
    )

    degraded = any(not value.startswith("ok") for value in dependencies.values())
    return HealthResponse(
        status="degraded" if degraded else "ok",
        dependencies=dependencies,
        index_fingerprint=components.settings.index_fingerprint(),
    )
