"""API routes.

Errors are mapped to status codes that say what the caller should do: 503 when a
dependency is down and retrying later may work, 502 when the upstream model failed,
409 when the index is stale and needs rebuilding. A blanket 500 would tell the caller
nothing and hide operational problems behind a generic failure.

TODO(phase-4): /v1/query/{id}/trace, auth, rate limiting, streaming
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from rag.api.deps import Components, get_components
from rag.api.schemas import (
    DocumentList,
    DocumentSummary,
    HealthResponse,
    QueryResponse,
    SearchHit,
    SearchResponse,
    UploadResponse,
)
from rag.api.schemas import QueryRequest as QueryRequestSchema
from rag.generation.llm import LLMError
from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.loaders import SUPPORTED_EXTENSIONS, EmptyTextLayerError
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.store import IndexFingerprintMismatchError

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
"""Refused above this. Read fully into memory before anything touches the disk, so an
oversized file costs one rejected request rather than a filled volume."""

DATA_DIR = Path("data")

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


def _summarise(components: Components) -> list[DocumentSummary]:
    """Describe the index by reading it, not by keeping a second list beside it.

    A separate table of "documents we ingested" is one more thing that can disagree
    with the index. Deriving the list from the chunks means it is always true by
    construction: if it is listed, it is searchable.
    """
    chunks_by_source: defaultdict[str, int] = defaultdict(int)
    pages_by_source: defaultdict[str, set[int]] = defaultdict(set)

    for chunk in components.store.iter_chunks():
        chunks_by_source[chunk.source] += 1
        pages_by_source[chunk.source].update(chunk.pages)

    return [
        DocumentSummary(
            source=source,
            chunks=count,
            pages=len(pages_by_source[source]),
        )
        for source, count in sorted(chunks_by_source.items())
    ]


@router.get("/documents", response_model=DocumentList)
async def list_documents(components: Deps) -> DocumentList:
    """Every document currently searchable, derived from the index."""
    try:
        documents = _summarise(components)
    except IndexFingerprintMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Vector store unavailable: {type(error).__name__}"
        ) from error

    return DocumentList(
        documents=documents,
        total_chunks=sum(d.chunks for d in documents),
    )


@router.post("/documents", response_model=UploadResponse, status_code=201)
async def upload_document(
    components: Deps,
    file: Annotated[UploadFile, File()],
) -> UploadResponse:
    """Accept a file, index it, and report what became searchable.

    Synchronous on purpose at this corpus size: embedding one document takes seconds,
    and a caller that gets 201 knows the document is queryable right then. A job queue
    would be the right answer for bulk ingestion and the wrong one for a single upload,
    because it replaces a definite answer with one the caller has to poll for.

    Re-uploading a name that is already indexed rebuilds it. Chunk IDs derive from the
    document's content hash, so re-ingesting identical bytes overwrites the same points
    and changed bytes would otherwise leave the old chunks orphaned in the index.
    """
    # `file.filename` is attacker-controlled. Taking only the final component defeats
    # "../../etc/passwd" and its encodings; the suffix check then refuses anything the
    # loaders cannot read, before a single byte reaches the disk.
    raw_name = Path(file.filename or "").name
    if not raw_name:
        raise HTTPException(status_code=422, detail="The upload has no filename.")

    suffix = Path(raw_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"{suffix or 'That file type'} cannot be indexed. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(payload) // (1024 * 1024)} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    destination = DATA_DIR / raw_name
    replaced = destination.exists()

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    except OSError as error:
        raise HTTPException(status_code=507, detail=f"Could not save the file: {error}") from error

    pipeline = IngestionPipeline(
        FixedSizeChunker(
            components.settings.chunk_size,
            components.settings.chunk_overlap,
        ),
        components.embedder,
        components.store,
    )

    try:
        result = pipeline.ingest_file(destination)
    except IndexFingerprintMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except EmptyTextLayerError as error:
        # The file is readable but unusable — a scan with no text layer, most often.
        # Remove it rather than leave a file in data/ that no re-ingest will ever index.
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503, detail=f"Indexing failed: {type(error).__name__}"
        ) from error

    if not result.ok:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=result.skipped or "Could not index that file.")

    pages = len(
        {
            page
            for chunk in components.store.iter_chunks()
            if chunk.source == raw_name
            for page in chunk.pages
        }
    )

    # Said plainly because it is easy to miss and expensive to discover later: adding a
    # document changes the corpus every recorded evaluation was measured against.
    warning = (
        "The corpus has changed, so existing evaluation runs are no longer a like-for-like "
        "baseline. Re-run `rag eval` before comparing against them."
    )

    return UploadResponse(
        source=raw_name,
        chunks=result.chunks,
        pages=pages,
        replaced=replaced,
        warning=warning,
    )
