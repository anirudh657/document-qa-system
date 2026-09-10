"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag import __version__
from rag.api.routes import router
from rag.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # TODO(phase-1): connect stores, load embedding and reranker models
    # TODO(phase-1): verify the collection's index fingerprint matches the config —
    #                a mismatch means the vectors are stale and must not be served
    # TODO(phase-4): configure logging and tracing here
    _ = settings
    yield
    # TODO(phase-1): close connections


app = FastAPI(
    title="Document QA System",
    description="Production-grade RAG over a private document corpus",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(router)