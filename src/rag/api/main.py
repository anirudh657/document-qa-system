"""FastAPI application entry point.

    uvicorn rag.api.main:app --reload

Interactive docs are served at /docs. Until the Phase 5 demo UI exists, that page is
the most convenient way to exercise the API by hand.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag import __version__
from rag.api.deps import build_components, set_components
from rag.api.routes import router
from rag.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build shared components once, before the first request.

    Startup deliberately does not fail on an unreachable vector store or a missing API
    key. The service comes up degraded and reports exactly what is wrong through
    /v1/health, which is more useful than a crash loop that hides the reason.
    """
    set_components(build_components(get_settings()))
    yield
    set_components(None)


app = FastAPI(
    title="Document QA System",
    description="Production-grade RAG over a private document corpus",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "document-qa-system", "version": __version__, "docs": "/docs"}
