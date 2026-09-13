"""Shared, process-lifetime components for the API.

The embedding model is hundreds of megabytes and takes seconds to load. Building it
per request would make every query pay that cost, so components are constructed once
at startup and reused. `lifespan` in main.py fills this in before the first request
and clears it on shutdown.

Kept in its own module so routes can depend on it without importing the app, which
would be a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from rag.config import Settings
from rag.generation.generator import Generator
from rag.generation.llm import LLMError, build_llm
from rag.ingestion.embedding import Embedder, build_embedder
from rag.retrieval.store import VectorStore


@dataclass
class Components:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    generator: Generator | None
    llm_error: str = ""


_components: Components | None = None


def build_components(settings: Settings) -> Components:
    """Construct everything the API needs.

    A missing or invalid API key is recorded rather than raised. The service should
    still start and still serve /v1/health and retrieval — reporting the problem is
    more useful than refusing to boot, and it is how an operator finds out what is
    wrong in the first place.
    """
    generator: Generator | None = None
    llm_error = ""
    try:
        generator = Generator(
            build_llm(settings.llm_provider, settings.llm_model, settings.llm_key())
        )
    except LLMError as error:
        llm_error = str(error)

    return Components(
        settings=settings,
        embedder=build_embedder(
            settings.embedding_model, settings.embedding_dim, settings.embedding_batch_size
        ),
        store=VectorStore.connect(
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.embedding_dim,
            settings.index_fingerprint(),
        ),
        generator=generator,
        llm_error=llm_error,
    )


def set_components(components: Components | None) -> None:
    global _components
    _components = components


def get_components() -> Components:
    if _components is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return _components
