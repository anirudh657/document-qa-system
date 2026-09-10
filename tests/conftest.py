"""Shared fixtures."""

import pytest

from rag.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        qdrant_collection="test_documents",
        chunk_size=128,
        chunk_overlap=16,
    )
