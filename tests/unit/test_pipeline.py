"""Ingestion pipeline tests.

Run against HashEmbedder and in-memory Qdrant, so no model download and no Docker.
"""

from pathlib import Path

import pytest

from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.embedding import HashEmbedder
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.store import VectorStore

DIM = 32
BODY = (
    "Set theory covers union intersection and difference operations.\n\n"
    "De Morgan laws relate complements of unions to intersections.\n\n"
    "Relations may be reflexive symmetric or transitive in nature."
)


@pytest.fixture
def pipeline():
    store = VectorStore.connect(":memory:", "test_ingest", DIM, fingerprint="test:512:64")
    store.ensure_collection()
    return IngestionPipeline(
        FixedSizeChunker(chunk_size=40, overlap=8),
        HashEmbedder(dimension=DIM),
        store,
    )


@pytest.fixture
def corpus(tmp_path: Path):
    (tmp_path / "notes.txt").write_text(BODY, encoding="utf-8")
    (tmp_path / "guide.md").write_text(f"# Guide\n\n{BODY}", encoding="utf-8")
    return tmp_path


def test_discover_finds_supported_files(pipeline, corpus):
    assert len(pipeline.discover(corpus)) == 2


def test_discover_ignores_unsupported_extensions(pipeline, corpus):
    (corpus / "photo.png").write_bytes(b"\x89PNG")
    (corpus / "archive.zip").write_bytes(b"PK")
    assert len(pipeline.discover(corpus)) == 2


def test_discover_recurses_by_default(pipeline, corpus):
    nested = corpus / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text(BODY, encoding="utf-8")
    assert len(pipeline.discover(corpus)) == 3


def test_discover_can_stay_shallow(pipeline, corpus):
    nested = corpus / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text(BODY, encoding="utf-8")
    assert len(pipeline.discover(corpus, recursive=False)) == 2


def test_discover_accepts_a_single_file(pipeline, corpus):
    assert pipeline.discover(corpus / "notes.txt") == [corpus / "notes.txt"]


def test_run_indexes_every_document(pipeline, corpus):
    report = pipeline.run(corpus)
    assert len(report.indexed) == 2
    assert report.total_chunks > 0
    assert pipeline.store.count() == report.total_chunks


def test_run_on_a_missing_path_raises(pipeline, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "nope")


def test_empty_file_is_skipped_not_fatal(pipeline, corpus):
    """A scanned PDF or blank file must cost one document, not the whole run."""
    (corpus / "blank.txt").write_text("  ", encoding="utf-8")
    report = pipeline.run(corpus)
    assert len(report.indexed) == 2
    assert len(report.skipped) == 1
    assert "no text layer" in report.skipped[0].skipped


def test_a_corrupt_file_does_not_stop_the_batch(pipeline, corpus):
    (corpus / "broken.md").write_bytes(b"\xff\xfe\x00invalid")
    report = pipeline.run(corpus)
    assert len(report.indexed) >= 2


def test_reingesting_does_not_duplicate(pipeline, corpus):
    """FR-6: ingestion is idempotent."""
    first = pipeline.run(corpus)
    pipeline.run(corpus)
    assert pipeline.store.count() == first.total_chunks


def test_indexed_chunks_are_searchable(pipeline, corpus):
    pipeline.run(corpus)
    hits = pipeline.store.search(pipeline.embedder.embed_query("de morgan"), top_k=3)
    assert hits
    assert hits[0].chunk.text
