"""Vector store tests.

These run against Qdrant's in-memory mode, so the suite needs no Docker and no
network. The API surface is identical to a real server.
"""

from datetime import UTC, datetime

import pytest

from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.embedding import HashEmbedder
from rag.ingestion.loaders import LoadedDocument
from rag.models import DocumentMeta
from rag.retrieval.store import (
    IndexFingerprintMismatchError,
    VectorStore,
    point_id,
)

DIM = 32
TOPICS = [
    "Set theory covers union intersection and difference operations here.",
    "De Morgan laws relate complements of unions to intersections.",
    "Power sets enumerate every possible subset of a given set.",
    "Relations may be reflexive symmetric or transitive in nature.",
    "Composite relations combine two relations into a single mapping.",
    "Graph theory studies vertices edges and their connectivity.",
    "Boolean algebra underpins digital logic circuit design work.",
    "Recurrence relations describe sequences in terms of prior terms.",
]
BODY = "\n\n".join(TOPICS)


def make_doc(text: str, content_hash: str = "abc123"):
    meta = DocumentMeta(
        source_path="data/practical.pdf",
        title="Practical File",
        doc_type="pdf",
        content_hash=content_hash,
        ingested_at=datetime.now(UTC),
        page_count=2,
    )
    sections = [
        {"heading": "Page 1", "level": 1, "page": 1, "start": 0},
        {"heading": "Page 2", "level": 1, "page": 2, "start": len(text) // 2},
    ]
    return LoadedDocument(text, meta, sections)


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=DIM)


@pytest.fixture
def store():
    store = VectorStore.connect(":memory:", "test_chunks", DIM, fingerprint="test:512:64")
    store.ensure_collection()
    return store


@pytest.fixture
def chunks():
    return FixedSizeChunker(chunk_size=16, overlap=2).split(make_doc(BODY))


def test_point_id_is_deterministic():
    assert point_id("a" * 32) == point_id("a" * 32)


def test_point_id_differs_per_chunk():
    assert point_id("a" * 32) != point_id("b" * 32)


def test_collection_is_created(store):
    assert store.exists()
    assert store.count() == 0


def test_upsert_indexes_every_chunk(store, chunks, embedder):
    written = store.upsert(chunks, embedder.embed_chunks(chunks))
    assert written == len(chunks)
    assert store.count() == len(chunks)


def test_upsert_is_idempotent(store, chunks, embedder):
    """FR-6: re-ingesting an unchanged document must not duplicate points."""
    vectors = embedder.embed_chunks(chunks)
    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)


def test_upsert_rejects_mismatched_lengths(store, chunks, embedder):
    with pytest.raises(ValueError, match="chunks but"):
        store.upsert(chunks, embedder.embed_chunks(chunks)[:-1])


def test_upsert_rejects_wrong_dimension(store, chunks):
    with pytest.raises(ValueError, match="dimensions"):
        store.upsert(chunks[:1], [[0.0] * (DIM + 5)])


def test_upsert_of_nothing_is_a_no_op(store):
    assert store.upsert([], []) == 0


def test_search_returns_the_nearest_chunk_first(store, chunks, embedder):
    store.upsert(chunks, embedder.embed_chunks(chunks))
    hits = store.search(embedder.embed_documents([chunks[0].embedding_text()])[0], top_k=3)
    assert hits
    assert hits[0].chunk.id == chunks[0].id


def test_search_respects_top_k(store, chunks, embedder):
    store.upsert(chunks, embedder.embed_chunks(chunks))
    assert len(store.search(embedder.embed_query("sets"), top_k=2)) <= 2


def test_search_preserves_chunk_metadata(store, chunks, embedder):
    store.upsert(chunks, embedder.embed_chunks(chunks))
    hit = store.search(embedder.embed_documents([chunks[0].embedding_text()])[0], top_k=1)[0]
    assert hit.chunk.page_number == chunks[0].page_number
    assert hit.chunk.contextual_header == chunks[0].contextual_header
    assert hit.chunk.document_id == chunks[0].document_id


def test_search_scores_are_populated(store, chunks, embedder):
    store.upsert(chunks, embedder.embed_chunks(chunks))
    hit = store.search(embedder.embed_query("relations"), top_k=1)[0]
    assert hit.dense_score is not None
    assert hit.final_score == hit.dense_score


def test_search_can_filter_by_metadata(store, chunks, embedder):
    store.upsert(chunks, embedder.embed_chunks(chunks))
    hits = store.search(embedder.embed_query("sets"), top_k=10, filters={"page_number": 1})
    assert hits
    assert all(h.chunk.page_number == 1 for h in hits)


def test_search_on_an_empty_collection_returns_nothing(store, embedder):
    assert store.search(embedder.embed_query("anything"), top_k=5) == []


def test_fingerprint_marker_is_never_returned_as_a_result(store, chunks, embedder):
    """The fingerprint is stored as a point; it must not leak into search results."""
    store.upsert(chunks, embedder.embed_chunks(chunks))
    hits = store.search([0.0] * DIM, top_k=20)
    assert all(h.chunk.id for h in hits)
    assert all("__index_fingerprint__" not in h.chunk.text for h in hits)


def test_mismatched_fingerprint_is_rejected(store):
    """Changing the embedding model invalidates stored vectors — fail loudly."""
    other = VectorStore(store.client, store.collection, DIM, fingerprint="other-model:256:32")
    with pytest.raises(IndexFingerprintMismatchError, match="not comparable"):
        other.ensure_collection()


def test_matching_fingerprint_is_accepted(store):
    same = VectorStore(store.client, store.collection, DIM, fingerprint="test:512:64")
    same.ensure_collection()
    assert same.exists()


def test_delete_document_removes_its_chunks(store, chunks, embedder):
    store.upsert(chunks, embedder.embed_chunks(chunks))
    store.delete_document(chunks[0].document_id)
    assert store.count() == 0
