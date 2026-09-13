"""Embedder tests.

These run against HashEmbedder so the suite never downloads a model. The real
SentenceTransformerEmbedder is exercised by an integration test that is skipped
unless the model is already cached locally.
"""

from datetime import UTC, datetime

import pytest

from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.embedding import (
    BGE_QUERY_PREFIX,
    HashEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    l2_normalise,
)
from rag.ingestion.loaders import LoadedDocument
from rag.models import DocumentMeta


def make_doc(text: str):
    meta = DocumentMeta(
        source_path="data/practical.pdf",
        title="Practical File",
        doc_type="pdf",
        content_hash="abc123",
        ingested_at=datetime.now(UTC),
        page_count=1,
    )
    return LoadedDocument(text, meta, [{"heading": "Page 1", "level": 1, "page": 1, "start": 0}])


def test_l2_normalise_gives_unit_length():
    vector = l2_normalise([3.0, 4.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_l2_normalise_handles_the_zero_vector():
    assert l2_normalise([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_similarity_of_identical_vectors_is_one():
    embedder = HashEmbedder(dimension=64)
    vector = embedder.embed_query("sets and relations")
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_embeddings_have_the_declared_dimension():
    embedder = HashEmbedder(dimension=128)
    assert len(embedder.embed_query("anything")) == 128
    assert all(len(v) == 128 for v in embedder.embed_documents(["a", "b"]))


def test_embedding_is_deterministic():
    embedder = HashEmbedder(dimension=64)
    assert embedder.embed_query("de morgan") == embedder.embed_query("de morgan")


def test_different_text_gives_different_vectors():
    embedder = HashEmbedder(dimension=64)
    assert embedder.embed_query("sets") != embedder.embed_query("graphs")


def test_embedding_an_empty_batch_returns_an_empty_list():
    assert HashEmbedder().embed_documents([]) == []


def test_batch_size_does_not_change_the_result():
    embedder = HashEmbedder(dimension=64)
    texts = [f"sentence number {n}" for n in range(10)]
    one_by_one = [embedder.embed_documents([t])[0] for t in texts]
    all_at_once = embedder.embed_documents(texts)
    assert one_by_one == all_at_once


def test_embed_chunks_uses_the_contextual_header():
    """A chunk must be embedded with its header, not its bare text.

    If this regresses, retrieval silently degrades: chunks lose the title and section
    context that makes them findable, and nothing errors to say so.
    """
    embedder = HashEmbedder(dimension=64)
    chunks = FixedSizeChunker().split(make_doc("Sets and relations are covered here in detail."))

    from_chunks = embedder.embed_chunks(chunks)
    from_header = embedder.embed_documents([chunks[0].embedding_text()])
    from_bare_text = embedder.embed_documents([chunks[0].text])

    assert from_chunks[0] == from_header[0]
    assert from_chunks[0] != from_bare_text[0]


def test_query_prefix_is_configured_for_bge():
    embedder = SentenceTransformerEmbedder()
    assert embedder.query_prefix == BGE_QUERY_PREFIX


def test_real_embedder_does_not_load_the_model_on_construction():
    """Construction must stay cheap — the model is hundreds of megabytes."""
    embedder = SentenceTransformerEmbedder()
    assert embedder._model is None
