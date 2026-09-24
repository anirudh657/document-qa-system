"""Hybrid retrieval and reciprocal rank fusion tests."""

import pytest

from rag.ingestion.embedding import HashEmbedder
from rag.models import Chunk, RetrievedChunk
from rag.retrieval.hybrid import RRF_K, HybridRetriever, reciprocal_rank_fusion
from rag.retrieval.sparse import SparseRetriever
from rag.retrieval.store import VectorStore

DIM = 32


def hit(chunk_id: str, dense: float | None = None, sparse: float | None = None):
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, document_id="d", text=f"text {chunk_id}"),
        dense_score=dense,
        sparse_score=sparse,
    )


# --- fusion ----------------------------------------------------------------


def test_fusion_of_one_list_preserves_its_order():
    fused = reciprocal_rank_fusion([[hit("a", 0.9), hit("b", 0.8)]])
    assert [h.chunk.id for h in fused] == ["a", "b"]


def test_a_chunk_found_by_both_retrievers_wins():
    """The whole point of hybrid: agreement outranks either list alone."""
    dense = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    sparse = [hit("c", sparse=5.0), hit("d", sparse=3.0)]
    assert reciprocal_rank_fusion([dense, sparse])[0].chunk.id == "c"


def test_fusion_deduplicates_by_chunk_id():
    fused = reciprocal_rank_fusion([[hit("a", 0.9)], [hit("a", sparse=5.0)]])
    assert len(fused) == 1


def test_a_merged_chunk_keeps_both_scores():
    """Useful when inspecting why something ranked where it did."""
    merged = reciprocal_rank_fusion([[hit("a", 0.9)], [hit("a", sparse=5.0)]])[0]
    assert merged.dense_score == 0.9
    assert merged.sparse_score == 5.0


def test_fusion_scores_are_recorded():
    fused = reciprocal_rank_fusion([[hit("a", 0.9)]])
    assert fused[0].fusion_score == pytest.approx(1 / (RRF_K + 1))


def test_final_score_prefers_fusion_over_dense():
    fused = reciprocal_rank_fusion([[hit("a", 0.9)]])
    assert fused[0].final_score == fused[0].fusion_score


def test_fusion_ignores_the_magnitude_of_scores():
    """RRF uses ranks only — an enormous BM25 score must not overwhelm the merge."""
    a = reciprocal_rank_fusion([[hit("a", 0.51)], [hit("b", sparse=999.0)]])
    b = reciprocal_rank_fusion([[hit("a", 0.99)], [hit("b", sparse=0.01)]])
    assert [h.fusion_score for h in a] == [h.fusion_score for h in b]


def test_a_larger_k_flattens_the_gap_between_ranks():
    tight = reciprocal_rank_fusion([[hit("a"), hit("b")]], k=1)
    flat = reciprocal_rank_fusion([[hit("a"), hit("b")]], k=1000)
    tight_gap = tight[0].fusion_score - tight[1].fusion_score
    flat_gap = flat[0].fusion_score - flat[1].fusion_score
    assert tight_gap > flat_gap


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([]) == []


def test_fusion_tolerates_an_empty_list():
    assert len(reciprocal_rank_fusion([[hit("a", 0.9)], []])) == 1


# --- hybrid retriever ------------------------------------------------------


def chunk_id(letter: str) -> str:
    """Qdrant point ids must be UUID-shaped, so chunk ids are 32 hex characters."""
    return letter * 32


@pytest.fixture
def chunks():
    rows = [
        ("a", "Experiment 1 (a): Array Operations", "oop.pdf", 2),
        ("b", "Create a user account and grant privileges", "sql.pdf", 3),
        ("c", "Step 1: Create a Database. CREATE DATABASE startersql;", "sql.pdf", 5),
        ("d", "Indexes speed up lookups with CREATE INDEX on columns", "sql.pdf", 64),
    ]
    return [
        Chunk(
            id=chunk_id(letter),
            document_id="doc",
            text=text,
            source=src,
            page_number=page,
        )
        for letter, text, src, page in rows
    ]


@pytest.fixture
def store(chunks):
    embedder = HashEmbedder(dimension=DIM)
    store = VectorStore.connect(":memory:", "hybrid_test", DIM, fingerprint="t:200:64")
    store.ensure_collection()
    store.upsert(chunks, embedder.embed_chunks(chunks))
    return store


def test_store_can_read_its_chunks_back(store, chunks):
    assert {c.id for c in store.iter_chunks()} == {c.id for c in chunks}


def test_read_back_chunks_keep_their_text_and_source(store):
    read = {c.id: c for c in store.iter_chunks()}
    assert read[chunk_id("c")].text.startswith("Step 1")
    assert read[chunk_id("c")].source == "sql.pdf"


def test_the_fingerprint_marker_is_not_returned_as_a_chunk(store, chunks):
    assert len(store.iter_chunks()) == len(chunks)


def test_hybrid_falls_back_to_dense_without_a_sparse_index(store):
    retriever = HybridRetriever(store, HashEmbedder(dimension=DIM), sparse=None)
    assert retriever.search("create a database", top_k=3)


def test_hybrid_falls_back_to_dense_on_an_empty_sparse_corpus(store):
    retriever = HybridRetriever(store, HashEmbedder(dimension=DIM), SparseRetriever([]))
    assert retriever.search("create a database", top_k=3)


def test_hybrid_surfaces_an_exact_match_dense_alone_would_miss(store, chunks):
    """HashEmbedder carries no meaning, so any hit here comes from BM25."""
    retriever = HybridRetriever(store, HashEmbedder(dimension=DIM), SparseRetriever(chunks))
    assert chunk_id("a") in [h.chunk.id for h in retriever.search("Experiment 1", top_k=3)]


def test_hybrid_respects_top_k(store, chunks):
    retriever = HybridRetriever(store, HashEmbedder(dimension=DIM), SparseRetriever(chunks))
    assert len(retriever.search("create", top_k=2)) == 2


def test_hybrid_results_carry_a_fusion_score(store, chunks):
    retriever = HybridRetriever(store, HashEmbedder(dimension=DIM), SparseRetriever(chunks))
    assert retriever.search("create a database", top_k=1)[0].fusion_score is not None
