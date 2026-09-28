"""Reranking tests.

ScriptedReranker throughout, so nothing downloads a model. These test the reranking
*pipeline* — candidate widening, score assignment, ordering, fallback — not the
quality of any particular cross-encoder.
"""

import pytest

from rag.models import Chunk, RetrievedChunk
from rag.retrieval.rerank import Reranker, RerankingRetriever, ScriptedReranker


def hit(chunk_id: str, dense: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, document_id="d", text=f"passage {chunk_id}"),
        dense_score=dense,
    )


class StubRetriever:
    """Returns a fixed list, recording how many results were asked for."""

    def __init__(self, hits: list[RetrievedChunk]):
        self.hits = hits
        self.requested: list[int] = []

    def search(self, query: str, top_k: int = 5, filters=None) -> list[RetrievedChunk]:
        self.requested.append(top_k)
        return self.hits[:top_k]


@pytest.fixture
def pool():
    return [hit(letter * 32) for letter in "abcdefghij"]


def ids(hits):
    return [h.chunk.id[0] for h in hits]


# --- reranker ---------------------------------------------------------------


def test_reranking_reorders_by_score(pool):
    reranked = ScriptedReranker(order=[pool[4].chunk.id]).rerank("q", pool, top_n=3)
    assert ids(reranked)[0] == "e"


def test_reranking_respects_top_n(pool):
    assert len(ScriptedReranker().rerank("q", pool, top_n=3)) == 3


def test_every_candidate_gets_a_rerank_score(pool):
    ScriptedReranker().rerank("q", pool, top_n=3)
    assert all(h.rerank_score is not None for h in pool)


def test_rerank_score_takes_priority_in_final_score(pool):
    """Downstream code sorts on final_score, so the new score must win."""
    reranked = ScriptedReranker(order=[pool[2].chunk.id]).rerank("q", pool, top_n=1)
    assert reranked[0].final_score == reranked[0].rerank_score


def test_reranking_nothing_returns_nothing():
    assert ScriptedReranker().rerank("q", [], top_n=5) == []


def test_a_single_candidate_is_returned_unchanged(pool):
    reranked = Reranker().rerank("q", pool[:1], top_n=5)
    assert len(reranked) == 1
    assert reranked[0].rerank_score == 1.0


def test_the_real_reranker_does_not_load_its_model_on_construction():
    """The cross-encoder is a few hundred megabytes."""
    assert Reranker()._model is None


# --- reranking retriever ----------------------------------------------------


def test_the_candidate_pool_is_wider_than_top_k(pool):
    """Reranking can only reorder what it is given.

    If the pool were truncated to top_k first, a passage the first-stage retriever
    ranked eighth could never be promoted — which is the entire reason to rerank.
    """
    retriever = StubRetriever(pool)
    RerankingRetriever(retriever, ScriptedReranker(), candidates=10).search(
        "q", top_k=3
    )
    assert retriever.requested == [10]


def test_top_k_is_respected_after_reranking(pool):
    result = RerankingRetriever(
        StubRetriever(pool), ScriptedReranker(), candidates=10
    ).search("q", top_k=4)
    assert len(result) == 4


def test_reranking_can_promote_a_low_ranked_candidate(pool):
    """The adjacency failures in the golden set need exactly this."""
    last = pool[-1].chunk.id
    result = RerankingRetriever(
        StubRetriever(pool), ScriptedReranker(order=[last]), candidates=10
    ).search("q", top_k=3)
    assert result[0].chunk.id == last


def test_the_reranker_sees_the_whole_pool_not_just_top_k(pool):
    reranker = ScriptedReranker()
    RerankingRetriever(StubRetriever(pool), reranker, candidates=10).search(
        "q", top_k=2
    )
    assert reranker.calls[0][1] == 10


def test_an_empty_result_skips_the_reranker():
    reranker = ScriptedReranker()
    assert RerankingRetriever(StubRetriever([]), reranker).search("q", top_k=5) == []
    assert reranker.calls == []


def test_a_failing_reranker_falls_back_to_the_original_order(pool):
    """A worse ranking beats no answer at all."""
    wrapper = RerankingRetriever(
        StubRetriever(pool), ScriptedReranker(fail=True), candidates=10
    )
    result = wrapper.search("q", top_k=3)
    assert ids(result) == ["a", "b", "c"]


def test_a_reranker_failure_is_recorded(pool):
    """Silent degradation is the thing to avoid — the trace must say it happened."""
    wrapper = RerankingRetriever(
        StubRetriever(pool), ScriptedReranker(fail=True), candidates=10
    )
    wrapper.search("q", top_k=3)
    assert "RuntimeError" in wrapper.last_error


def test_a_successful_rerank_clears_a_previous_error(pool):
    wrapper = RerankingRetriever(StubRetriever(pool), ScriptedReranker(), candidates=10)
    wrapper.last_error = "stale"
    wrapper.search("q", top_k=3)
    assert wrapper.last_error == ""


def test_candidates_never_below_top_k(pool):
    """Asking for fewer candidates than results would silently truncate."""
    retriever = StubRetriever(pool)
    RerankingRetriever(retriever, ScriptedReranker(), candidates=2).search("q", top_k=5)
    assert retriever.requested == [5]