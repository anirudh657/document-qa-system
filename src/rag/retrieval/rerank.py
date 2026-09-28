"""Cross-encoder reranking.

**Why a second scoring pass exists.**

The embedder compresses a query into one vector and each chunk into another, then
compares them. Everything the model knew about word order, which term modifies which,
and what the passage is actually claiming is gone by the time the comparison happens.
Two vectors near each other means "these are about the same topic", not "this passage
answers this question".

A cross-encoder feeds the query and one passage through the model *together*, so
attention runs across both and it can judge the actual relationship. That is far more
accurate and far more expensive: it is one forward pass per candidate, so it cannot
run over the whole corpus. It reranks a shortlist the cheap retrievers produced.

**The failure this is meant to fix**, from the project's own golden set: six of eleven
retrieval failures returned the page *directly beside* the correct one — expected 63,
got 62; expected 30, got 29; expected 65, got 64 and 66. The retriever finds the right
chapter and picks the wrong page in it. That is a ranking-precision problem, which is
what a cross-encoder addresses. It cannot fix the other failures, where the correct
page never entered the candidate pool at all.

**What it cannot do.** Reranking only reorders. If retrieval missed the page, no
amount of rescoring recovers it — which is why hit rate at a wide candidate count is
the ceiling on what this can achieve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag.models import RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence


class Reranker:
    """Scores (query, passage) pairs jointly with a cross-encoder.

    The model is loaded lazily: construction stays cheap so the settings object can be
    built in tests and CLI paths that never rerank anything.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> list[RetrievedChunk]:
        """Rescore candidates against the query and return the best `top_n`.

        The passage sent to the model is `embedding_text()`, not the bare chunk text,
        so the contextual header ("MySQL Handbook > Page 63") is part of what gets
        judged — consistent with how the chunk was embedded and retrieved.
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            candidates[0].rerank_score = 1.0
            return list(candidates)

        pairs = [(query, hit.chunk.embedding_text()) for hit in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        for hit, score in zip(candidates, scores, strict=True):
            hit.rerank_score = float(score)

        return sorted(candidates, key=lambda hit: hit.rerank_score or 0.0, reverse=True)[:top_n]


class ScriptedReranker:
    """Test double that reorders by a fixed chunk-id ranking. No model download.

    Exists so the reranking *pipeline* — candidate widening, score assignment, fallback
    on failure — can be tested without a 400 MB download in CI. It carries no semantic
    judgement whatsoever.
    """

    def __init__(self, order: Sequence[str] | None = None, fail: bool = False):
        self.order = list(order or [])
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(candidates)))
        if self.fail:
            raise RuntimeError("reranker unavailable")

        def rank(hit: RetrievedChunk) -> int:
            return self.order.index(hit.chunk.id) if hit.chunk.id in self.order else len(self.order)

        ranked = sorted(candidates, key=rank)
        for position, hit in enumerate(ranked):
            hit.rerank_score = 1.0 - position / max(len(ranked), 1)
        return ranked[:top_n]


class RerankingRetriever:
    """Wraps any retriever with a cross-encoder rescoring pass.

    Asks the inner retriever for `candidates` results, reranks them, and returns
    `top_n`. The gap between those numbers is the whole point: reranking can only
    reorder what it is given, so the candidate pool must be wide enough to contain the
    right answer even when the first-stage retriever ranked it poorly.

    A reranker failure degrades to the original order rather than failing the request.
    A slightly worse ranking is a far better outcome than no answer, and the trace
    records that it happened.
    """

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        candidates: int = 25,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates
        self.last_error: str = ""

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        pool: list[RetrievedChunk] = self.retriever.search(
            query, top_k=max(self.candidates, top_k), filters=filters
        )
        if not pool:
            return []

        self.last_error = ""
        try:
            reranked: list[RetrievedChunk] = self.reranker.rerank(query, pool, top_n=top_k)
        # A bad rerank must not lose the answer; fall back to the retriever's order.
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return pool[:top_k]
        return reranked
