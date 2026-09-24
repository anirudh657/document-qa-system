"""Hybrid retrieval via reciprocal rank fusion.

Dense and sparse retrieval fail in different places. Dense misses rare literal terms;
sparse misses paraphrase. Running both and merging the results covers each one's blind
spot with the other's strength.

**Why RRF rather than combining the scores.**

BM25 scores and cosine similarities are not comparable. Cosine sits in [0, 1] and
clusters tightly — most of your real results land between 0.5 and 0.8. BM25 is
unbounded and corpus-dependent, and a rare term can score 20 where a common one scores
0.4. Interpolating them means first normalising two different distributions and then
fitting a weight, and that weight has to be refit for every corpus.

RRF sidesteps all of it by throwing the scores away and using only the ranks:

    score(d) = sum over retrievers of 1 / (k + rank(d))      with k = 60

A document ranked first by either retriever scores 1/61. One ranked first by both
scores 2/61 and wins. The constant k flattens the curve so the gap between rank 1 and
rank 2 is not overwhelming — without it, a single retriever's top hit could never be
beaten by a document both retrievers agreed on at rank 2. k = 60 is the value from the
original RRF paper and is not worth tuning before there is a golden set large enough
to tune it against.

The cost is real and worth stating: RRF cannot tell a confident match from a marginal
one. A retriever that is barely above threshold contributes the same rank-1 weight as
one that is certain. Reranking (Phase 2, rerank.py) is what recovers that signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag.models import RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag.ingestion.embedding import Embedder
    from rag.retrieval.sparse import SparseRetriever
    from rag.retrieval.store import VectorStore

RRF_K = 60
"""Rank-smoothing constant from the original RRF paper.

Larger flattens the difference between ranks, so agreement between retrievers matters
more than either one's confidence. Smaller lets a single retriever's top hit dominate.
"""


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Merge ranked lists into one, scoring each chunk by 1/(k + rank) summed across lists.

    Chunks are identified by chunk id, so the same chunk appearing in both lists is
    merged rather than duplicated, and it keeps whichever scores each retriever gave it
    — useful when inspecting why something ranked where it did.
    """
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = hit.chunk.id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            if key not in fused:
                fused[key] = RetrievedChunk(
                    chunk=hit.chunk,
                    dense_score=hit.dense_score,
                    sparse_score=hit.sparse_score,
                )
            else:
                existing = fused[key]
                if hit.dense_score is not None:
                    existing.dense_score = hit.dense_score
                if hit.sparse_score is not None:
                    existing.sparse_score = hit.sparse_score

    for key, hit in fused.items():
        hit.fusion_score = scores[key]

    return sorted(fused.values(), key=lambda hit: scores[hit.chunk.id], reverse=True)


class HybridRetriever:
    """Dense and sparse retrieval, fused.

    Each retriever is asked for `candidates` results rather than `top_k`, because a
    chunk ranked eighth by one retriever and second by the other should still be able
    to win the fusion — and it cannot if the lists were truncated to five first.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        sparse: SparseRetriever | None = None,
        candidates: int = 25,
        k: int = RRF_K,
    ):
        self.store = store
        self.embedder = embedder
        self.sparse = sparse
        self.candidates = candidates
        self.k = k

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        dense = self.store.search(
            self.embedder.embed_query(query), top_k=self.candidates, filters=filters
        )

        if self.sparse is None or not len(self.sparse):
            return dense[:top_k]

        sparse = self.sparse.search(query, top_k=self.candidates)
        if not sparse:
            return dense[:top_k]

        return reciprocal_rank_fusion([dense, sparse], k=self.k)[:top_k]


def build_sparse(store: VectorStore) -> SparseRetriever:
    """Build a sparse index from everything currently in the vector store."""
    from rag.retrieval.sparse import SparseRetriever

    return SparseRetriever(store.iter_chunks())
