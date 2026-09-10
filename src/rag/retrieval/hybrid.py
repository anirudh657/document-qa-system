"""Hybrid retrieval via reciprocal rank fusion.

RRF is used rather than weighted score interpolation because BM25 scores and cosine
similarities are not on comparable scales, and normalising them introduces a tuning
parameter that has to be re-fit per corpus. RRF only needs the ranks.

    score(d) = sum over retrievers of 1 / (k + rank(d))     with k = 60

TODO(phase-2): fuse(), parallel dispatch to dense and sparse
"""

from typing import Any

from rag.models import RetrievedChunk

RRF_K = 60


class HybridRetriever:
    def search(
        self, query: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("TODO(phase-2)")

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[RetrievedChunk]], k: int = RRF_K
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("TODO(phase-2)")
