"""BM25 sparse retrieval.

Exists because dense retrieval misses exact-match queries. The two have
complementary failure modes; that complementarity is the entire argument for hybrid.

TODO(phase-2): index building, persistence, search
"""

from rag.models import RetrievedChunk


class SparseRetriever:
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("TODO(phase-2)")
