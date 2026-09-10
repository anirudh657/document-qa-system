"""Dense vector retrieval over Qdrant.

Strong on paraphrase and conceptual similarity. Weak on exact identifiers, product
codes and rare proper nouns — which is precisely what sparse.py is for.

TODO(phase-1): search(), collection setup, metadata filters
"""

from rag.models import RetrievedChunk


class DenseRetriever:
    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievedChunk]:
        raise NotImplementedError("TODO(phase-1)")
