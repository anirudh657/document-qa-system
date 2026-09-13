"""Qdrant vector store: persistence and nearest-neighbour search over chunk embeddings.

Without this, embeddings live in a Python list and vanish when the process exits.
Qdrant keeps them on disk, searches them in sub-linear time, and filters by metadata —
none of which a list can do once the corpus grows past a few hundred chunks.

Two design points worth stating plainly:

**Index fingerprinting.** A collection is stamped with the embedding model and chunking
settings that produced it. Change the embedding model and the stored vectors become
meaningless — not wrong in a way that errors, wrong in a way that silently returns
nonsense. The fingerprint is checked at startup so that mismatch is caught loudly.

**Deterministic point IDs.** Chunk IDs are derived from the document hash plus position,
so re-ingesting an unchanged document overwrites the same points instead of appending
duplicates. That is FR-6, idempotent ingestion.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from rag.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

FINGERPRINT_KEY = "__index_fingerprint__"


class IndexFingerprintMismatchError(RuntimeError):
    """Raised when a collection was built with different embedding or chunking settings.

    Serving queries against such an index produces plausible-looking but meaningless
    results, so this fails loudly rather than degrading quietly.
    """


def point_id(chunk_id: str) -> str:
    """Map a chunk ID to a Qdrant point ID.

    Qdrant accepts only unsigned integers or UUIDs. Chunk IDs are 32 hex characters,
    which is exactly a UUID's worth, so the mapping is lossless and deterministic —
    the same chunk always lands on the same point.
    """
    return str(uuid.UUID(hex=chunk_id[:32]))


class VectorStore:
    """Thin wrapper over a Qdrant collection holding chunk vectors and their payloads."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dimension: int,
        fingerprint: str = "",
    ):
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self.fingerprint = fingerprint

    @classmethod
    def connect(
        cls,
        url: str,
        collection: str,
        dimension: int,
        fingerprint: str = "",
    ) -> VectorStore:
        from qdrant_client import QdrantClient

        client = QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url)
        return cls(client, collection, dimension, fingerprint)

    def exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if absent, or verify its fingerprint if present."""
        from qdrant_client import models

        if recreate and self.exists():
            self.client.delete_collection(self.collection)

        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            self._write_fingerprint()
            return

        self._verify_fingerprint()

    def _fingerprint_point_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.collection}:fingerprint"))

    def _write_fingerprint(self) -> None:
        from qdrant_client import models

        if not self.fingerprint:
            return
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=self._fingerprint_point_id(),
                    vector=[0.0] * self.dimension,
                    payload={FINGERPRINT_KEY: self.fingerprint},
                )
            ],
        )

    def _verify_fingerprint(self) -> None:
        if not self.fingerprint:
            return
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._fingerprint_point_id()],
            with_payload=True,
        )
        if not records:
            self._write_fingerprint()
            return

        stored = (records[0].payload or {}).get(FINGERPRINT_KEY)
        if stored and stored != self.fingerprint:
            raise IndexFingerprintMismatchError(
                f"Collection '{self.collection}' was built with {stored!r} but the current "
                f"configuration is {self.fingerprint!r}. The stored vectors are not comparable "
                "to newly embedded queries. Re-ingest with recreate=True."
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Write chunks and their vectors. Re-running with the same chunks overwrites."""
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0

        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"vector has {len(vector)} dimensions, collection expects {self.dimension}"
                )

        points = [
            models.PointStruct(
                id=point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "contextual_header": chunk.contextual_header,
                    "section_path": chunk.section_path,
                    "page_number": chunk.page_number,
                    "parent_id": chunk.parent_id,
                    "token_count": chunk.token_count,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks to a query vector, highest score first."""
        from qdrant_client import models

        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            if FINGERPRINT_KEY in payload:
                continue
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=payload.get("chunk_id", ""),
                        document_id=payload.get("document_id", ""),
                        text=payload.get("text", ""),
                        contextual_header=payload.get("contextual_header", ""),
                        section_path=payload.get("section_path") or [],
                        page_number=payload.get("page_number"),
                        parent_id=payload.get("parent_id"),
                        token_count=payload.get("token_count", 0),
                    ),
                    dense_score=point.score,
                )
            )
        return results

    def count(self) -> int:
        """Number of indexed chunks, excluding the fingerprint marker."""
        total = self.client.count(collection_name=self.collection, exact=True).count
        return max(0, total - (1 if self.fingerprint else 0))

    def delete_document(self, document_id: str) -> None:
        """Remove every chunk belonging to one document, for re-indexing."""
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
        )
