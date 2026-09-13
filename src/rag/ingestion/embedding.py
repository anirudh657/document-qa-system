"""Embedding: turning text into vectors that can be compared by meaning.

An embedding maps a piece of text to a fixed-length list of numbers positioned so that
texts meaning similar things land near each other. That proximity is what makes
"search by meaning" possible: the query becomes a vector too, and retrieval is then
just finding the nearest chunk vectors.

Two details here are easy to get wrong and expensive to debug:

1. **Query/document asymmetry.** BGE models are trained with an instruction prefix on
   the query side only. Embedding a question the same way as a passage measurably hurts
   retrieval, and the failure is silent — results are simply worse, nothing errors.

2. **Normalisation.** Vectors are L2-normalised, so cosine similarity reduces to a dot
   product. Qdrant is configured for cosine distance, and feeding it unnormalised
   vectors produces subtly wrong rankings rather than an error.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from rag.models import Chunk

if TYPE_CHECKING:
    from collections.abc import Iterable

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
"""Instruction prefix BGE expects on queries but not on passages.

Documented in the model card. Omitting it costs a few points of retrieval accuracy;
applying it to passages as well costs about the same. It belongs on queries only.
"""


class Embedder(ABC):
    """Interface every embedding backend implements.

    Split into query and document methods rather than one `embed` call precisely
    because the two are not interchangeable — see the asymmetry note above. Making
    that distinction part of the interface means a caller cannot accidentally ignore it.
    """

    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    def embed_chunks(self, chunks: Iterable[Chunk]) -> list[list[float]]:
        """Embed chunks using their contextual header, not the bare text.

        `Chunk.embedding_text()` prepends the document title and section path. A chunk
        reading "The window is 30 days" is unretrievable alone; headed with
        "Billing Policy > Refunds" it carries the context a query can match against.
        """
        return self.embed_documents([chunk.embedding_text() for chunk in chunks])


def l2_normalise(vector: list[float]) -> list[float]:
    """Scale a vector to unit length so cosine similarity equals the dot product."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


class SentenceTransformerEmbedder(Embedder):
    """Local embedding via sentence-transformers. No API calls, no per-chunk cost.

    The model is loaded lazily on first use. Construction therefore stays cheap, which
    matters because the settings object is built in tests and CLI paths that never
    embed anything, and loading the model costs several hundred megabytes and seconds
    of startup.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        dimension: int = 768,
        batch_size: int = 64,
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self._model: Any = None

    @property
    def model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return list(map(float, vector))


class HashEmbedder(Embedder):
    """Deterministic fake embeddings for tests. Carries no semantic meaning.

    Exists so the test suite and CI never download a real model. Identical text always
    produces an identical vector, which is enough to exercise batching, dimensions,
    normalisation and the ingestion pipeline end to end.

    Never use this in a real index — nearest neighbours would be meaningless.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [
            (digest[index % len(digest)] + index) % 256 / 255.0 for index in range(self.dimension)
        ]
        return l2_normalise(raw)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarity between two vectors, 1.0 identical and 0.0 unrelated.

    Assumes both are normalised, which every embedder here guarantees, so this is a
    plain dot product.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


def build_embedder(model_name: str, dimension: int, batch_size: int = 64) -> Embedder:
    """Construct the configured embedder.

    TODO(phase-4): add hosted-API backends behind this same interface.
    """
    return SentenceTransformerEmbedder(model_name, dimension, batch_size)
