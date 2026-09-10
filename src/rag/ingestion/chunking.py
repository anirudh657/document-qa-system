"""Chunking strategies.

Phase 1 ships a fixed-size chunker so the pipeline runs end to end. Phase 2 replaces
it with a structure-aware chunker, which is where most of the retrieval quality
actually comes from — more than reranking on many corpora. Both are kept so the
ablation in Phase 2 can compare them fairly.

TODO(phase-1): FixedSizeChunker
TODO(phase-2): StructureAwareChunker, contextual headers, parent-child linking
"""

from abc import ABC, abstractmethod

from rag.ingestion.loaders import LoadedDocument
from rag.models import Chunk


class Chunker(ABC):
    @abstractmethod
    def split(self, doc: LoadedDocument) -> list[Chunk]: ...


class FixedSizeChunker(Chunker):
    """Split by token budget with overlap, respecting sentence boundaries."""

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: LoadedDocument) -> list[Chunk]:
        raise NotImplementedError("TODO(phase-1)")
