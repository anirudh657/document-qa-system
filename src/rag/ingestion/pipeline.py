"""Ingestion pipeline: load, chunk, embed, upsert.

This is the wiring that turns four independent components into one operation. It lives
here rather than inside the CLI so the API, the eval harness and any future worker all
drive ingestion through the same path — otherwise each grows its own slightly different
copy and they drift.

Failure is per-document, not per-run. One corrupt PDF in a directory of fifty should
cost you that one file and a logged reason, not the other forty-nine.

TODO(phase-4): async worker queue, incremental re-index by content hash (FR-5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.ingestion.chunking import Chunker
from rag.ingestion.embedding import Embedder
from rag.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    EmptyTextLayerError,
    UnsupportedFormatError,
    load,
)
from rag.retrieval.store import VectorStore


@dataclass
class FileResult:
    path: Path
    chunks: int = 0
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.skipped


@dataclass
class IngestionReport:
    results: list[FileResult] = field(default_factory=list)

    @property
    def indexed(self) -> list[FileResult]:
        return [r for r in self.results if r.ok]

    @property
    def skipped(self) -> list[FileResult]:
        return [r for r in self.results if not r.ok]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results)


class IngestionPipeline:
    def __init__(self, chunker: Chunker, embedder: Embedder, store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def discover(self, source: Path, recursive: bool = True) -> list[Path]:
        """List the files that will be ingested, sorted for reproducibility.

        Unsupported extensions are filtered out here rather than raising later, so
        pointing the CLI at a folder containing images and zip files does the obvious
        thing instead of failing.
        """
        if source.is_file():
            return [source]
        pattern = "**/*" if recursive else "*"
        return sorted(
            path
            for path in source.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def ingest_file(self, path: Path) -> FileResult:
        try:
            document = load(path)
        except EmptyTextLayerError as error:
            return FileResult(path, skipped=f"no text layer — {error}")
        except UnsupportedFormatError as error:
            return FileResult(path, skipped=str(error))
        except Exception as error:
            return FileResult(path, skipped=f"{type(error).__name__}: {error}")

        chunks = self.chunker.split(document)
        if not chunks:
            return FileResult(path, skipped="produced no chunks")

        vectors = self.embedder.embed_chunks(chunks)
        self.store.upsert(chunks, vectors)
        return FileResult(path, chunks=len(chunks))

    def run(self, source: Path, recursive: bool = True) -> IngestionReport:
        """Index a file or directory. Returns a per-file summary of what happened."""
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist")

        self.store.ensure_collection()
        report = IngestionReport()
        for path in self.discover(source, recursive):
            report.results.append(self.ingest_file(path))
        return report
