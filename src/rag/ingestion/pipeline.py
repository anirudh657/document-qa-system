"""Ingestion pipeline: load, chunk, embed, upsert.

Idempotency is a requirement (FR-6), not a nicety: chunk IDs are derived
deterministically from the document hash and chunk index, so re-running ingestion
upserts in place instead of duplicating.

TODO(phase-1): run(), batch embedding, deterministic chunk IDs
TODO(phase-4): async worker queue, incremental re-index by content hash
"""

from pathlib import Path
from typing import Any


class IngestionPipeline:
    def run(self, source: Path) -> dict[str, Any]:
        """Index a file or directory. Returns a summary of what changed."""
        raise NotImplementedError("TODO(phase-1)")

    def content_hash(self, path: Path) -> str:
        raise NotImplementedError("TODO(phase-1)")
