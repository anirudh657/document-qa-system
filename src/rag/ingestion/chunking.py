"""Chunking strategies.

Chunking is where most retrieval quality is won or lost — more than reranking on many
corpora. A chunk has to be small enough that its embedding means one specific thing,
but large enough to answer a question on its own. Split too finely and every chunk is
a fragment; split too coarsely and the embedding averages several topics into mush.

Phase 1 ships a fixed-size chunker so the pipeline runs end to end. Phase 2 replaces
it with a structure-aware chunker, and both are kept so the Phase 2 ablation can
compare them on the same golden set rather than on intuition.

TODO(phase-2): StructureAwareChunker, parent-child linking for small-to-big retrieval
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from rag.ingestion.loaders import LoadedDocument
from rag.models import Chunk

CHARS_PER_TOKEN = 4.0
"""Rough characters-to-tokens ratio used for budgeting.

Deliberately an approximation. The real tokenizer lives inside the embedding model,
and loading it here would make chunking depend on a multi-hundred-megabyte download
for what is currently a sizing heuristic. English prose sits near 4 characters per
token; code and tables run denser, so this slightly over-estimates chunk size there —
which errs toward smaller chunks, the safer direction.

TODO(phase-2): swap in the embedding model's own tokenizer and re-run the ablation.
"""

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


class Unit(NamedTuple):
    """A sentence or paragraph, with its real offset in the source document.

    The offset is carried rather than searched for later. Looking a chunk's text back
    up with `str.find` returns the FIRST match in the document, and running headers or
    repeated footers ("URN: 2581258" on every page) then attribute a chunk to whatever
    page that boilerplate first appeared on. Tracking the true position is the only
    way page citations stay correct.
    """

    text: str
    start: int


def _split_units(text: str) -> list[Unit]:
    """Break text into the smallest pieces a chunk boundary may fall between.

    Paragraphs first, then sentences within each paragraph. Splitting mid-sentence is
    avoided because a fragment starting halfway through a clause embeds poorly and
    reads badly when shown as a citation.
    """
    units: list[Unit] = []
    cursor = 0

    for paragraph in _PARAGRAPH.split(text):
        if not paragraph.strip():
            cursor += len(paragraph)
            continue

        para_start = text.find(paragraph, cursor)
        if para_start == -1:
            para_start = cursor

        offset = 0
        for piece in _SENTENCE.split(paragraph):
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                units.append(Unit(piece.strip(), para_start + offset + lead))
            offset += len(piece)

        cursor = para_start + len(paragraph)

    return units


def chunk_id(document_id: str, index: int) -> str:
    """Deterministic ID so re-ingesting the same document upserts instead of duplicating.

    Derived from the document hash and the chunk's position, which satisfies FR-6:
    running ingestion twice over an unchanged corpus produces an identical index.
    """
    return hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest()[:32]


class Chunker(ABC):
    @abstractmethod
    def split(self, doc: LoadedDocument) -> list[Chunk]: ...


class FixedSizeChunker(Chunker):
    """Packs whole sentences into a token budget, with overlap between neighbours.

    Overlap exists because a question's answer often straddles a boundary. Repeating
    the tail of each chunk at the head of the next means a fact near a boundary is
    retrievable from either side, at the cost of some duplicate text in the index.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        token_counter: Callable[[str], int] = estimate_tokens,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
                "equal or larger values never advance and would loop forever"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.count_tokens = token_counter

    def split(self, doc: LoadedDocument) -> list[Chunk]:
        units = _split_units(doc.text)
        if not units:
            return []

        document_id = doc.meta.content_hash
        source = Path(doc.meta.source_path).name
        groups = self._pack(units)

        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            text = " ".join(unit.text for unit in group)
            page, path = self._locate(doc, group[0].start)
            chunks.append(
                Chunk(
                    id=chunk_id(document_id, index),
                    document_id=document_id,
                    text=text,
                    source=source,
                    contextual_header=self._header(doc, path),
                    section_path=path,
                    page_number=page,
                    token_count=self.count_tokens(text),
                )
            )
        return chunks

    def _pack(self, units: list[Unit]) -> list[list[Unit]]:
        """Greedily fill chunks to the token budget, then step back for overlap."""
        groups: list[list[Unit]] = []
        current: list[Unit] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit.text)

            if current and current_tokens + unit_tokens > self.chunk_size:
                groups.append(current)
                current = self._overlap_tail(current)
                current_tokens = sum(self.count_tokens(u.text) for u in current)

            current.append(unit)
            current_tokens += unit_tokens

            # A single unit larger than the budget cannot be split further without
            # breaking mid-sentence, so it is emitted alone and left oversized.
            if current_tokens >= self.chunk_size and len(current) == 1:
                groups.append(current)
                current = []
                current_tokens = 0

        if current:
            groups.append(current)
        return groups

    def _overlap_tail(self, group: list[Unit]) -> list[Unit]:
        """Take trailing units from a finished chunk, up to the overlap budget."""
        tail: list[Unit] = []
        budget = 0
        for unit in reversed(group):
            tokens = self.count_tokens(unit.text)
            if budget + tokens > self.overlap:
                break
            tail.insert(0, unit)
            budget += tokens
        return tail

    @staticmethod
    def _locate(doc: LoadedDocument, offset: int) -> tuple[int | None, list[str]]:
        """Find the page number and heading path covering a character offset.

        Sections are recorded in document order by the loaders, so the last section
        starting at or before the offset is the one containing it.
        """
        if offset < 0 or not doc.sections:
            return None, []

        current: dict[str, Any] | None = None
        for section in doc.sections:
            if section["start"] <= offset:
                current = section
            else:
                break

        if current is None:
            return None, []

        page = current.get("page")
        heading = current.get("heading")
        return page, [heading] if heading else []

    @staticmethod
    def _header(doc: LoadedDocument, path: list[str]) -> str:
        """Build the contextual header prepended before embedding.

        "The window is 30 days" is unretrievable on its own. Headed with its document
        title and section path, the same sentence carries the context a query needs.
        """
        parts = [doc.meta.title] if doc.meta.title else []
        parts.extend(path)
        return " > ".join(p for p in parts if p)
