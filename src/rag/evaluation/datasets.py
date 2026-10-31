"""Golden dataset loading and validation.

**Why questions point at pages, not chunk IDs.**

The obvious schema records the chunk IDs a question should retrieve. It does not
survive this project. Chunk IDs are derived from the document hash and the chunk's
position, so changing chunk size — the single most important thing Phase 2 ablates —
renumbers every chunk and invalidates the entire dataset. A golden set you must rewrite
to run the experiment is not a golden set.

Pages are stable. A fact on page 5 stays on page 5 whatever the chunker does, so the
same dataset scores a 256-token run and a 1024-token run and the comparison is real.
Sources are matched by filename for the same reason: paths move, filenames rarely do.

See docs/EVALUATION.md for the rules that keep a dataset honest. The one that matters
most: write the questions from the source documents, before running the system. A
question written after reading model output measures self-agreement, not correctness.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from rag.models import RetrievedChunk


class GoldenQuestion(BaseModel):
    """One test case: a question and where its answer lives."""

    id: str
    question: str
    source: str = ""
    """Filename the answer lives in. Empty means any source counts."""

    expected_pages: list[int] = Field(default_factory=list)
    expected_answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    answerable: bool = True
    note: str = ""

    @model_validator(mode="after")
    def check_consistency(self) -> GoldenQuestion:
        if self.answerable and not self.expected_pages:
            raise ValueError(
                f"{self.id}: answerable questions need expected_pages — otherwise there is "
                "nothing to score retrieval against"
            )
        if not self.answerable and self.expected_pages:
            raise ValueError(f"{self.id}: unanswerable questions must not list expected_pages")
        return self

    def relevant_keys(self) -> list[str]:
        """The page keys a correct retrieval should surface."""
        return [page_key(self.source, page) for page in self.expected_pages]


def page_key(source: str, page: int | None) -> str:
    """Stable identifier for 'this page of this document'.

    The unit of relevance throughout evaluation. Chunk IDs change when chunking changes;
    this does not.

    The extension is stripped deliberately. A golden question names its source as a
    filename ("MySQL Handbook.pdf"), while a chunk carries the document *title*, which
    the loaders derive from the stem ("MySQL Handbook"). Normalising both sides to the
    stem is what lets the two meet — without it every question silently fails to match
    no matter how good retrieval is.
    """
    stem = Path(source).stem
    return f"{stem}#p{page}" if page is not None else f"{stem}#?"


def retrieved_keys(hits: list[RetrievedChunk]) -> list[str]:
    """Convert retrieval results into page keys, preserving rank and dropping repeats.

    Two chunks from the same page are one hit, not two — counting them separately would
    inflate precision for no reason, and smaller chunk sizes would score better purely
    by returning more pieces of the same page.

    Every page a chunk covers is emitted, not only the page it starts on. A chunk whose
    text runs from page 14 into page 15 did retrieve page 15, and crediting it with 14
    alone marked six golden questions as retrieval failures while their text sat in the
    index. See `Chunk.page_end`.

    One consequence for callers: a chunk can now contribute several keys, so this list
    may be longer than the number of hits. Truncating it to k would then silently
    discard the pages covered by the lowest-ranked chunks, which is why the runner
    scores the whole list and limits retrieval to k chunks instead.
    """
    keys: list[str] = []
    for hit in hits:
        covered: Sequence[int | None] = hit.chunk.pages or [None]
        for page in covered:
            key = page_key(hit.chunk.source, page)
            if key not in keys:
                keys.append(key)
    return keys


def retrieved_chunk_pages(hits: list[RetrievedChunk]) -> list[list[str]]:
    """The page keys each retrieved chunk covers, one inner list per chunk, in rank order.

    Kept separate from `retrieved_keys` because precision is a question about chunks,
    not pages: what fills the context window is passages, so "how many of the k
    passages were relevant" is the thing worth knowing. Flattening to pages first makes
    that unanswerable.
    """
    rows: list[list[str]] = []
    for hit in hits:
        covered: Sequence[int | None] = hit.chunk.pages or [None]
        rows.append([page_key(hit.chunk.source, page) for page in covered])
    return rows


class DatasetError(ValueError):
    """Raised when a golden dataset is malformed or internally inconsistent."""


def load_dataset(path: Path) -> list[GoldenQuestion]:
    """Read a JSONL golden set, reporting the line number of any bad row."""
    if not path.exists():
        raise DatasetError(f"{path} does not exist. See eval/datasets/golden.example.jsonl.")

    questions: list[GoldenQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            questions.append(GoldenQuestion(**json.loads(line)))
        except json.JSONDecodeError as error:
            raise DatasetError(f"{path}:{number} is not valid JSON — {error}") from error
        except ValueError as error:
            raise DatasetError(f"{path}:{number} — {error}") from error

    if not questions:
        raise DatasetError(f"{path} contains no questions.")

    duplicates = [qid for qid, count in Counter(q.id for q in questions).items() if count > 1]
    if duplicates:
        raise DatasetError(f"{path}: duplicate question ids {duplicates}")

    return questions


def filter_by_tags(questions: list[GoldenQuestion], tags: list[str]) -> list[GoldenQuestion]:
    """Keep questions carrying any of the given tags, for scoring one slice."""
    wanted = set(tags)
    return [q for q in questions if wanted & set(q.tags)]


def describe(questions: list[GoldenQuestion]) -> dict[str, int | float]:
    """Summarise a dataset's shape.

    Worth checking before trusting any score. A set with no unanswerable questions
    cannot measure refusal; one under about fifty questions moves on noise.
    """
    unanswerable = len([q for q in questions if not q.answerable])
    tags = Counter(tag for q in questions for tag in q.tags)
    return {
        "questions": len(questions),
        "answerable": len(questions) - unanswerable,
        "unanswerable": unanswerable,
        "sources": len({q.source for q in questions if q.source}),
        "distinct_tags": len(tags),
    }
