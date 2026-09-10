"""Golden dataset loading and validation.

See docs/EVALUATION.md for the rules that keep the dataset honest. The most important
one: write the questions from the source documents before running the system. A
question written after reading model output measures self-agreement, not correctness.

TODO(phase-3): load(), validate(), stratified sampling by tag
"""

from pathlib import Path

from pydantic import BaseModel


class GoldenQuestion(BaseModel):
    id: str
    question: str
    expected_chunk_ids: list[str]
    expected_answer: str | None = None
    tags: list[str] = []
    answerable: bool = True


def load_dataset(path: Path) -> list[GoldenQuestion]:
    raise NotImplementedError("TODO(phase-3)")
