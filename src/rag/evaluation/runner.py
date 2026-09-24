"""Eval harness: run a golden set through retrieval and score the results.

Retrieval only, deliberately. Retrieval is the ceiling on everything downstream — if
the right page was never fetched, no prompt recovers the answer — and scoring it needs
no LLM, so a full run is free, fast and repeatable. Generation metrics need a judge and
judge calibration, which is separate work with its own failure modes.

Every run records the config that produced it, so a number in the final report can be
traced back to the exact settings behind it. Without that, a results table is a list of
numbers with no way to reproduce any of them.

TODO(phase-3): faithfulness and answer relevance via a calibrated LLM judge
TODO(phase-4): persist runs to Postgres, CI gate on regression
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config import Settings
from rag.evaluation.datasets import GoldenQuestion, retrieved_keys
from rag.evaluation.metrics import aggregate, hit_rate_at_k, reciprocal_rank
from rag.ingestion.embedding import Embedder
from rag.retrieval.store import VectorStore


@dataclass
class QuestionResult:
    """What one question did, kept in enough detail to debug a failure.

    `retrieved` and `expected` are stored verbatim so a bad score can be inspected
    directly rather than reproduced by hand.
    """

    id: str
    question: str
    expected: list[str]
    retrieved: list[str]
    hit: float
    rr: float
    tags: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def failed(self) -> bool:
        return self.hit == 0.0


@dataclass
class EvalRun:
    started_at: str
    config: dict[str, str | int]
    metrics: dict[str, float]
    results: list[QuestionResult]
    dataset: str = ""
    git_sha: str = ""

    @property
    def failures(self) -> list[QuestionResult]:
        return [r for r in self.results if r.failed]

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2)


def config_snapshot(settings: Settings, top_k: int) -> dict[str, str | int]:
    """The settings that materially change a score.

    Anything that affects retrieval belongs here. Anything that does not is noise in a
    diff between two runs.
    """
    return {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": top_k,
        "index_fingerprint": settings.index_fingerprint(),
    }


def git_sha() -> str:
    """Short commit hash, or empty outside a git checkout."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


class EvalRunner:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: Settings,
        retriever: Any = None,
    ):
        """`retriever`, when given, replaces plain dense search.

        Passing a HybridRetriever here is what makes a dense-versus-hybrid comparison
        possible on the same golden set: only the retrieval step changes, so any
        difference in the metrics is attributable to it and nothing else.
        """
        self.retriever = retriever
        self.embedder = embedder
        self.store = store
        self.settings = settings

    def score_question(self, question: GoldenQuestion, top_k: int) -> QuestionResult:
        hits = (
            self.retriever.search(question.question, top_k=top_k)
            if self.retriever is not None
            else self.store.search(self.embedder.embed_query(question.question), top_k=top_k)
        )
        retrieved = retrieved_keys(hits)
        expected = question.relevant_keys()

        return QuestionResult(
            id=question.id,
            question=question.question,
            expected=expected,
            retrieved=retrieved,
            hit=hit_rate_at_k(retrieved, expected, k=top_k) if expected else 0.0,
            rr=reciprocal_rank(retrieved, expected) if expected else 0.0,
            tags=question.tags,
            note=question.note,
        )

    def run(
        self,
        questions: list[GoldenQuestion],
        top_k: int = 5,
        dataset: str = "",
    ) -> EvalRun:
        """Score every answerable question.

        Unanswerable questions are skipped here: they have no expected pages, so there
        is nothing for retrieval to be right or wrong about. They are scored in the
        generation pass, where refusal is the thing being measured.
        """
        answerable = [q for q in questions if q.answerable]
        results = [self.score_question(q, top_k) for q in answerable]

        metrics = aggregate(
            [r.retrieved for r in results],
            [r.expected for r in results],
            k=top_k,
        )

        return EvalRun(
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            config=config_snapshot(self.settings, top_k),
            metrics=metrics,
            results=results,
            dataset=dataset,
            git_sha=git_sha(),
        )


def compare(baseline: EvalRun, current: EvalRun) -> dict[str, dict[str, float]]:
    """Metric-by-metric delta between two runs, for the before/after table."""
    return {
        name: {
            "baseline": baseline.metrics.get(name, 0.0),
            "current": value,
            "delta": value - baseline.metrics.get(name, 0.0),
        }
        for name, value in current.metrics.items()
    }


def regressions(
    baseline: EvalRun, current: EvalRun, max_drop: float = 0.02
) -> list[tuple[str, float]]:
    """Metrics that fell further than max_drop below the baseline.

    The point of a gate: a change that improves one metric while quietly destroying
    another must not pass unnoticed.
    """
    return [
        (name, delta["delta"])
        for name, delta in compare(baseline, current).items()
        if delta["delta"] < -max_drop
    ]


def save_run(run: EvalRun, directory: Path) -> Path:
    """Write a run to disk, named by timestamp and commit."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = run.started_at.replace(":", "").replace("-", "")
    path = directory / f"{stamp}_{run.git_sha or 'nogit'}.json"
    path.write_text(run.to_json(), encoding="utf-8")
    return path


def load_run(path: Path) -> EvalRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"] = [QuestionResult(**row) for row in payload["results"]]
    return EvalRun(**payload)


def format_report(run: EvalRun) -> str:
    """Render a run as markdown, ready to paste into docs/EVALUATION.md.

    The config line is not decoration: a metric without the settings that produced it
    cannot be reproduced or compared, which makes it worthless as evidence.
    """
    lines = [
        f"**Run** {run.started_at}"
        + (f" · `{run.git_sha}`" if run.git_sha else "")
        + (f" · {run.dataset}" if run.dataset else ""),
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:.3f} |" for name, value in run.metrics.items()]

    config = ", ".join(f"{key}={value}" for key, value in run.config.items())
    lines += ["", f"Config: {config}"]

    if run.failures:
        lines += ["", f"**Failures ({len(run.failures)})**", ""]
        for result in run.failures:
            lines.append(f"- `{result.id}` {result.question}")
            lines.append(f"  - expected {result.expected}, got {result.retrieved[:3]}")
            if result.note:
                lines.append(f"  - {result.note}")

    return "\n".join(lines)
