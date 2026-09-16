"""Eval runner tests.

HashEmbedder and in-memory Qdrant, so no model download and no Docker.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.config import Settings
from rag.evaluation.datasets import GoldenQuestion
from rag.evaluation.runner import (
    EvalRun,
    EvalRunner,
    QuestionResult,
    compare,
    config_snapshot,
    format_report,
    load_run,
    regressions,
    save_run,
)
from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.embedding import HashEmbedder
from rag.ingestion.loaders import LoadedDocument
from rag.models import DocumentMeta
from rag.retrieval.store import VectorStore

DIM = 32
PAGES = [
    "Create a database with the CREATE DATABASE statement in MySQL.",
    "Create a user account and grant it the necessary privileges here.",
    "Indexes speed up lookups; use CREATE INDEX on frequently queried columns.",
]


@pytest.fixture
def runner():
    embedder = HashEmbedder(dimension=DIM)
    store = VectorStore.connect(":memory:", "eval_test", DIM, fingerprint="test:512:64")
    store.ensure_collection()

    text = "\n\n".join(PAGES)
    sections = []
    cursor = 0
    for number, page in enumerate(PAGES, start=1):
        sections.append({"heading": f"Page {number}", "level": 1, "page": number, "start": cursor})
        cursor += len(page) + 2

    meta = DocumentMeta(
        source_path="data/Handbook.pdf",
        title="Handbook",
        doc_type="pdf",
        content_hash="abc123",
        ingested_at=datetime.now(UTC),
        page_count=len(PAGES),
    )
    chunks = FixedSizeChunker(chunk_size=20, overlap=3).split(LoadedDocument(text, meta, sections))
    store.upsert(chunks, embedder.embed_chunks(chunks))

    return EvalRunner(embedder, store, Settings(embedding_dim=DIM))


def question(qid="q-1", pages=None, answerable=True, **kwargs) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        question=kwargs.pop("text", "how do I create a database"),
        source="Handbook.pdf",
        expected_pages=pages if pages is not None else ([1] if answerable else []),
        answerable=answerable,
        **kwargs,
    )


def make_run(**metrics) -> EvalRun:
    return EvalRun(
        started_at="2026-01-01T00:00:00+00:00",
        config={"chunk_size": 512},
        metrics=metrics or {"hit_rate@5": 0.5, "mrr": 0.4},
        results=[],
    )


# --- scoring ---------------------------------------------------------------


def test_scoring_a_question_records_both_sides(runner):
    result = runner.score_question(question(), top_k=5)
    assert result.expected == ["Handbook#p1"]
    assert result.retrieved


def test_a_question_whose_page_is_retrieved_counts_as_a_hit(runner):
    """Retrieve everything, so the expected page must be in there somewhere."""
    result = runner.score_question(question(), top_k=20)
    assert result.hit == 1.0
    assert not result.failed


def test_a_question_pointing_at_a_missing_page_fails(runner):
    result = runner.score_question(question(pages=[99]), top_k=5)
    assert result.hit == 0.0
    assert result.failed


def test_reciprocal_rank_is_recorded(runner):
    result = runner.score_question(question(), top_k=20)
    assert 0.0 < result.rr <= 1.0


def test_tags_and_notes_survive_into_the_result(runner):
    result = runner.score_question(question(tags=["mysql"], note="known miss"), top_k=5)
    assert result.tags == ["mysql"]
    assert result.note == "known miss"


# --- runs ------------------------------------------------------------------


def test_run_scores_every_answerable_question(runner):
    run = runner.run([question("q-1"), question("q-2")], top_k=20)
    assert len(run.results) == 2


def test_run_skips_unanswerable_questions(runner):
    """They have no expected pages, so retrieval cannot be right or wrong about them."""
    run = runner.run([question("q-1"), question("q-2", answerable=False)], top_k=5)
    assert len(run.results) == 1


def test_run_reports_every_metric(runner):
    run = runner.run([question()], top_k=5)
    assert "mrr" in run.metrics
    assert "ndcg@10" in run.metrics


def test_run_collects_failures(runner):
    run = runner.run([question("q-1", pages=[99])], top_k=5)
    assert len(run.failures) == 1


def test_run_records_the_config_that_produced_it(runner):
    """A metric without its settings cannot be reproduced or compared."""
    run = runner.run([question()], top_k=7)
    assert run.config["top_k"] == 7
    assert run.config["chunk_size"] == runner.settings.chunk_size


def test_config_snapshot_includes_what_changes_a_score():
    snapshot = config_snapshot(Settings(), top_k=5)
    assert {"embedding_model", "chunk_size", "chunk_overlap", "top_k"} <= set(snapshot)


# --- comparison and gating -------------------------------------------------


def test_compare_reports_deltas():
    delta = compare(make_run(**{"hit_rate@5": 0.5}), make_run(**{"hit_rate@5": 0.8}))
    assert delta["hit_rate@5"]["delta"] == pytest.approx(0.3)


def test_no_regression_when_metrics_improve():
    assert regressions(make_run(**{"hit_rate@5": 0.5}), make_run(**{"hit_rate@5": 0.9})) == []


def test_a_large_drop_is_a_regression():
    dropped = regressions(make_run(**{"hit_rate@5": 0.9}), make_run(**{"hit_rate@5": 0.5}))
    assert [name for name, _ in dropped] == ["hit_rate@5"]


def test_a_drop_within_tolerance_is_not_a_regression():
    assert (
        regressions(
            make_run(**{"hit_rate@5": 0.90}), make_run(**{"hit_rate@5": 0.89}), max_drop=0.02
        )
        == []
    )


def test_a_gain_in_one_metric_does_not_hide_a_drop_in_another():
    """The whole point of gating on every metric rather than an average."""
    before = make_run(**{"hit_rate@5": 0.9, "mrr": 0.9})
    after = make_run(**{"hit_rate@5": 0.99, "mrr": 0.4})
    assert [name for name, _ in regressions(before, after)] == ["mrr"]


# --- persistence -----------------------------------------------------------


def test_a_run_survives_a_round_trip(tmp_path: Path, runner):
    original = runner.run([question()], top_k=5)
    restored = load_run(save_run(original, tmp_path))
    assert restored.metrics == original.metrics
    assert restored.config == original.config
    assert len(restored.results) == len(original.results)


def test_saved_runs_do_not_overwrite_each_other(tmp_path: Path):
    a = make_run()
    b = EvalRun(
        started_at="2026-01-02T00:00:00+00:00", config={}, metrics={}, results=[], git_sha="abc1234"
    )
    assert save_run(a, tmp_path) != save_run(b, tmp_path)


def test_report_renders_a_markdown_table(runner):
    report = format_report(runner.run([question()], top_k=5))
    assert "| Metric | Value |" in report
    assert "Config:" in report


def test_report_lists_failures(runner):
    report = format_report(runner.run([question("q-9", pages=[99], note="known miss")], top_k=5))
    assert "Failures" in report
    assert "known miss" in report


def test_result_knows_whether_it_failed():
    assert QuestionResult("q", "?", ["a"], ["b"], hit=0.0, rr=0.0).failed
    assert not QuestionResult("q", "?", ["a"], ["a"], hit=1.0, rr=1.0).failed
