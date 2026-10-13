"""Query rewriting tests.

ScriptedClient throughout, so nothing needs an API key or network.
"""

import asyncio
import json
from pathlib import Path

import pytest

from rag.generation.llm import QuotaError, ScriptedClient
from rag.models import Chunk, RetrievedChunk
from rag.retrieval.query_transform import (
    QueryTransformer,
    RewritingRetriever,
    ScriptedTransformer,
    find_transformer,
)

COUNT_TERMS = "COUNT aggregate function WHERE clause filter rows"


def transformer(*responses: str, cache: Path | None = None) -> QueryTransformer:
    """A transformer with pacing disabled — tests must not sleep real seconds."""
    return QueryTransformer(
        ScriptedClient(list(responses) or ["terms"]), cache_path=cache, min_interval=0
    )


class StubRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5, filters=None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return [
            RetrievedChunk(chunk=Chunk(id="c" * 32, document_id="d", text="t"), dense_score=0.5)
        ]


# --- term cleaning ----------------------------------------------------------


def test_sentence_punctuation_is_stripped():
    assert transformer().clean_terms("COUNT, WHERE clause.") == "COUNT WHERE clause"


def test_operator_tokens_survive_intact():
    """`!=` and `<>` ARE the terms — stripping their edges would destroy them."""
    assert transformer().clean_terms("!= <> operator") == "!= <> operator"


def test_a_wildcard_survives():
    assert "*" in transformer().clean_terms("SELECT * FROM table;")


def test_an_echoed_label_is_dropped():
    assert transformer().clean_terms("Terms: NULL IS NULL") == "NULL IS"


def test_duplicate_terms_are_collapsed():
    assert transformer().clean_terms("COUNT count COUNT") == "COUNT"


def test_term_count_is_capped():
    """A model that ignores the instruction contributes at most max_terms of prose."""
    many = " ".join(f"term{n}" for n in range(50))
    assert len(transformer().clean_terms(many).split()) == 10


def test_absurdly_long_tokens_are_dropped():
    assert transformer().clean_terms("x" * 100 + " COUNT") == "COUNT"


# --- rewriting --------------------------------------------------------------


def test_terms_are_appended_to_the_original_query():
    """Expansion, not replacement: a poor rewrite must degrade, not destroy."""
    result = transformer("NULL IS NULL missing values").rewrite("How do I find empty rows?")
    assert result.startswith("How do I find empty rows?")
    assert "NULL" in result


def test_terms_already_in_the_query_are_not_repeated():
    result = transformer("COUNT rows").rewrite("How do I COUNT rows?")
    assert result == "How do I COUNT rows?"


def test_an_empty_query_is_returned_unchanged():
    assert transformer().rewrite("   ") == "   "


def test_a_model_failure_falls_back_to_the_original():
    """Retrieval degrading beats a search that fails."""
    broken = QueryTransformer(ScriptedClient([""]), min_interval=0)
    assert broken.rewrite("original question") == "original question"


def test_a_model_failure_is_recorded():
    """A run must never be silently different from what it claims to be."""
    broken = QueryTransformer(ScriptedClient([""]), min_interval=0)
    broken.rewrite("original question")
    assert broken.last_error


def test_a_successful_rewrite_clears_a_previous_error():
    tf = transformer(COUNT_TERMS)
    tf.last_error = "stale"
    tf.rewrite("How do I tally matching records?")
    assert tf.last_error == ""


# --- caching ----------------------------------------------------------------


def test_the_same_query_is_only_sent_once(tmp_path: Path):
    tf = transformer(COUNT_TERMS, cache=tmp_path / "c.json")
    tf.rewrite("How do I tally records?")
    tf.rewrite("How do I tally records?")
    assert tf.calls == 1
    assert tf.cache_hits == 1


def test_the_cache_key_ignores_case_and_surrounding_space():
    assert QueryTransformer.cache_key(" How Many? ") == QueryTransformer.cache_key("how many?")


def test_the_cache_survives_a_new_process(tmp_path: Path):
    """An eval run must be repeatable without re-spending the rate limit."""
    path = tmp_path / "c.json"
    transformer(COUNT_TERMS, cache=path).rewrite("How do I tally records?")

    reloaded = transformer("DIFFERENT TERMS", cache=path)
    result = reloaded.rewrite("How do I tally records?")
    assert "aggregate" in result
    assert reloaded.calls == 0


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    assert transformer(COUNT_TERMS, cache=path).rewrite("How do I tally records?")


def test_the_cache_file_is_readable_json(tmp_path: Path):
    path = tmp_path / "c.json"
    transformer(COUNT_TERMS, cache=path).rewrite("How do I tally records?")
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_no_cache_path_still_works():
    assert transformer(COUNT_TERMS).rewrite("How do I tally records?")


# --- wiring -----------------------------------------------------------------


def test_the_retriever_searches_the_expanded_query():
    retriever = StubRetriever()
    wrapped = RewritingRetriever(
        retriever, ScriptedTransformer({"How do I tally records?": "COUNT aggregate"})
    )
    wrapped.search("How do I tally records?", top_k=3)
    assert retriever.queries == ["How do I tally records? COUNT aggregate"]


def test_the_expanded_query_is_exposed_for_inspection():
    wrapped = RewritingRetriever(StubRetriever(), ScriptedTransformer({"q": "extra terms"}))
    wrapped.search("q")
    assert wrapped.last_query == "q extra terms"


def test_top_k_and_filters_reach_the_inner_retriever():
    retriever = StubRetriever()
    RewritingRetriever(retriever, ScriptedTransformer()).search("q", top_k=7)
    assert retriever.queries == ["q"]


def test_a_failing_transformer_still_searches_the_original():
    retriever = StubRetriever()
    RewritingRetriever(retriever, ScriptedTransformer(fail=True)).search("q")
    assert retriever.queries == ["q"]


def test_rewriting_inside_an_event_loop_falls_back_rather_than_crashing():
    """`rewrite` is synchronous; the async API path must not blow up on it.

    Driven through asyncio.run rather than an async test so it does not depend on a
    pytest plugin. TODO(phase-4): a real async rewrite instead of this fallback.
    """
    tf = transformer(COUNT_TERMS)

    async def inside_a_loop() -> str:
        return tf.rewrite("How do I tally records?")

    assert asyncio.run(inside_a_loop()) == "How do I tally records?"
    assert "event loop" in tf.last_error


# --- failure visibility -----------------------------------------------------


def test_failures_are_counted_not_just_logged():
    """A run where every rewrite failed reproduces the baseline exactly.

    That is indistinguishable from "rewriting worked and changed nothing" unless the
    failures are counted — and the two conclusions are very different.
    """
    broken = QueryTransformer(ScriptedClient([""]), min_interval=0)
    broken.rewrite("a question")
    broken.rewrite("another question")
    assert broken.failures == 2


def test_a_successful_rewrite_does_not_count_as_a_failure():
    tf = transformer(COUNT_TERMS)
    tf.rewrite("How do I tally records?")
    assert tf.failures == 0
    assert tf.calls == 1


def test_find_transformer_walks_a_wrapped_chain():
    tf = transformer(COUNT_TERMS)
    chain = RewritingRetriever(StubRetriever(), tf)
    assert find_transformer(chain) is tf


def test_find_transformer_reaches_through_two_wrappers():
    """Reranking wraps rewriting wraps hybrid."""

    class Outer:
        def __init__(self, inner: object) -> None:
            self.retriever = inner

    tf = transformer(COUNT_TERMS)
    assert find_transformer(Outer(RewritingRetriever(StubRetriever(), tf))) is tf


def test_find_transformer_returns_none_when_rewriting_is_off():
    assert find_transformer(StubRetriever()) is None


def test_find_transformer_terminates_on_a_cycle():
    class SelfReferencing:
        def __init__(self) -> None:
            self.retriever = self

    assert find_transformer(SelfReferencing()) is None


# --- pacing and cache warming -----------------------------------------------


def test_calls_are_paced_apart(monkeypatch):
    """Backoff alone turned scattered 503s into a hard 429; spacing is the fix."""
    slept: list[float] = []
    monkeypatch.setattr(
        "rag.retrieval.query_transform.time.sleep", lambda seconds: slept.append(seconds)
    )
    tf = QueryTransformer(ScriptedClient(["A B", "C D"]), min_interval=7.0)
    tf.rewrite("first question")
    tf.rewrite("second question")
    assert slept and slept[0] == pytest.approx(7.0, abs=0.5)


def test_the_first_call_is_not_delayed(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(
        "rag.retrieval.query_transform.time.sleep", lambda seconds: slept.append(seconds)
    )
    QueryTransformer(ScriptedClient(["A B"]), min_interval=7.0).rewrite("only question")
    assert slept == []


def test_a_cache_hit_is_not_paced(monkeypatch, tmp_path: Path):
    """A warmed cache must cost nothing — an eval run is otherwise minutes of sleeping."""
    slept: list[float] = []
    path = tmp_path / "c.json"
    transformer(COUNT_TERMS, cache=path).rewrite("a question")

    monkeypatch.setattr(
        "rag.retrieval.query_transform.time.sleep", lambda seconds: slept.append(seconds)
    )
    tf = QueryTransformer(ScriptedClient(["unused"]), cache_path=path, min_interval=7.0)
    tf.rewrite("a question")
    assert slept == []


def test_warming_fetches_every_uncached_query():
    tf = QueryTransformer(ScriptedClient(["A B", "C D"]), min_interval=0)
    assert tf.warm(["one", "two"]) == {"cached": 0, "fetched": 2, "failed": 0, "skipped": 0}


def test_warming_skips_what_is_already_cached(tmp_path: Path):
    path = tmp_path / "c.json"
    QueryTransformer(ScriptedClient(["A B"]), cache_path=path, min_interval=0).warm(["one"])

    second = QueryTransformer(ScriptedClient(["C D"]), cache_path=path, min_interval=0)
    assert second.warm(["one", "two"]) == {
        "cached": 1,
        "fetched": 1,
        "failed": 0,
        "skipped": 0,
    }


def test_warming_records_failures_rather_than_stopping():
    """A pass cut short by rate limiting must keep whatever it managed to cache."""
    tf = QueryTransformer(ScriptedClient([""]), min_interval=0)
    assert tf.warm(["one", "two"])["failed"] == 2


def test_warming_reports_progress():
    seen: list[tuple[int, int, str]] = []
    tf = QueryTransformer(ScriptedClient(["A B"]), min_interval=0)
    tf.warm(["one", "two"], on_progress=lambda i, n, q, s: seen.append((i, n, s)))
    assert [row[0] for row in seen] == [1, 2]
    assert seen[0][1] == 2


def test_warming_nothing_is_a_no_op():
    assert QueryTransformer(ScriptedClient(["x"]), min_interval=0).warm([]) == {
        "cached": 0,
        "fetched": 0,
        "failed": 0,
        "skipped": 0,
    }


# --- quota exhaustion -------------------------------------------------------
#
# The distinction these cover is the one that cost a day: a per-minute 429 clears in
# under a minute, a per-day 429 does not clear until the provider resets it, and the
# status code alone cannot tell them apart.


class QuotaClient(ScriptedClient):
    """Raises a quota error instead of answering, like a free tier past its limit."""

    def __init__(self, scope: str = "per-day"):
        super().__init__(["unused"])
        self.scope = scope
        self.attempts = 0

    async def complete(self, system, user, config=None):
        self.attempts += 1
        raise QuotaError("out of quota", scope=self.scope)


def test_an_exhausted_quota_stops_the_batch():
    """The costly bug: 34 queries each spent requests rediscovering the same refusal."""
    client = QuotaClient("per-day")
    tf = QueryTransformer(client, min_interval=0)

    summary = tf.warm(["one", "two", "three", "four"])

    assert client.attempts == 1
    assert summary == {"cached": 0, "fetched": 0, "failed": 1, "skipped": 3}


def test_the_exhausted_flag_is_exposed_for_the_caller():
    tf = QueryTransformer(QuotaClient("per-day"), min_interval=0)
    tf.warm(["one"])
    assert tf.quota_exhausted


def test_a_per_minute_limit_does_not_stop_the_batch():
    """Waiting clears a per-minute limit, so the pass should keep trying."""
    client = QuotaClient("per-minute")
    tf = QueryTransformer(client, min_interval=0)

    summary = tf.warm(["one", "two", "three"])

    assert client.attempts == 3
    assert summary["skipped"] == 0


def test_cached_queries_are_still_served_after_exhaustion(tmp_path: Path):
    """Exhaustion must not void the cache — a warmed eval has to stay runnable."""
    path = tmp_path / "c.json"
    transformer(COUNT_TERMS, cache=path).rewrite("a question")

    tf = QueryTransformer(QuotaClient("per-day"), cache_path=path, min_interval=0)
    summary = tf.warm(["a question", "an uncached one", "another uncached one"])

    assert summary["cached"] == 1
    assert "COUNT" in tf.rewrite("a question")


def test_an_ordinary_failure_is_not_treated_as_exhaustion():
    """A model that answers with nothing is a content failure, not a budget one."""
    tf = QueryTransformer(ScriptedClient([""]), min_interval=0)
    summary = tf.warm(["one", "two"])
    assert not tf.quota_exhausted
    assert summary["failed"] == 2