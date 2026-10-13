"""Query rewriting: closing the vocabulary gap between question and document.

**The failure this targets**, from the project's own golden set. Four of eleven
retrieval misses are queries that describe an *intent* without ever using the term the
document is indexed under:

    "How do I count only the rows matching a condition?"   never says COUNT
    "How do I find rows where a column has no value?"      never says NULL
    "How do I make MySQL generate the id automatically?"   never says AUTO_INCREMENT
    "How do I select every column from a table?"           never says SELECT *

Neither retriever can bridge that. Dense embeddings match topic, so these land
somewhere in the right document but nowhere near the right page. BM25 matches literal
terms, and the literal term is absent from the query by construction. Reranking cannot
help either: the correct page never enters the candidate pool.

Rewriting attacks it at the only point where the gap can close — before retrieval, by
asking a language model what words the document would have used.

**Expansion, not replacement.** The rewritten terms are appended to the original query
rather than replacing it. A replacement bets everything on the rewrite being right; an
expansion keeps the original signal and adds vocabulary on top, so a poor rewrite
degrades the query rather than destroying it.

**Caching is not an optimisation here, it is what makes the experiment runnable.** A
single eval pass is 32 questions, and the Gemini free tier allows roughly ten requests
per minute. Without a cache, every run costs several minutes and risks rate-limiting
partway through, which would silently change the result. Cached, a repeat run is free
and identical — and identical matters, because a measurement that moves between runs
for reasons unrelated to the change being tested is not a measurement.

TODO(phase-4): an async path, so the API can rewrite too. `rewrite` is synchronous
because every current caller (CLI, eval runner) is.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag.models import RetrievedChunk

if TYPE_CHECKING:
    from rag.generation.llm import LLMClient

REWRITE_SYSTEM = """You expand search queries for a technical document retrieval system.

Given a question, output the exact technical terms, keywords, and syntax that the
source document most likely uses to describe the same thing. The question is often
phrased in plain language while the document uses precise terminology.

Rules:
- Output ONLY space-separated terms. No sentences, no explanation, no punctuation
  beyond what belongs to the terms themselves.
- Prefer the exact identifier, keyword or command over a description of it.
- At most 10 terms.
- If the question already uses the precise terminology, repeat those terms.

Example:
Question: How do I count only the rows matching a condition?
Terms: COUNT aggregate function WHERE clause filter rows

Example:
Question: How do I find rows where a column has no value at all?
Terms: NULL IS NULL missing values WHERE clause"""

MAX_EXPANSION_TERMS = 10
MAX_TERM_CHARS = 40

MIN_SECONDS_BETWEEN_CALLS = 7.0
"""Pacing between model calls, to stay under the free tier's ~10 requests per minute.

Retry with backoff is the right response to an occasional 503 and the wrong one for a
batch. Measured on this project: 33 queries fired as fast as possible, each retrying
four times, issued up to 132 requests inside two minutes against a ten-per-minute
ceiling — turning scattered 503s into a hard 429 and leaving 27 of 33 queries
unexpanded. Spacing the calls out is what actually fixes it; backoff then only has to
handle the genuine spikes.
"""

_SENTENCE_PUNCT = ".,;:!?"
_LABEL = re.compile(r"^\s*(terms?|keywords?|answer)\s*:", re.IGNORECASE)


class QueryTransformer:
    """Expands a query with terminology the source document is likely to use.

    Falls back to the original query on any failure — a rate limit, a timeout, an empty
    response. Retrieval degrading to its previous behaviour is always better than a
    search that fails, and `last_error` records that it happened so a run is never
    silently different from what it claims to be.
    """

    def __init__(
        self,
        llm: LLMClient,
        cache_path: Path | None = None,
        max_terms: int = MAX_EXPANSION_TERMS,
        min_interval: float = MIN_SECONDS_BETWEEN_CALLS,
    ):
        self.llm = llm
        self.max_terms = max_terms
        self.cache_path = cache_path
        self.min_interval = min_interval
        self._last_call_at = 0.0
        self.last_error: str = ""
        self.quota_exhausted = False
        """Set when the provider reports a quota that waiting a moment cannot clear.

        The difference between "this call failed" and "no further call can succeed"
        is what stops a batch from spending its remaining budget discovering the same
        answer once per query.
        """
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        """Rewrites that fell back to the original query.

        Counted, not just logged. A run where every rewrite failed reproduces the
        baseline exactly and looks like "the change had no effect" — which is a
        different and far more misleading conclusion than "the change never ran".
        """
        self._cache: dict[str, str] = self._load_cache()

    # --- cache --------------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        except OSError:
            # A cache that cannot be written is a lost optimisation, not a failed query.
            pass

    @staticmethod
    def cache_key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]

    # --- rewriting ----------------------------------------------------------

    def clean_terms(self, raw: str) -> str:
        """Normalise the model's output into a bounded list of search terms.

        Sentence punctuation is stripped from the edges of each token while internal
        punctuation survives, because the terms worth adding are often punctuation:
        `SELECT *`, `!=`, `<>`, `COUNT(*)`. A `Terms:` label is dropped if the model
        echoes one.

        This cannot tell a term from a word in a sentence, and does not try. The real
        protection against a chatty response is `max_terms`: a model that ignores the
        instruction and writes prose contributes at most ten tokens of it, which
        degrades the query rather than swamping it.
        """
        tokens = _LABEL.sub("", raw).split()
        kept: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            # A token made entirely of punctuation IS the term — `!=`, `<>`, `*` —
            # so stripping its edges would destroy it.
            term = token if not any(c.isalnum() for c in token) else token.strip(_SENTENCE_PUNCT)
            if not term or len(term) > MAX_TERM_CHARS:
                continue
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            kept.append(term)
            if len(kept) >= self.max_terms:
                break
        return " ".join(kept)

    def expansion_for(self, query: str) -> str:
        """The added terms for a query, from cache or the model. Empty on failure."""
        key = self.cache_key(query)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        from rag.generation.llm import GenerationConfig, LLMError, QuotaError

        self._wait_for_rate_limit()
        try:
            if _loop_is_running():
                raise LLMError("rewrite is synchronous and cannot run inside an event loop")
            completion = asyncio.run(
                self.llm.complete(
                    REWRITE_SYSTEM,
                    f"Question: {query}\nTerms:",
                    GenerationConfig(temperature=0.0, max_output_tokens=64),
                )
            )
        # A failed rewrite must not fail the search; fall back to the original query.
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.failures += 1
            if isinstance(error, QuotaError) and error.exhausted:
                self.quota_exhausted = True
            return ""

        self.calls += 1
        if completion.blocked or not completion.text.strip():
            self.last_error = "model returned nothing"
            self.failures += 1
            return ""

        terms = self.clean_terms(completion.text)
        self._cache[key] = terms
        self._save_cache()
        return terms

    def _wait_for_rate_limit(self) -> None:
        """Sleep so consecutive calls stay at least `min_interval` apart.

        Only cache misses reach here, so a fully warmed cache costs nothing.
        """
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_at
        if self._last_call_at and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_at = time.monotonic()

    def warm(self, queries: Sequence[str], on_progress: Any = None) -> dict[str, int]:
        """Pre-compute expansions for a batch, skipping anything already cached.

        Kept separate from evaluation on purpose. An eval run that makes live API calls
        is slow, rate-limited and non-deterministic — three properties a measurement
        must not have. Warm the cache once, then every run scores from it instantly and
        identically.

        Resumable: a run interrupted by rate limiting keeps whatever it cached, so
        running it again picks up where it stopped.

        Stops early once the provider reports an exhausted quota. The first version
        pressed on through all 34 queries after the budget was gone, and each attempt
        cost requests it could no longer afford — the batch spent the quota proving a
        fact it already knew from the first failure. `skipped` counts what it left.
        """
        summary = {"cached": 0, "fetched": 0, "failed": 0, "skipped": 0}
        for index, query in enumerate(queries, start=1):
            if self.cache_key(query) in self._cache:
                summary["cached"] += 1
                status = "cached"
            elif self.quota_exhausted:
                summary["skipped"] += 1
                status = "skipped"
            else:
                before = self.failures
                terms = self.expansion_for(query)
                if terms and self.failures == before:
                    summary["fetched"] += 1
                    status = "fetched"
                else:
                    summary["failed"] += 1
                    status = "failed"
            if on_progress is not None:
                on_progress(index, len(queries), query, status)
        return summary

    def rewrite(self, query: str) -> str:
        """Return the query with document terminology appended.

        Returns the query unchanged when rewriting fails or adds nothing, so a caller
        can always use the result without checking.
        """
        self.last_error = ""
        if not query.strip():
            return query

        terms = self.expansion_for(query)
        if not terms:
            return query

        # Terms already present contribute nothing and would just repeat tokens, which
        # BM25 weights by frequency. Query tokens are normalised the same way the
        # model's output is, or "rows?" would fail to match the term "rows".
        present = {token.strip(_SENTENCE_PUNCT).lower() for token in query.split()} - {""}
        added = [term for term in terms.split() if term.lower() not in present]
        return f"{query} {' '.join(added)}" if added else query


class ScriptedTransformer:
    """Test double returning fixed expansions. No model, no network.

    Lets the rewriting *pipeline* — caching, fallback, term cleaning, wiring into
    retrieval — be tested without an API key.
    """

    def __init__(self, expansions: dict[str, str] | None = None, fail: bool = False):
        self.expansions = expansions or {}
        self.fail = fail
        self.calls: list[str] = []
        self.last_error = ""

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        if self.fail:
            self.last_error = "RuntimeError: transformer unavailable"
            return query
        terms = self.expansions.get(query, "")
        return f"{query} {terms}" if terms else query


class RewritingRetriever:
    """Wraps a retriever, expanding the query before it is searched.

    Deliberately a wrapper rather than a change inside each retriever: the rewritten
    query must reach dense and sparse retrieval alike, and keeping it outside means the
    A/B comparison is a single flag rather than a code path per retriever.

    When reranking is also enabled this sits *inside* it, so the cross-encoder scores
    passages against the question the user actually asked rather than the expansion.
    """

    def __init__(
        self,
        retriever: Any,
        transformer: Any,
    ):
        self.retriever = retriever
        self.transformer = transformer
        self.last_query: str = ""

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.last_query = str(self.transformer.rewrite(query))
        hits: list[RetrievedChunk] = self.retriever.search(
            self.last_query, top_k=top_k, filters=filters
        )
        return hits


def _loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def find_transformer(retriever: Any) -> QueryTransformer | None:
    """Locate the transformer inside a wrapped retriever chain.

    The chain is built by nesting — reranking wraps rewriting wraps hybrid — so a
    caller that wants to report on rewriting has to walk down to it.
    """
    seen = 0
    while retriever is not None and seen < 10:
        transformer = getattr(retriever, "transformer", None)
        if isinstance(transformer, QueryTransformer):
            return transformer
        retriever = getattr(retriever, "retriever", None)
        seen += 1
    return None