"""Retrieval metrics.

Retrieval and generation are scored separately because they fail separately. A wrong
answer means either the retriever fetched the wrong passages or the generator ignored
the right ones, and a single end-to-end number cannot distinguish the two.

Hit rate is the ceiling on everything downstream: if the correct passage was never
retrieved, no amount of prompting recovers the answer. Improving reranking should move
MRR and NDCG far more than it moves hit rate, because reranking reorders a candidate
set rather than widening it — that relationship is how you confirm a reranker is doing
what you think it is.

Every function here is pure: lists in, float out. No documents, no models, no network.

TODO(phase-3): faithfulness and answer relevance, which need an LLM judge and
calibration against hand labels before their numbers mean anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _truncate(retrieved: Sequence[str], k: int | None) -> list[str]:
    if k is None:
        return list(retrieved)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return list(retrieved[:k])


def hit_rate_at_k(
    retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = 5
) -> float:
    """Did any expected item appear in the top k? 1.0 or 0.0 for a single query.

    The bluntest and most important retrieval metric. Averaged over a dataset it reads
    as "what fraction of questions could possibly have been answered".
    """
    if not expected_ids:
        return 0.0
    return 1.0 if set(_truncate(retrieved_ids, k)) & set(expected_ids) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    """1/rank of the first correct item, or 0.0 if none was retrieved.

    Rewards putting the right passage first rather than merely somewhere in the list,
    which matters because generation only sees the top few.
    """
    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_per_query: Sequence[Sequence[str]],
    expected_per_query: Sequence[Sequence[str]],
) -> float:
    """Average reciprocal rank across a dataset."""
    if len(retrieved_per_query) != len(expected_per_query):
        raise ValueError(
            f"{len(retrieved_per_query)} result lists but {len(expected_per_query)} expectations"
        )
    if not retrieved_per_query:
        return 0.0
    return sum(
        reciprocal_rank(retrieved, expected)
        for retrieved, expected in zip(retrieved_per_query, expected_per_query, strict=True)
    ) / len(retrieved_per_query)


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain: relevance, discounted by how far down it sits."""
    return sum(relevance / math.log2(rank + 1) for rank, relevance in enumerate(relevances, 1))


def ndcg_at_k(
    retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = 10
) -> float:
    """NDCG@k — judges the whole ordering, not just the first hit.

    Normalised against the best possible ordering, so 1.0 means every expected item was
    retrieved and placed as high as it could be. Unlike hit rate it distinguishes
    "correct passage at position 1" from "correct passage at position 8", and unlike MRR
    it accounts for every expected item rather than only the first.
    """
    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    top = _truncate(retrieved_ids, k)

    actual = dcg([1.0 if item in expected else 0.0 for item in top])
    # The ideal ordering is bounded by how many positions are being judged. With k=None
    # that is the length of the list itself, not None — scoring the whole list has to
    # normalise against the whole list.
    positions = len(top) if k is None else k
    ideal = dcg([1.0] * min(len(expected), positions))
    return actual / ideal if ideal else 0.0


def precision_at_k(
    retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = 5
) -> float:
    """Of the pages retrieved, what fraction were relevant?

    Kept for comparison with older runs, but `chunk_precision_at_k` is the metric to
    read. Dividing by the number of DISTINCT retrieved pages makes this track chunk
    size rather than retrieval quality: smaller chunks land several pieces on the same
    page, the page list collapses, the divisor shrinks and the score rises without
    anything having been retrieved better. Measured on this project, precision@5 ran
    0.278 / 0.258 / 0.218 / 0.193 for chunk sizes 128 / 160 / 200 / 256 — monotonic in
    chunk size, which is the artifact and not a finding.

    `retrieved_keys` deduplicates pages for exactly this reason, with the comment that
    otherwise "smaller chunk sizes would score better purely by returning more pieces
    of the same page". The deduplication fixed the numerator; this divisor put the same
    bias straight back.
    """
    top = _truncate(retrieved_ids, k)
    if not top:
        return 0.0
    return len([item for item in top if item in set(expected_ids)]) / len(top)


def chunk_precision_at_k(
    chunk_pages: Sequence[Sequence[str]], expected_ids: Sequence[str], k: int | None = 5
) -> float:
    """Of the k retrieved chunks, what fraction covered an expected page?

    Precision is a question about passages, because passages are what fill the context
    window and distract the model. Measuring it over chunks makes the divisor the
    number of chunks actually sent — a fixed k — so the score no longer moves when
    chunk size changes the number of distinct pages those chunks happen to cover.

    Takes one page-key list per chunk, from `retrieved_chunk_pages`.
    """
    top = list(chunk_pages) if k is None else list(chunk_pages[:k])
    if not top:
        return 0.0
    expected = set(expected_ids)
    relevant = sum(1 for pages in top if expected & set(pages))
    return relevant / len(top)


def recall_at_k(
    retrieved_ids: Sequence[str], expected_ids: Sequence[str], k: int | None = 5
) -> float:
    """Of the passages that should have been found, what fraction were?"""
    if not expected_ids:
        return 0.0
    found = set(_truncate(retrieved_ids, k)) & set(expected_ids)
    return len(found) / len(set(expected_ids))


def correct_refusal_rate(results: Sequence[dict[str, Any]]) -> float:
    """Fraction of unanswerable questions the system correctly declined.

    Scored only over questions marked unanswerable. A system that never refuses is not
    grounded — it is guessing politely — and this is the metric that catches it.
    Returns 0.0 when the dataset contains no unanswerable questions, which is itself a
    signal the dataset needs work.
    """
    unanswerable = [row for row in results if not row.get("answerable", True)]
    if not unanswerable:
        return 0.0
    return len([row for row in unanswerable if not row.get("answered", True)]) / len(unanswerable)


def aggregate(
    retrieved_per_query: Sequence[Sequence[str]],
    expected_per_query: Sequence[Sequence[str]],
    k: int = 5,
    truncate: bool = True,
    chunk_pages_per_query: Sequence[Sequence[Sequence[str]]] | None = None,
) -> dict[str, float]:
    """Every retrieval metric over a whole dataset, for the results table.

    `truncate` controls whether the page lists are cut to k before scoring, and the
    caller has to choose deliberately now that one chunk can cover several pages.
    Retrieval already returns k chunks; cutting the resulting page list to k as well
    would throw away the pages covered by the lowest-ranked chunks and report them as
    misses. The runner therefore passes `truncate=False`, and `k` only names the
    metrics. It defaults to True so existing callers and runs are unaffected.

    `chunk_pages_per_query`, when given, adds `chunk_precision@k` — precision measured
    over chunks rather than pages, which is the one worth comparing between chunk
    sizes. See `chunk_precision_at_k`.
    """
    if len(retrieved_per_query) != len(expected_per_query):
        raise ValueError(
            f"{len(retrieved_per_query)} result lists but {len(expected_per_query)} expectations"
        )
    if not retrieved_per_query:
        return {}

    pairs = list(zip(retrieved_per_query, expected_per_query, strict=True))
    n = len(pairs)
    cap = k if truncate else None
    metrics = {
        f"hit_rate@{k}": sum(hit_rate_at_k(r, e, cap) for r, e in pairs) / n,
        "mrr": mean_reciprocal_rank(retrieved_per_query, expected_per_query),
        "ndcg@10": sum(ndcg_at_k(r, e, 10 if truncate else None) for r, e in pairs) / n,
        f"precision@{k}": sum(precision_at_k(r, e, cap) for r, e in pairs) / n,
        f"recall@{k}": sum(recall_at_k(r, e, cap) for r, e in pairs) / n,
    }

    if chunk_pages_per_query is not None:
        if len(chunk_pages_per_query) != n:
            raise ValueError(f"{len(chunk_pages_per_query)} chunk-page lists but {n} queries")
        metrics[f"chunk_precision@{k}"] = (
            sum(
                chunk_precision_at_k(chunks, expected, k)
                for chunks, expected in zip(chunk_pages_per_query, expected_per_query, strict=True)
            )
            / n
        )
    return metrics