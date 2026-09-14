"""Retrieval metric tests.

Pure functions, so these are exact-value assertions rather than smoke tests. If a
metric silently changes behaviour, every recorded score in the project becomes
incomparable with every other — worth pinning precisely.
"""

import math

import pytest

from rag.evaluation.metrics import (
    aggregate,
    correct_refusal_rate,
    dcg,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]


# --- hit rate --------------------------------------------------------------


def test_hit_rate_is_one_when_an_expected_item_is_in_the_top_k():
    assert hit_rate_at_k(RETRIEVED, ["c"], k=5) == 1.0


def test_hit_rate_is_zero_when_the_hit_falls_outside_k():
    assert hit_rate_at_k(RETRIEVED, ["e"], k=3) == 0.0


def test_hit_rate_ignores_position_within_k():
    assert hit_rate_at_k(RETRIEVED, ["a"], k=5) == hit_rate_at_k(RETRIEVED, ["e"], k=5)


def test_hit_rate_is_zero_with_no_expectations():
    assert hit_rate_at_k(RETRIEVED, [], k=5) == 0.0


def test_hit_rate_rejects_a_nonsense_k():
    with pytest.raises(ValueError, match="at least 1"):
        hit_rate_at_k(RETRIEVED, ["a"], k=0)


# --- reciprocal rank -------------------------------------------------------


def test_reciprocal_rank_rewards_a_first_place_hit():
    assert reciprocal_rank(RETRIEVED, ["a"]) == 1.0


def test_reciprocal_rank_discounts_by_position():
    assert reciprocal_rank(RETRIEVED, ["c"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_the_first_hit_only():
    assert reciprocal_rank(RETRIEVED, ["d", "b"]) == pytest.approx(1 / 2)


def test_reciprocal_rank_is_zero_on_a_miss():
    assert reciprocal_rank(RETRIEVED, ["z"]) == 0.0


def test_mrr_averages_across_queries():
    result = mean_reciprocal_rank([["a", "b"], ["x", "c"]], [["a"], ["c"]])
    assert result == pytest.approx((1.0 + 0.5) / 2)


def test_mrr_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="expectations"):
        mean_reciprocal_rank([["a"]], [["a"], ["b"]])


def test_mrr_of_nothing_is_zero():
    assert mean_reciprocal_rank([], []) == 0.0


# --- ndcg ------------------------------------------------------------------


def test_dcg_discounts_later_positions():
    assert dcg([1.0, 1.0]) == pytest.approx(1.0 + 1.0 / math.log2(3))


def test_ndcg_is_one_for_a_perfect_ordering():
    assert ndcg_at_k(RETRIEVED, ["a", "b"], k=5) == pytest.approx(1.0)


def test_ndcg_falls_when_hits_sit_lower():
    good = ndcg_at_k(RETRIEVED, ["a"], k=5)
    bad = ndcg_at_k(RETRIEVED, ["e"], k=5)
    assert good > bad > 0


def test_ndcg_is_zero_on_a_complete_miss():
    assert ndcg_at_k(RETRIEVED, ["z"], k=5) == 0.0


def test_ndcg_distinguishes_orderings_that_hit_rate_cannot():
    """The reason NDCG exists alongside hit rate."""
    assert hit_rate_at_k(RETRIEVED, ["a"], 5) == hit_rate_at_k(RETRIEVED, ["e"], 5)
    assert ndcg_at_k(RETRIEVED, ["a"], 5) != ndcg_at_k(RETRIEVED, ["e"], 5)


# --- precision and recall --------------------------------------------------


def test_precision_counts_relevant_items_among_those_returned():
    assert precision_at_k(RETRIEVED, ["a", "b"], k=4) == pytest.approx(0.5)


def test_recall_counts_expected_items_that_were_found():
    assert recall_at_k(RETRIEVED, ["a", "z"], k=5) == pytest.approx(0.5)


def test_recall_is_one_when_everything_expected_was_found():
    assert recall_at_k(RETRIEVED, ["a", "b"], k=5) == 1.0


def test_precision_of_an_empty_result_is_zero():
    assert precision_at_k([], ["a"], k=5) == 0.0


# --- refusal ---------------------------------------------------------------


def test_refusal_rate_scores_only_unanswerable_questions():
    results = [
        {"answerable": True, "answered": True},
        {"answerable": False, "answered": False},
        {"answerable": False, "answered": True},
    ]
    assert correct_refusal_rate(results) == pytest.approx(0.5)


def test_refusal_rate_is_one_when_every_refusal_is_correct():
    assert correct_refusal_rate([{"answerable": False, "answered": False}]) == 1.0


def test_refusal_rate_is_zero_without_unanswerable_questions():
    """A dataset with no unanswerable questions cannot measure refusal at all."""
    assert correct_refusal_rate([{"answerable": True, "answered": True}]) == 0.0


# --- aggregate -------------------------------------------------------------


def test_aggregate_reports_every_metric():
    summary = aggregate([RETRIEVED], [["a"]], k=5)
    assert set(summary) == {"hit_rate@5", "mrr", "ndcg@10", "precision@5", "recall@5"}


def test_aggregate_of_a_perfect_run():
    summary = aggregate([["a"], ["b"]], [["a"], ["b"]], k=1)
    assert summary["hit_rate@1"] == 1.0
    assert summary["mrr"] == 1.0


def test_aggregate_of_nothing_is_empty():
    assert aggregate([], []) == {}
