"""Retrieval and generation metrics.

Retrieval and generation are scored separately because they fail separately. A wrong
answer means either the retriever missed or the generator ignored good context, and a
single end-to-end number cannot distinguish them.

Hit rate is the ceiling on everything downstream: if the right passage was never
retrieved, no amount of prompting recovers the answer.

TODO(phase-3): all of the below
"""

from typing import Any


def hit_rate_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    raise NotImplementedError("TODO(phase-3)")


def mean_reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    raise NotImplementedError("TODO(phase-3)")


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 10) -> float:
    raise NotImplementedError("TODO(phase-3)")


def faithfulness(answer: str, context: list[str]) -> float:
    """LLM-as-judge. Calibrate against hand labels before trusting this (see docs)."""
    raise NotImplementedError("TODO(phase-3)")


def answer_relevance(answer: str, question: str) -> float:
    raise NotImplementedError("TODO(phase-3)")


def correct_refusal_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of unanswerable questions the system correctly declined."""
    raise NotImplementedError("TODO(phase-3)")
