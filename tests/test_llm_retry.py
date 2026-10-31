"""Transient-failure retry tests for the Gemini client.

Driven through asyncio.run rather than async test functions, so they do not depend on
a pytest plugin or on asyncio_mode configuration.
"""

import asyncio

import httpx
import pytest

from rag.generation.llm import (
    MAX_ATTEMPTS,
    RATE_LIMITED,
    TRANSIENT_STATUS,
    GeminiClient,
    LLMError,
    QuotaError,
)

OK_BODY = {
    "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
}


class FakeClient:
    """Stands in for httpx.AsyncClient, replaying a fixed sequence of statuses."""

    def __init__(self, statuses: list[int]):
        self.statuses = statuses
        self.attempts = 0

    async def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        status = self.statuses[min(self.attempts, len(self.statuses) - 1)]
        self.attempts += 1
        body = OK_BODY if status == 200 else {"error": {"code": status}}
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))


def post(statuses: list[int]) -> tuple[httpx.Response, FakeClient]:
    client = FakeClient(statuses)
    gemini = GeminiClient(api_key="k")

    async def run() -> httpx.Response:
        return await gemini._post_with_retry(client, "http://x", {})

    return asyncio.run(run()), client


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Backoff is real seconds; the tests should not actually wait them out."""

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


def test_a_successful_call_is_not_retried():
    _, client = post([200])
    assert client.attempts == 1


def test_a_503_is_retried():
    """Gemini's free tier returns 503 on demand spikes — it means try again."""
    _, client = post([503, 200])
    assert client.attempts == 2


def test_the_retried_response_is_the_successful_one():
    response, _ = post([503, 503, 200])
    assert response.status_code == 200


@pytest.mark.parametrize("status", sorted(TRANSIENT_STATUS))
def test_every_transient_status_is_retried(status):
    _, client = post([status, 200])
    assert client.attempts == 2


def test_a_client_error_is_not_retried():
    """401 and 400 are the caller's fault; retrying wastes the rate limit."""
    _, client = post([401])
    assert client.attempts == 1


def test_retries_are_bounded():
    _, client = post([503])
    assert client.attempts == MAX_ATTEMPTS


def test_exhausted_retries_surface_as_an_actionable_error():
    response, _ = post([503])
    with pytest.raises(LLMError, match="after 4 attempts"):
        GeminiClient._raise_for_status(response)


# --- 429 is a budget, not a spike -------------------------------------------
#
# These exist because the opposite behaviour had a measured cost. Retrying 429 four
# times per query turned a 34-query warming pass into ~108 requests, which exhausted
# the free tier's daily allowance and cached two expansions for it.


def test_a_429_is_not_retried():
    _, client = post([RATE_LIMITED, 200])
    assert client.attempts == 1


def test_429_is_not_in_the_retryable_set():
    assert RATE_LIMITED not in TRANSIENT_STATUS


def test_a_429_raises_a_quota_error():
    response, _ = post([RATE_LIMITED])
    with pytest.raises(QuotaError):
        GeminiClient._raise_for_status(response)


def quota_response(quota_id: str, retry_delay: str = "", value: str = "") -> httpx.Response:
    """A 429 shaped like Gemini's: the quota it hit is in error.details."""
    details: list[dict] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": value}],
        }
    ]
    if retry_delay:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    return httpx.Response(
        RATE_LIMITED,
        json={"error": {"code": RATE_LIMITED, "status": "RESOURCE_EXHAUSTED", "details": details}},
        request=httpx.Request("POST", "http://x"),
    )


def test_a_per_day_quota_is_identified_as_such():
    error = GeminiClient._quota_failure(
        quota_response("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )
    assert error.scope == "per-day"


def test_a_per_minute_quota_is_identified_as_such():
    error = GeminiClient._quota_failure(
        quota_response("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    )
    assert error.scope == "per-minute"


def test_a_per_day_quota_says_waiting_will_not_help():
    """The advice that matters: re-running tonight cannot succeed."""
    error = GeminiClient._quota_failure(
        quota_response("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    )
    assert "DAILY" in str(error)
    assert "midnight US Pacific" in str(error)


def test_only_a_per_minute_quota_is_worth_waiting_out():
    assert not GeminiClient._quota_failure(quota_response("...PerMinute...")).exhausted
    assert GeminiClient._quota_failure(quota_response("...PerDay...")).exhausted


def test_an_unlabelled_quota_is_treated_as_exhausted():
    """Unknown means it might be the daily one, and grinding on is the costlier guess."""
    error = GeminiClient._quota_failure(quota_response(""))
    assert error.scope == "unknown"
    assert error.exhausted


def test_the_quota_id_and_limit_are_reported():
    error = GeminiClient._quota_failure(
        quota_response("GenerateRequestsPerDayPerProjectPerModel-FreeTier", value="250")
    )
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(error)
    assert "250" in str(error)


def test_a_suggested_retry_delay_is_captured():
    error = GeminiClient._quota_failure(quota_response("...PerMinute...", retry_delay="25s"))
    assert error.retry_after == pytest.approx(25.0)
    assert "Retry in 25s" in str(error)


def test_a_daily_quota_does_not_repeat_gemini_s_retry_suggestion():
    """Gemini sends a RetryInfo on daily exhaustions too, where a few seconds is wrong.

    Printing it would contradict "resets at midnight" in the same breath, so it stays
    on the exception and out of the message.
    """
    error = GeminiClient._quota_failure(quota_response("...PerDay...", retry_delay="33s"))
    assert error.retry_after == pytest.approx(33.0)
    assert "Retry in" not in str(error)


def test_a_body_without_quota_details_still_raises_cleanly():
    """Providers change their error shapes; a parse miss must not mask the 429."""
    response = httpx.Response(
        RATE_LIMITED, text="upstream html", request=httpx.Request("POST", "http://x")
    )
    error = GeminiClient._quota_failure(response)
    assert error.scope == "unknown"
    assert "429" in str(error)


def test_a_5xx_error_no_longer_blames_the_rate_limit():
    """It used to, which pointed every demand spike at the wrong cause."""
    response, _ = post([503])
    with pytest.raises(LLMError, match="unrelated to your quota"):
        GeminiClient._raise_for_status(response)


def test_a_rejected_key_is_still_reported_distinctly():
    response, _ = post([403])
    with pytest.raises(LLMError, match="rejected the API key"):
        GeminiClient._raise_for_status(response)
