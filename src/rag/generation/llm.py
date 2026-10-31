"""LLM client: the only component that talks to a language model provider.

Kept behind a narrow interface so the rest of the system never imports a vendor SDK.
Swapping providers, or substituting a fake in tests, then touches one file.

Gemini is called over plain REST with httpx rather than through a vendor SDK. The
SDKs churn — the Google client library has been renamed and rewritten more than once —
while the HTTP contract stays stable, and httpx is already a dependency. One less
package to break on upgrade.
"""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

TRANSIENT_STATUS = frozenset({500, 502, 503, 504})
"""Statuses worth retrying.

503 is not an edge case on a free tier — Gemini returns it whenever demand spikes, and
it means "try again", not "this request was wrong". Treating it as fatal made a
32-question eval pass silently fall back to its baseline: every call failed, every
failure degraded gracefully, and the run reported numbers that measured nothing.
"""

RATE_LIMITED = 429
"""Deliberately **not** retried, unlike the 5xx statuses above.

429 does not mean "try again", it means "you are over quota" — and each retry spends
another unit of the quota that is already gone. Measured on this project: a 34-query
cache-warming pass retried every 429 four times, spent roughly 108 requests to cache
two expansions, and ran the free tier's daily allowance down to nothing doing it. The
retry was not merely useless here, it was the thing that exhausted the quota.

So the two failure modes now get the two different responses they need. A 5xx is a
spike: back off and retry. A 429 is a budget: stop, report which limit was hit and when
it clears, and let the caller decide. Pacing calls apart (see `QueryTransformer`) is
what prevents the per-minute limit; nothing prevents the per-day one but waiting.
"""

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0


class LLMError(RuntimeError):
    """Raised when the provider rejects a request or returns something unusable."""


class QuotaError(LLMError):
    """A 429, carrying which quota was actually hit.

    `scope` is the field that decides what to do next, and a bare status code hides it:
    a per-minute limit clears in under a minute, a per-day limit does not clear until
    the provider resets it. An evening spent re-running a command that cannot succeed
    until tomorrow is the cost of not knowing which one you are looking at.
    """

    def __init__(
        self,
        message: str,
        scope: str = "unknown",
        quota_id: str = "",
        limit: str = "",
        retry_after: float = 0.0,
    ):
        super().__init__(message)
        self.scope = scope
        self.quota_id = quota_id
        self.limit = limit
        self.retry_after = retry_after

    @property
    def exhausted(self) -> bool:
        """True when waiting a moment cannot help, so a batch should stop, not grind on."""
        return self.scope != "per-minute"


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    blocked: bool = False


@dataclass
class GenerationConfig:
    """Sampling settings.

    Temperature defaults to 0.0. A RAG system is meant to restate what the retrieved
    passages say, not to be creative about them — sampling variety here shows up as
    invented detail and as eval scores that move between runs for no reason.
    """

    temperature: float = 0.0
    max_output_tokens: int = 1024
    stop: list[str] = field(default_factory=list)


class LLMClient(ABC):
    model: str

    @abstractmethod
    async def complete(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> Completion: ...

    @abstractmethod
    def stream(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> AsyncIterator[str]: ...


class GeminiClient(LLMClient):
    """Google Gemini via the REST API.

    The free tier is rate-limited rather than credit-limited, so 429 is an expected
    condition during normal development, not an exceptional one. It is surfaced with a
    message that says what to do rather than a bare status code.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-latest",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise LLMError(
                "No Gemini API key. Set GEMINI_API_KEY in .env — "
                "get one free at https://aistudio.google.com/apikey"
            )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _payload(self, system: str, user: str, config: GenerationConfig) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_output_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if config.stop:
            body["generationConfig"]["stopSequences"] = config.stop
        return body

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        """POST, retrying transient failures with exponential backoff and jitter.

        Jitter matters because a batch of queries fails together and would otherwise
        retry in lockstep, re-creating the same spike that caused the failure.

        A 429 is not in `TRANSIENT_STATUS` and so returns on the first attempt — see
        that constant for why retrying it is actively harmful.
        """
        last: httpx.Response | None = None
        for attempt in range(MAX_ATTEMPTS):
            response = await client.post(url, headers=self._headers, json=payload)
            if response.status_code not in TRANSIENT_STATUS:
                return response

            last = response
            if attempt == MAX_ATTEMPTS - 1:
                break
            delay = BACKOFF_BASE_SECONDS * (2**attempt)
            await asyncio.sleep(delay + random.uniform(0, delay * 0.25))

        assert last is not None
        return last

    @staticmethod
    def _quota_failure(response: httpx.Response) -> QuotaError:
        """Turn a 429 body into a specific, actionable error.

        Gemini already says exactly which limit was hit, in `error.details`: a
        `QuotaFailure` names the quota and its value, and a `RetryInfo` gives a delay.
        The previous version of this method discarded the body and reported the status
        code alone, which is how a per-day exhaustion spent an evening being mistaken
        for a per-minute one.
        """
        quota_id = limit = ""
        retry_after = 0.0
        try:
            details = (response.json().get("error") or {}).get("details") or []
        except ValueError:
            details = []

        for detail in details:
            kind = str(detail.get("@type", ""))
            if kind.endswith("QuotaFailure"):
                for violation in detail.get("violations") or []:
                    quota_id = str(violation.get("quotaId", "")) or quota_id
                    limit = str(violation.get("quotaValue", "")) or limit
            elif kind.endswith("RetryInfo"):
                raw = str(detail.get("retryDelay", "")).rstrip("s")
                if raw.replace(".", "", 1).isdigit():
                    retry_after = float(raw)

        flat = quota_id.lower()
        if "perday" in flat:
            scope = "per-day"
        elif "perminute" in flat:
            scope = "per-minute"
        else:
            scope = "unknown"

        if scope == "per-day":
            advice = (
                "This is the DAILY quota, not the per-minute one. Waiting a few minutes "
                "will not help — it resets at midnight US Pacific."
            )
        elif scope == "per-minute":
            advice = "This is the per-minute quota. It clears within a minute."
        else:
            advice = (
                "Gemini did not say which quota this was. Check the per-day request count "
                "at https://aistudio.google.com before retrying."
            )

        parts = [f"Gemini returned HTTP {RATE_LIMITED} (quota exhausted).", advice]
        if quota_id:
            parts.append(f"Quota: {quota_id}" + (f" (limit {limit})." if limit else "."))
        # Gemini attaches a RetryInfo to daily exhaustions too, where it is wrong: the
        # suggested few seconds cannot clear a quota that resets tomorrow. Reporting it
        # anyway would contradict the advice above in the same sentence, so it is kept
        # on the exception for a caller that wants it and left out of the message.
        if retry_after and scope == "per-minute":
            parts.append(f"Retry in {retry_after:g}s.")
        return QuotaError(
            " ".join(parts),
            scope=scope,
            quota_id=quota_id,
            limit=limit,
            retry_after=retry_after,
        )

    @classmethod
    def _raise_for_status(cls, response: httpx.Response) -> None:
        if response.status_code == RATE_LIMITED:
            raise cls._quota_failure(response)
        if response.status_code in TRANSIENT_STATUS:
            raise LLMError(
                f"Gemini returned HTTP {response.status_code} after {MAX_ATTEMPTS} attempts. "
                "Gemini returns 5xx when demand spikes; this usually clears within a few "
                "minutes and is unrelated to your quota."
            )
        if response.status_code in (401, 403):
            raise LLMError(f"Gemini rejected the API key (HTTP {response.status_code}).")
        if response.status_code >= 400:
            raise LLMError(f"Gemini returned HTTP {response.status_code}: {response.text[:200]}")

    def _parse(self, data: dict[str, Any]) -> Completion:
        candidates = data.get("candidates") or []
        if not candidates:
            # A blocked prompt returns no candidates at all. Reporting this as an empty
            # answer would look like the model simply had nothing to say.
            reason = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
            return Completion(text="", finish_reason=reason, blocked=True, model=self.model)

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text.strip(),
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            model=self.model,
            finish_reason=candidate.get("finishReason", ""),
        )

    async def complete(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> Completion:
        config = config or GenerationConfig()
        url = f"{GEMINI_BASE}/{self.model}:generateContent"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await self._post_with_retry(client, url, self._payload(system, user, config))
        self._raise_for_status(response)
        return self._parse(response.json())

    async def stream(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        """Yield text as it is produced, so the caller can show partial answers."""
        config = config or GenerationConfig()
        url = f"{GEMINI_BASE}/{self.model}:streamGenerateContent?alt=sse"

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream(
                "POST", url, headers=self._headers, json=self._payload(system, user, config)
            ) as response,
        ):
            if response.status_code >= 400:
                await response.aread()
                self._raise_for_status(response)

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                for candidate in chunk.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        if part.get("text"):
                            yield part["text"]


class ScriptedClient(LLMClient):
    """Returns canned responses. For tests, so the suite needs no key and no network.

    Records every call so a test can assert on what the prompt actually contained —
    which is the only way to verify that retrieved passages reached the model.
    """

    def __init__(self, responses: list[str] | None = None, model: str = "scripted"):
        self.responses = responses or ["The provided documents do not contain an answer."]
        self.model = model
        self.calls: list[tuple[str, str]] = []
        self._index = 0

    def _next(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        text = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return text

    async def complete(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> Completion:
        text = self._next(system, user)
        return Completion(
            text=text,
            input_tokens=len(system) + len(user),
            output_tokens=len(text),
            model=self.model,
            finish_reason="STOP",
        )

    async def stream(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        for word in self._next(system, user).split(" "):
            yield word + " "


def build_llm(provider: str, model: str, api_key: str) -> LLMClient:
    """Construct the configured client.

    TODO(phase-4): anthropic and openai backends behind this same interface.
    """
    if provider == "gemini":
        return GeminiClient(api_key, model)
    raise LLMError(f"Unsupported LLM provider {provider!r}. Supported: gemini")
