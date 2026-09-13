"""LLM client: the only component that talks to a language model provider.

Kept behind a narrow interface so the rest of the system never imports a vendor SDK.
Swapping providers, or substituting a fake in tests, then touches one file.

Gemini is called over plain REST with httpx rather than through a vendor SDK. The
SDKs churn — the Google client library has been renamed and rewritten more than once —
while the HTTP contract stays stable, and httpx is already a dependency. One less
package to break on upgrade.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class LLMError(RuntimeError):
    """Raised when the provider rejects a request or returns something unusable."""


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

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 429:
            raise LLMError(
                "Gemini rate limit reached. The free tier allows roughly 10 requests "
                "per minute — wait a moment and retry."
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
            response = await client.post(
                url, headers=self._headers, json=self._payload(system, user, config)
            )
        self._raise_for_status(response)
        return self._parse(response.json())

    async def stream(
        self, system: str, user: str, config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        """Yield text as it is produced, so the caller can show partial answers."""
        import json

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
