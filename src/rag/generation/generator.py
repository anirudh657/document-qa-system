"""Answer synthesis: retrieved passages in, cited answer out.

Citation validation (FR-14) runs after generation. A citation pointing at a chunk that
was never retrieved is a fabrication, and it is treated as a defect rather than logged
as a warning — an answer that cites a passage nobody supplied is exactly the failure
RAG exists to prevent.
"""

from __future__ import annotations

import re
import time
import uuid

from rag.generation.llm import GenerationConfig, LLMClient
from rag.generation.prompts import NOT_FOUND, SYSTEM_PROMPT, build_prompt
from rag.models import Answer, Citation, RetrievedChunk

CITATION_PATTERN = re.compile(r"\[([0-9a-f]{6,32})\]")
QUOTE_CHARS = 160


class Generator:
    """Turns retrieved passages into an answer with validated citations."""

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str = SYSTEM_PROMPT,
        max_context_chars: int = 12000,
        config: GenerationConfig | None = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.config = config or GenerationConfig()

    async def generate(self, question: str, passages: list[RetrievedChunk]) -> Answer:
        started = time.perf_counter()
        query_id = str(uuid.uuid4())

        # Retrieval returning nothing is answered without calling the model at all.
        # There is no context to ground an answer in, so a call could only produce an
        # ungrounded one — and it would cost a request against the rate limit.
        if not passages:
            return Answer(
                query_id=query_id,
                text=NOT_FOUND,
                citations=[],
                answered=False,
                latency_ms=self._elapsed(started),
            )

        completion = await self.llm.complete(
            self.system_prompt,
            build_prompt(question, passages, self.max_context_chars),
            self.config,
        )

        if completion.blocked:
            return Answer(
                query_id=query_id,
                text="The request was blocked by the provider's safety filters.",
                citations=[],
                answered=False,
                latency_ms=self._elapsed(started),
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )

        citations = self.extract_citations(completion.text, passages)
        return Answer(
            query_id=query_id,
            text=completion.text,
            citations=citations,
            answered=self._is_answered(completion.text),
            latency_ms=self._elapsed(started),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

    @staticmethod
    def _elapsed(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _is_answered(text: str) -> bool:
        """Whether the model actually answered or declined.

        Tracked separately from the text so the refusal rate is measurable in Phase 3.
        A system that never refuses is not grounded, it is guessing fluently.
        """
        return NOT_FOUND.lower().rstrip(".") not in text.lower()

    @staticmethod
    def extract_citations(text: str, passages: list[RetrievedChunk]) -> list[Citation]:
        """Pull cited chunk IDs out of the answer and resolve them to real passages.

        Only citations matching a retrieved passage are returned. A cited ID that was
        never supplied is dropped here and reported by `validate_citations`.
        """
        by_prefix = {p.chunk.id[:8]: p.chunk for p in passages}
        citations: list[Citation] = []
        seen: set[str] = set()

        for match in CITATION_PATTERN.findall(text):
            chunk = by_prefix.get(match[:8])
            if chunk is None or chunk.id in seen:
                continue
            seen.add(chunk.id)
            citations.append(
                # `source_path` is the document, not a display string. It held
                # `contextual_header` — "Report > Page 5" — because it was written
                # before chunks carried a `source`, and a reader then saw the page
                # twice: once inside the name and once in the page field.
                Citation(
                    chunk_id=chunk.id,
                    source_path=chunk.source or chunk.document_id,
                    section=chunk.section_path[0] if chunk.section_path else None,
                    page_number=chunk.page_number,
                    quote=chunk.text[:QUOTE_CHARS].strip(),
                )
            )
        return citations

    @staticmethod
    def validate_citations(answer: Answer, passages: list[RetrievedChunk]) -> list[str]:
        """Return any cited chunk IDs that were not in the retrieved set.

        A non-empty result means the model invented a citation. Phase 3 treats this as
        a hard failure rather than a score, because it cannot be traded off against
        anything else.
        """
        supplied = {p.chunk.id[:8] for p in passages}
        return [
            cited for cited in CITATION_PATTERN.findall(answer.text) if cited[:8] not in supplied
        ]
