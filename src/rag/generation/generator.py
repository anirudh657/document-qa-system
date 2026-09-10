"""Answer synthesis.

Citation validation (FR-14) runs after generation: any cited chunk ID that was not in
the retrieved set is a bug and is treated as one, not logged as a warning.

TODO(phase-1): generate(), context packing within the token budget
TODO(phase-4): streaming, citation validation, token and cost accounting
"""

from collections.abc import AsyncIterator

from rag.models import Answer, RetrievedChunk


class Generator:
    async def generate(self, question: str, passages: list[RetrievedChunk]) -> Answer:
        raise NotImplementedError("TODO(phase-1)")

    async def stream(self, question: str, passages: list[RetrievedChunk]) -> AsyncIterator[str]:
        raise NotImplementedError("TODO(phase-4)")

    @staticmethod
    def validate_citations(answer: Answer, passages: list[RetrievedChunk]) -> list[str]:
        """Return any cited chunk IDs that were not in the retrieved set."""
        raise NotImplementedError("TODO(phase-4)")
