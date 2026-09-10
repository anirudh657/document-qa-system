"""Query rewriting and expansion.

Users ask short, underspecified questions in vocabulary that does not match the
source documents. Rewriting closes that gap before retrieval runs.

HyDE (generate a hypothetical answer, embed that instead of the question) helps on
some corpora and hurts on others, so it sits behind a flag and the eval harness
decides — not intuition.

TODO(phase-2): rewrite(), expand(), hyde(), filter extraction
"""

from typing import Any


class QueryTransformer:
    def rewrite(self, query: str) -> str:
        raise NotImplementedError("TODO(phase-2)")

    def hyde(self, query: str) -> str:
        raise NotImplementedError("TODO(phase-2)")

    def extract_filters(self, query: str) -> dict[str, Any]:
        """Pull date ranges, source names and doc types out of natural language."""
        raise NotImplementedError("TODO(phase-2)")
