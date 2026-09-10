"""Cross-encoder reranking.

A bi-encoder embeds query and passage independently and compares vectors. A
cross-encoder reads both together and scores the pair, which is much more accurate
and much more expensive — hence a capped candidate set rather than the whole corpus.

Usually the single largest quality win in the pipeline. Watch NFR-2: this is also
the most likely thing to break the latency budget, and it must degrade gracefully
(fall back to fusion order, flag it in the trace) rather than fail the request.

TODO(phase-2): rerank(), latency benchmark
TODO(phase-4): timeout and fallback path
"""

from rag.models import RetrievedChunk


class CrossEncoderReranker:
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("TODO(phase-2)")
