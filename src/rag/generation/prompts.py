"""Prompt templates.

Two non-obvious choices encoded here:

1. Passage ordering. The strongest passages go at the start and end of the context
   block, because attention degrades in the middle of long contexts.
2. Explicit refusal. The model is told to say the corpus does not contain the answer
   rather than reasoning from general knowledge. Without this instruction a RAG
   system silently becomes a plain LLM whenever retrieval fails, which is the worst
   possible failure mode because it looks like success.
"""

from typing import Any

SYSTEM_PROMPT = """You answer questions using only the provided context passages.

Rules:
- Every factual claim must be supported by a passage. Cite it as [chunk_id].
- Never cite a passage that is not in the context below.
- If the context does not contain the answer, say exactly: "The provided documents \
do not contain an answer to this question." Do not answer from general knowledge.
- If the context partially answers the question, answer that part and state plainly \
what is missing.
- Be concise. Do not restate the question.
"""

ANSWER_TEMPLATE = """<context>
{context}
</context>

Question: {question}"""

PASSAGE_TEMPLATE = """[{chunk_id}] {source} {section}
{text}"""


def order_passages(passages: list[Any]) -> list[Any]:
    """Reorder so the highest-scoring passages sit at the edges of the context."""
    raise NotImplementedError("TODO(phase-1)")
