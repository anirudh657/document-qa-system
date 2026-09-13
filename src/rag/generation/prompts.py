"""Prompt templates.

Two non-obvious choices encoded here:

1. Passage ordering. The strongest passages go at the start and end of the context
   block, because attention degrades in the middle of long contexts.
2. Explicit refusal. The model is told to say the corpus does not contain the answer
   rather than reasoning from general knowledge. Without this instruction a RAG
   system silently becomes a plain LLM whenever retrieval fails, which is the worst
   possible failure mode because it looks like success.
"""

from rag.models import RetrievedChunk

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


NOT_FOUND = "The provided documents do not contain an answer to this question."


def order_passages(passages: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder so the highest-scoring passages sit at the edges of the context.

    Models attend less reliably to the middle of a long context than to its start and
    end — the "lost in the middle" effect. Retrieval hands back a list sorted best to
    worst, and putting it in the prompt unchanged buries the second- and third-best
    passages exactly where they are least likely to be read.

    Strongest first, second-strongest last, and the rest folded inward.
    """
    ranked = sorted(passages, key=lambda p: p.final_score, reverse=True)
    head: list[RetrievedChunk] = []
    tail: list[RetrievedChunk] = []
    for index, passage in enumerate(ranked):
        (head if index % 2 == 0 else tail).append(passage)
    return head + tail[::-1]


def format_passage(passage: RetrievedChunk) -> str:
    """Render one passage with the citation label the model is told to reuse."""
    chunk = passage.chunk
    source = chunk.contextual_header or chunk.document_id
    section = f"(page {chunk.page_number})" if chunk.page_number else ""
    return PASSAGE_TEMPLATE.format(
        chunk_id=chunk.id[:8],
        source=source,
        section=section,
        text=chunk.text,
    )


def build_context(passages: list[RetrievedChunk], max_chars: int = 12000) -> str:
    """Pack passages into the context block, respecting a character budget.

    The budget is applied after ordering, so if it truncates it drops the weakest
    passages from the middle rather than the strongest from either end.
    """
    blocks: list[str] = []
    used = 0
    for passage in order_passages(passages):
        block = format_passage(passage)
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_prompt(question: str, passages: list[RetrievedChunk], max_chars: int = 12000) -> str:
    return ANSWER_TEMPLATE.format(
        context=build_context(passages, max_chars),
        question=question,
    )
