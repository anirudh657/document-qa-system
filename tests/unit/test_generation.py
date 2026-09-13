"""Generation tests: prompt assembly, citation handling, refusal behaviour.

Run against ScriptedClient, so the suite needs no API key and makes no network calls.
"""

import pytest

from rag.generation.generator import Generator
from rag.generation.llm import (
    GeminiClient,
    GenerationConfig,
    LLMError,
    ScriptedClient,
    build_llm,
)
from rag.generation.prompts import (
    NOT_FOUND,
    build_context,
    build_prompt,
    order_passages,
)
from rag.models import Chunk, RetrievedChunk


def passage(chunk_id: str, text: str, score: float, page: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id.ljust(32, "0"),
            document_id="doc1",
            text=text,
            contextual_header="Practical File",
            section_path=[f"Page {page}"] if page else [],
            page_number=page,
        ),
        dense_score=score,
    )


@pytest.fixture
def passages():
    return [
        passage("aaaaaaaa", "De Morgan's laws relate complements of unions.", 0.9, 7),
        passage("bbbbbbbb", "A relation is transitive when a-b and b-c imply a-c.", 0.7, 10),
        passage("cccccccc", "Power sets contain every subset of a set.", 0.5, 9),
    ]


# --- prompt assembly -------------------------------------------------------


def test_order_passages_puts_the_strongest_at_the_edges(passages):
    """Models attend less reliably to the middle of a long context."""
    ordered = order_passages(passages)
    assert ordered[0].final_score == 0.9
    assert ordered[-1].final_score == 0.7
    assert ordered[1].final_score == 0.5


def test_order_passages_handles_a_single_passage(passages):
    assert len(order_passages(passages[:1])) == 1


def test_context_includes_every_passage(passages):
    context = build_context(passages)
    for item in passages:
        assert item.chunk.text in context


def test_context_labels_passages_with_a_citable_id(passages):
    context = build_context(passages)
    assert "[aaaaaaaa]" in context


def test_context_includes_page_numbers(passages):
    assert "(page 7)" in build_context(passages)


def test_context_respects_the_character_budget(passages):
    assert len(build_context(passages, max_chars=120)) < len(build_context(passages))


def test_context_keeps_at_least_one_passage_however_small_the_budget(passages):
    assert build_context(passages, max_chars=1).strip()


def test_prompt_contains_the_question_and_the_context(passages):
    prompt = build_prompt("What are De Morgan's laws?", passages)
    assert "What are De Morgan's laws?" in prompt
    assert "De Morgan's laws relate" in prompt


# --- generation ------------------------------------------------------------


async def test_answer_carries_the_model_text(passages):
    llm = ScriptedClient(["De Morgan's laws state the complement rule [aaaaaaaa]."])
    answer = await Generator(llm).generate("What are De Morgan's laws?", passages)
    assert "complement rule" in answer.text
    assert answer.answered


async def test_retrieved_passages_actually_reach_the_model(passages):
    """If this regresses the system is a plain LLM wearing a RAG costume."""
    llm = ScriptedClient(["ok"])
    await Generator(llm).generate("question", passages)
    _, user_prompt = llm.calls[0]
    assert "De Morgan's laws relate" in user_prompt


async def test_empty_retrieval_refuses_without_calling_the_model():
    """No context means no grounded answer is possible — and no request is spent."""
    llm = ScriptedClient(["should never be used"])
    answer = await Generator(llm).generate("anything", [])
    assert answer.text == NOT_FOUND
    assert not answer.answered
    assert llm.calls == []


async def test_citations_resolve_to_real_passages(passages):
    llm = ScriptedClient(["Transitivity means a-b and b-c imply a-c [bbbbbbbb]."])
    answer = await Generator(llm).generate("what is transitivity", passages)
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 10
    assert answer.citations[0].quote


async def test_multiple_citations_are_all_captured(passages):
    llm = ScriptedClient(["Both apply [aaaaaaaa] and also [cccccccc]."])
    answer = await Generator(llm).generate("q", passages)
    assert len(answer.citations) == 2


async def test_repeated_citations_are_deduplicated(passages):
    llm = ScriptedClient(["First [aaaaaaaa], again [aaaaaaaa]."])
    answer = await Generator(llm).generate("q", passages)
    assert len(answer.citations) == 1


async def test_invented_citations_are_not_returned(passages):
    """A citation to a passage that was never supplied is a fabrication."""
    llm = ScriptedClient(["Made up source [deadbeef]."])
    answer = await Generator(llm).generate("q", passages)
    assert answer.citations == []


async def test_validate_citations_reports_fabricated_ids(passages):
    llm = ScriptedClient(["Real [aaaaaaaa] and fake [deadbeef]."])
    answer = await Generator(llm).generate("q", passages)
    assert Generator.validate_citations(answer, passages) == ["deadbeef"]


async def test_validate_citations_is_empty_for_a_clean_answer(passages):
    llm = ScriptedClient(["Grounded [aaaaaaaa]."])
    answer = await Generator(llm).generate("q", passages)
    assert Generator.validate_citations(answer, passages) == []


async def test_refusal_is_recorded_as_not_answered(passages):
    """Refusal rate must be measurable — Phase 3 scores it directly."""
    llm = ScriptedClient([NOT_FOUND])
    answer = await Generator(llm).generate("unrelated question", passages)
    assert not answer.answered


async def test_token_usage_is_recorded(passages):
    answer = await Generator(ScriptedClient(["short"])).generate("q", passages)
    assert answer.input_tokens > 0
    assert answer.output_tokens > 0


async def test_latency_is_recorded(passages):
    answer = await Generator(ScriptedClient(["short"])).generate("q", passages)
    assert answer.latency_ms >= 0


# --- client ----------------------------------------------------------------


def test_gemini_client_requires_a_key():
    with pytest.raises(LLMError, match=r"aistudio\.google\.com"):
        GeminiClient(api_key="")


def test_build_llm_returns_a_gemini_client():
    assert isinstance(build_llm("gemini", "gemini-flash-latest", "key"), GeminiClient)


def test_build_llm_rejects_unknown_providers():
    with pytest.raises(LLMError, match="Unsupported"):
        build_llm("nonexistent", "model", "key")


def test_temperature_defaults_to_zero():
    """A RAG answer should restate its sources, not improvise around them."""
    assert GenerationConfig().temperature == 0.0


def test_gemini_payload_sends_the_system_instruction_separately():
    client = GeminiClient(api_key="k")
    payload = client._payload("SYSTEM", "USER", GenerationConfig())
    assert payload["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert payload["contents"][0]["parts"][0]["text"] == "USER"


def test_blocked_prompt_is_reported_as_blocked():
    """No candidates means the safety filter fired — not that the model had nothing to say."""
    client = GeminiClient(api_key="k")
    completion = client._parse({"promptFeedback": {"blockReason": "SAFETY"}})
    assert completion.blocked
    assert completion.finish_reason == "SAFETY"


def test_parse_extracts_text_and_usage():
    client = GeminiClient(api_key="k")
    completion = client._parse(
        {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
        }
    )
    assert completion.text == "hello"
    assert completion.input_tokens == 12
    assert completion.output_tokens == 3
    assert not completion.blocked
