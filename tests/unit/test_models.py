"""Tests for the domain models — the one part of Phase 0 that is already real."""

from rag.models import Chunk, RetrievedChunk


def make_chunk(**kwargs) -> Chunk:
    defaults = dict(id="c1", document_id="d1", text="The refund window is 30 days.")
    return Chunk(**{**defaults, **kwargs})


def test_embedding_text_includes_contextual_header():
    chunk = make_chunk(contextual_header="Billing Policy > Refunds")
    assert "Billing Policy > Refunds" in chunk.embedding_text()
    assert "30 days" in chunk.embedding_text()


def test_embedding_text_without_header_is_just_the_text():
    assert make_chunk().embedding_text() == "The refund window is 30 days."


def test_final_score_prefers_rerank_over_fusion():
    rc = RetrievedChunk(chunk=make_chunk(), dense_score=0.4, fusion_score=0.6, rerank_score=0.9)
    assert rc.final_score == 0.9


def test_final_score_falls_back_through_the_chain():
    assert RetrievedChunk(chunk=make_chunk(), dense_score=0.4).final_score == 0.4
    assert RetrievedChunk(chunk=make_chunk()).final_score == 0.0
