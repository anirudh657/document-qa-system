"""Chunker tests."""

from datetime import UTC, datetime

import pytest

from rag.ingestion.chunking import (
    FixedSizeChunker,
    chunk_id,
    estimate_tokens,
)
from rag.ingestion.loaders import LoadedDocument
from rag.models import DocumentMeta

SENTENCE = "Sets and relations are covered in this section of the practical file. "


def make_doc(text: str, sections: list[dict] | None = None, title: str = "Practical File"):
    meta = DocumentMeta(
        source_path="data/practical.pdf",
        title=title,
        doc_type="pdf",
        content_hash="abc123",
        ingested_at=datetime.now(UTC),
        page_count=3,
    )
    return LoadedDocument(text, meta, sections or [])


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("") == 1


def test_chunk_ids_are_deterministic():
    assert chunk_id("doc", 0) == chunk_id("doc", 0)


def test_chunk_ids_differ_by_position_and_document():
    assert chunk_id("doc", 0) != chunk_id("doc", 1)
    assert chunk_id("doc-a", 0) != chunk_id("doc-b", 0)


def test_empty_document_yields_no_chunks():
    assert FixedSizeChunker().split(make_doc("   \n\n  ")) == []


def test_short_document_is_a_single_chunk():
    chunks = FixedSizeChunker(chunk_size=512, overlap=64).split(make_doc(SENTENCE))
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Sets and relations")


def test_long_document_is_split_into_several_chunks():
    chunks = FixedSizeChunker(chunk_size=50, overlap=10).split(make_doc(SENTENCE * 20))
    assert len(chunks) > 1


def test_chunks_respect_the_token_budget():
    chunker = FixedSizeChunker(chunk_size=60, overlap=10)
    chunks = chunker.split(make_doc(SENTENCE * 20))
    # A lone sentence longer than the budget is emitted intact rather than cut mid-clause.
    assert all(c.token_count <= 60 for c in chunks[:-1])


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="must be smaller"):
        FixedSizeChunker(chunk_size=100, overlap=100)


def test_sentences_are_never_cut_in_half():
    chunks = FixedSizeChunker(chunk_size=40, overlap=8).split(make_doc(SENTENCE * 10))
    for chunk in chunks:
        assert not chunk.text.startswith("and relations")


def test_contextual_header_combines_title_and_section():
    doc = make_doc(
        SENTENCE,
        sections=[{"heading": "Page 1", "level": 1, "page": 1, "start": 0}],
    )
    chunk = FixedSizeChunker().split(doc)[0]
    assert chunk.contextual_header == "Practical File > Page 1"
    assert chunk.section_path == ["Page 1"]


def test_embedding_text_includes_the_header():
    doc = make_doc(SENTENCE, sections=[{"heading": "Page 1", "level": 1, "page": 1, "start": 0}])
    chunk = FixedSizeChunker().split(doc)[0]
    assert "Practical File > Page 1" in chunk.embedding_text()


def test_page_numbers_follow_document_order_despite_repeated_footers():
    """Regression: repeated boilerplate must not misattribute a chunk's page.

    Looking a chunk's text up with str.find returns the FIRST match in the document,
    so a running footer that appears on every page pulled later chunks back to page 1.
    Offsets are now tracked from the split itself.
    """
    footer = "URN: 2581258 25301P "
    body = "".join(
        f"{footer}Page {n} discusses topic number {n} in detail here. " for n in range(1, 6)
    )

    sections = []
    cursor = 0
    for n in range(1, 6):
        sections.append({"heading": f"Page {n}", "level": 1, "page": n, "start": cursor})
        cursor += len(f"{footer}Page {n} discusses topic number {n} in detail here. ")

    chunks = FixedSizeChunker(chunk_size=15, overlap=3).split(make_doc(body, sections))
    pages = [c.page_number for c in chunks if c.page_number]
    assert pages == sorted(pages)
    assert len(set(pages)) > 1


def test_chunks_carry_the_document_hash_as_their_document_id():
    chunks = FixedSizeChunker().split(make_doc(SENTENCE))
    assert all(c.document_id == "abc123" for c in chunks)


def test_re_chunking_the_same_document_produces_identical_ids():
    doc = make_doc(SENTENCE * 12)
    first = FixedSizeChunker(chunk_size=40, overlap=8).split(doc)
    second = FixedSizeChunker(chunk_size=40, overlap=8).split(doc)
    assert [c.id for c in first] == [c.id for c in second]
