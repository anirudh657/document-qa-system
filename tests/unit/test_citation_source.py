"""Regression: a citation names its document, not a display header."""

from rag.generation.generator import Generator
from rag.models import Chunk, RetrievedChunk


def passage(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc-hash",
            text="A UNIQUE constraint ensures all values in a column are different.",
            source="MySQL Handbook.pdf",
            contextual_header="MySQL Handbook > Page 30",
            section_path=["Page 30"],
            page_number=30,
        )
    )


def citations_for(text: str):
    return Generator.extract_citations(text, [passage("a" * 32)])


def test_a_citation_names_the_document_file():
    assert citations_for("Use UNIQUE [aaaaaaaa].")[0].source_path == "MySQL Handbook.pdf"


def test_the_page_is_not_repeated_inside_the_name():
    """It read 'MySQL Handbook > Page 30' beside a separate 'p. 30' field."""
    citation = citations_for("Use UNIQUE [aaaaaaaa].")[0]
    assert "Page" not in citation.source_path
    assert citation.page_number == 30


def test_the_section_still_carries_the_heading():
    assert citations_for("Use UNIQUE [aaaaaaaa].")[0].section == "Page 30"
