"""Loader tests.

Every fixture is written to a tmp_path rather than committed, so the suite runs
anywhere and does not depend on any particular document being present.
"""

from pathlib import Path

import pytest

from rag.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    EmptyTextLayerError,
    HtmlLoader,
    MarkdownLoader,
    PdfLoader,
    TextLoader,
    UnsupportedFormatError,
    content_hash,
    load,
    loader_for,
)

BODY = "Discrete structures cover sets, relations, and graphs in some detail here."


def test_registry_dispatches_by_extension():
    assert isinstance(loader_for(Path("a.pdf")), PdfLoader)
    assert isinstance(loader_for(Path("a.md")), MarkdownLoader)
    assert isinstance(loader_for(Path("a.txt")), TextLoader)
    assert isinstance(loader_for(Path("a.html")), HtmlLoader)


def test_registry_is_case_insensitive():
    assert isinstance(loader_for(Path("REPORT.PDF")), PdfLoader)


def test_unsupported_extension_names_the_alternatives():
    with pytest.raises(UnsupportedFormatError, match="No loader for"):
        loader_for(Path("archive.zip"))


def test_supported_extensions_are_exposed():
    assert {".pdf", ".md", ".txt", ".html", ".docx"} <= SUPPORTED_EXTENSIONS


def test_text_loader_reads_a_file(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text(BODY, encoding="utf-8")
    doc = load(path)
    assert doc.text == BODY
    assert doc.meta.doc_type == "text"
    assert doc.meta.title == "notes"


def test_empty_file_is_rejected_rather_than_indexed(tmp_path: Path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(EmptyTextLayerError):
        load(path)


def test_near_empty_file_is_rejected(tmp_path: Path):
    """A scanned PDF often leaks a few stray characters — still unusable."""
    path = tmp_path / "scan.txt"
    path.write_text("f\f\f", encoding="utf-8")
    with pytest.raises(EmptyTextLayerError, match="characters extracted"):
        load(path)


def test_markdown_captures_heading_hierarchy(tmp_path: Path):
    path = tmp_path / "guide.md"
    path.write_text(f"# Setup\n\n{BODY}\n\n## Installation\n\n{BODY}\n", encoding="utf-8")
    doc = load(path)
    headings = [s["heading"] for s in doc.sections]
    levels = [s["level"] for s in doc.sections]
    assert headings == ["Setup", "Installation"]
    assert levels == [1, 2]
    assert doc.meta.title == "Setup"


def test_markdown_sections_record_their_offset(tmp_path: Path):
    path = tmp_path / "guide.md"
    path.write_text(f"# One\n\n{BODY}\n\n## Two\n\n{BODY}\n", encoding="utf-8")
    doc = load(path)
    for section in doc.sections:
        assert doc.text[section["start"] :].startswith("#")


def test_html_strips_scripts_and_keeps_headings(tmp_path: Path):
    path = tmp_path / "page.html"
    path.write_text(
        f"<html><head><title>Doc</title></head><body>"
        f"<script>alert('x')</script><h1>Overview</h1><p>{BODY}</p></body></html>",
        encoding="utf-8",
    )
    doc = load(path)
    assert "alert" not in doc.text
    assert BODY in doc.text
    assert doc.meta.title == "Doc"
    assert [s["heading"] for s in doc.sections] == ["Overview"]


def test_normalisation_collapses_excess_blank_lines(tmp_path: Path):
    path = tmp_path / "messy.txt"
    path.write_text(f"{BODY}\n\n\n\n\n{BODY}", encoding="utf-8")
    assert "\n\n\n" not in load(path).text


def test_content_hash_is_stable_and_content_sensitive(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text(BODY, encoding="utf-8")
    b.write_text(BODY, encoding="utf-8")
    assert content_hash(a) == content_hash(b)

    b.write_text(BODY + " changed", encoding="utf-8")
    assert content_hash(a) != content_hash(b)
