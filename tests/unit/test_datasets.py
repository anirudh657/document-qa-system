"""Golden dataset tests."""

import pytest

from rag.evaluation.datasets import (
    DatasetError,
    GoldenQuestion,
    describe,
    filter_by_tags,
    load_dataset,
    page_key,
    retrieved_keys,
)
from rag.models import Chunk, RetrievedChunk

ROW = (
    '{"id": "q-001", "question": "How do I create a database in MySQL?", '
    '"source": "MySQL Handbook.pdf", "expected_pages": [5], '
    '"expected_answer": "CREATE DATABASE startersql;", "tags": ["mysql", "rare-term"]}'
)
REFUSAL = (
    '{"id": "q-002", "question": "Who won the 2026 world cup?", '
    '"expected_pages": [], "answerable": false, "tags": ["refusal"]}'
)


def hit(source: str, page: int | None, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id="d1",
            text="text",
            source=source,
            contextual_header=f"{source} > Page {page}",
            page_number=page,
        ),
        dense_score=0.5,
    )


def write(tmp_path, *lines):
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- page keys -------------------------------------------------------------


def test_page_key_is_stable_across_chunking_changes():
    """The whole reason expectations are pages, not chunk ids."""
    assert page_key("MySQL Handbook.pdf", 5) == page_key("MySQL Handbook.pdf", 5)


def test_page_key_ignores_the_directory():
    assert page_key("data/MySQL Handbook.pdf", 5) == page_key("MySQL Handbook.pdf", 5)


def test_page_key_ignores_the_extension():
    """A golden source is a filename; a chunk carries the title, which has no
    extension. Both must normalise to the same key or nothing ever matches."""
    assert page_key("MySQL Handbook.pdf", 5) == page_key("MySQL Handbook", 5)


def test_page_key_distinguishes_pages():
    assert page_key("a.pdf", 5) != page_key("a.pdf", 6)


def test_retrieved_keys_preserve_rank():
    keys = retrieved_keys([hit("a.pdf", 3), hit("a.pdf", 1, "c2")])
    assert keys == ["a#p3", "a#p1"]


def test_two_chunks_from_one_page_count_once():
    """Otherwise smaller chunks would score better purely by returning more pieces."""
    keys = retrieved_keys([hit("a.pdf", 5), hit("a.pdf", 5, "c2")])
    assert keys == ["a#p5"]


# --- schema ----------------------------------------------------------------


def test_answerable_question_requires_expected_pages():
    with pytest.raises(ValueError, match="expected_pages"):
        GoldenQuestion(id="q", question="?", answerable=True)


def test_unanswerable_question_must_not_list_pages():
    with pytest.raises(ValueError, match="must not list"):
        GoldenQuestion(id="q", question="?", answerable=False, expected_pages=[3])


def test_relevant_keys_match_the_retrieved_key_format():
    question = GoldenQuestion(id="q", question="?", source="MySQL Handbook.pdf", expected_pages=[5])
    assert question.relevant_keys() == retrieved_keys([hit("MySQL Handbook.pdf", 5)])


# --- loading ---------------------------------------------------------------


def test_load_reads_every_question(tmp_path):
    assert len(load_dataset(write(tmp_path, ROW, REFUSAL))) == 2


def test_load_preserves_fields(tmp_path):
    question = load_dataset(write(tmp_path, ROW))[0]
    assert question.expected_pages == [5]
    assert question.expected_answer == "CREATE DATABASE startersql;"
    assert "rare-term" in question.tags


def test_load_skips_blank_lines_and_comments(tmp_path):
    assert len(load_dataset(write(tmp_path, ROW, "", "// a note", REFUSAL))) == 2


def test_load_reports_the_offending_line_number(tmp_path):
    with pytest.raises(DatasetError, match=":2"):
        load_dataset(write(tmp_path, ROW, "{not json"))


def test_load_rejects_duplicate_ids(tmp_path):
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(write(tmp_path, ROW, ROW))


def test_load_rejects_an_empty_file(tmp_path):
    with pytest.raises(DatasetError, match="no questions"):
        load_dataset(write(tmp_path, ""))


def test_load_reports_a_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="does not exist"):
        load_dataset(tmp_path / "absent.jsonl")


# --- slicing and shape -----------------------------------------------------


def test_filter_by_tags_keeps_matching_questions(tmp_path):
    questions = load_dataset(write(tmp_path, ROW, REFUSAL))
    assert len(filter_by_tags(questions, ["refusal"])) == 1


def test_filter_by_tags_returns_nothing_for_an_unknown_tag(tmp_path):
    questions = load_dataset(write(tmp_path, ROW))
    assert filter_by_tags(questions, ["nonexistent"]) == []


def test_describe_counts_answerable_and_unanswerable(tmp_path):
    summary = describe(load_dataset(write(tmp_path, ROW, REFUSAL)))
    assert summary["questions"] == 2
    assert summary["answerable"] == 1
    assert summary["unanswerable"] == 1


def test_retrieved_keys_use_the_filename_not_the_title():
    """Regression: matching on the document title silently scored every question 0.

    A PDF's embedded title often differs from its filename, and golden questions are
    written against filenames. Reading the title meant expected and retrieved keys
    never matched, and the harness reported a well-formatted table of zeros.
    """
    chunk = Chunk(
        id="c1",
        document_id="d1",
        text="text",
        source="MySQL Handbook.pdf",
        contextual_header="Some Other Title > Page 5",
        page_number=5,
    )
    assert retrieved_keys([RetrievedChunk(chunk=chunk, dense_score=0.5)]) == ["MySQL Handbook#p5"]
