"""BM25 sparse retrieval tests."""

import pytest

from rag.models import Chunk
from rag.retrieval.sparse import SparseRetriever, tokenize

CORPUS = [
    ("a", "Experiment 1 (a): Array Operations", "oop.pdf", 2),
    ("b", "Create a user account and grant it the necessary privileges", "sql.pdf", 3),
    ("c", "Step 1: Create a Database. CREATE DATABASE startersql;", "sql.pdf", 5),
    (
        "d",
        "Indexes speed up lookups. CREATE INDEX idx_email ON users(email);",
        "sql.pdf",
        64,
    ),
    ("e", "A do-while loop guarantees the body runs at least once", "oop.pdf", 9),
]


@pytest.fixture
def retriever():
    return SparseRetriever(
        [
            Chunk(id=cid, document_id="doc", text=text, source=source, page_number=page)
            for cid, text, source, page in CORPUS
        ]
    )


# --- tokenisation ----------------------------------------------------------


def test_tokenize_lowercases():
    assert tokenize("CREATE DATABASE") == ["create", "database"]


def test_tokenize_keeps_identifiers_whole():
    """Splitting idx_email into two common words would destroy its rarity."""
    assert "idx_email" in tokenize("CREATE INDEX idx_email ON users(email);")


def test_tokenize_drops_punctuation():
    assert tokenize("users(email);") == ["users", "email"]


def test_tokenize_of_punctuation_only_is_empty():
    assert tokenize("--- ;;; ---") == []


def test_stopwords_are_kept_by_default():
    """The corpus is indexed unfiltered — only queries are stripped."""
    assert "the" in tokenize("the database")


def test_stopwords_are_dropped_from_queries():
    assert tokenize("What is Experiment 1 in the OOP lab file?", drop_stopwords=True) == [
        "experiment",
        "1",
        "oop",
        "lab",
        "file",
    ]


def test_a_query_of_only_stopwords_falls_back_to_the_unfiltered_tokens():
    """Filtering everything away would search for nothing at all."""
    assert tokenize("what is the", drop_stopwords=True) == ["what", "is", "the"]


# --- retrieval -------------------------------------------------------------


def test_exact_term_wins(retriever):
    """The case dense retrieval fails: a label carrying almost no meaning."""
    assert retriever.search("Experiment 1", top_k=3)[0].chunk.id == "a"


def test_rare_term_beats_topical_similarity(retriever):
    """'create a database' must beat 'create a user' — the distinguishing word is rare."""
    assert retriever.search("how do I create a database", top_k=3)[0].chunk.id == "c"


def test_identifier_is_retrievable(retriever):
    assert retriever.search("idx_email", top_k=3)[0].chunk.id == "d"


def test_results_carry_a_sparse_score(retriever):
    hit = retriever.search("database", top_k=1)[0]
    assert hit.sparse_score is not None
    assert hit.sparse_score > 0
    assert hit.dense_score is None


def test_results_are_ordered_best_first(retriever):
    scores = [hit.sparse_score for hit in retriever.search("create database", top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(retriever):
    assert len(retriever.search("create", top_k=2)) <= 2


def test_zero_scoring_chunks_are_dropped(retriever):
    """Padding results with chunks sharing no query term would poison fusion."""
    assert all(hit.sparse_score > 0 for hit in retriever.search("database", top_k=5))


def test_a_query_matching_nothing_returns_nothing(retriever):
    assert retriever.search("xylophone marsupial", top_k=5) == []


def test_a_query_of_pure_punctuation_returns_nothing(retriever):
    assert retriever.search("!!! ???", top_k=5) == []


def test_an_empty_corpus_returns_nothing():
    assert SparseRetriever([]).search("anything", top_k=5) == []


def test_the_contextual_header_is_searchable(retriever):
    """A query naming the document should match chunks from it.

    The corpus needs several documents for this to be meaningful: BM25 weights a term
    by how rare it is, so in a single-document corpus every term appears everywhere,
    IDF collapses to zero and nothing scores. That is a property of the metric, not a
    bug — but it means tests must use a realistic corpus.
    """
    chunks = [
        Chunk(
            id=cid,
            document_id="doc",
            text=text,
            source=source,
            contextual_header=f"{source} > Page {page}",
            page_number=page,
        )
        for cid, text, source, page in CORPUS
    ]
    hits = SparseRetriever(chunks).search("oop", top_k=3)
    assert hits
    assert all(hit.chunk.source == "oop.pdf" for hit in hits)


def test_a_chunk_that_tokenises_to_nothing_does_not_crash():
    chunk = Chunk(id="x", document_id="d", text="--- ;;; ---")
    assert SparseRetriever([chunk]).search("database", top_k=1) == []
