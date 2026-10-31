"""Upload and document-listing tests.

HashEmbedder and an in-memory Qdrant, so nothing downloads a model and nothing needs
Docker. The uploads land in a tmp_path that replaces `routes.DATA_DIR`, so a test run
never writes into the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api import routes
from rag.api.deps import Components, set_components
from rag.api.main import app
from rag.config import Settings
from rag.ingestion.embedding import HashEmbedder
from rag.retrieval.store import VectorStore

DIM = 32


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """An API with a real store and embedder, writing uploads into tmp_path."""
    store = VectorStore.connect(":memory:", "docs", DIM)
    store.ensure_collection()

    components = Components(
        settings=Settings(embedding_dim=DIM, chunk_size=160, chunk_overlap=64),
        embedder=HashEmbedder(DIM),
        store=store,
        generator=None,
        llm_error="no model configured for this test",
    )
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)

    # After the context opens, not before: entering TestClient runs the app's lifespan,
    # which calls set_components itself and would replace these with real ones built
    # against a Qdrant that is not running here.
    with TestClient(app) as test_client:
        set_components(components)
        yield test_client

    set_components(None)


def upload(client: TestClient, name: str, body: bytes):
    return client.post("/v1/documents", files={"file": (name, body, "text/plain")})


NOTES = (
    b"The UNIQUE constraint stops a column accepting duplicate values.\n\n"
    b"PRIMARY KEY combines NOT NULL and UNIQUE, and a table may define only one.\n\n"
    b"A FOREIGN KEY points at the primary key of another table.\n"
)


# --- listing ----------------------------------------------------------------


def test_an_empty_index_lists_no_documents(client):
    response = client.get("/v1/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": [], "total_chunks": 0}


def test_an_uploaded_document_appears_in_the_listing(client):
    upload(client, "notes.txt", NOTES)

    body = client.get("/v1/documents").json()

    assert [d["source"] for d in body["documents"]] == ["notes.txt"]
    assert body["total_chunks"] > 0


def test_the_listing_is_derived_from_the_index_not_a_side_table(client):
    """Deleting from the store must remove it from the listing, with nothing else touched."""
    upload(client, "notes.txt", NOTES)

    from rag.api.deps import get_components

    components = get_components()
    document_id = components.store.iter_chunks()[0].document_id
    components.store.delete_document(document_id)

    assert client.get("/v1/documents").json()["documents"] == []


# --- a successful upload ----------------------------------------------------


def test_upload_returns_201_and_what_became_searchable(client):
    response = upload(client, "notes.txt", NOTES)

    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "notes.txt"
    assert body["chunks"] > 0
    assert body["replaced"] is False


def test_an_uploaded_document_is_immediately_queryable(client):
    """201 has to mean indexed. A caller that has to poll was not told the truth."""
    upload(client, "notes.txt", NOTES)

    hits = client.post("/v1/search", json={"question": "UNIQUE constraint"}).json()["hits"]

    assert hits
    assert any("UNIQUE" in hit["text"] for hit in hits)


def test_the_file_is_written_to_the_data_directory(client, tmp_path: Path):
    upload(client, "notes.txt", NOTES)
    assert (tmp_path / "notes.txt").read_bytes() == NOTES


def test_re_uploading_the_same_name_reports_it_as_replaced(client):
    upload(client, "notes.txt", NOTES)
    second = upload(client, "notes.txt", NOTES + b"\nViews do not store a copy.\n")

    assert second.status_code == 201
    assert second.json()["replaced"] is True


def test_upload_warns_that_recorded_evaluations_no_longer_compare(client):
    """Easy to miss, expensive to discover: the corpus every run was measured against moved."""
    body = upload(client, "notes.txt", NOTES).json()
    assert "evaluation" in body["warning"].lower()


# --- rejected uploads -------------------------------------------------------


def test_a_filename_cannot_escape_the_data_directory(client, tmp_path: Path):
    """The filename is attacker-controlled; only its final component may be used."""
    response = upload(client, "../../../evil.txt", NOTES)

    assert response.status_code == 201
    assert response.json()["source"] == "evil.txt"
    assert (tmp_path / "evil.txt").exists()
    assert not (tmp_path.parent.parent.parent / "evil.txt").exists()


def test_a_windows_style_path_is_also_stripped(client, tmp_path: Path):
    response = upload(client, r"C:\Windows\System32\evil.txt", NOTES)

    assert response.status_code == 201
    assert "/" not in response.json()["source"] and "\\" not in response.json()["source"]


def test_an_unsupported_extension_is_refused_before_anything_is_written(client, tmp_path: Path):
    response = upload(client, "payload.exe", b"MZ\x90\x00")

    assert response.status_code == 415
    assert list(tmp_path.iterdir()) == []


def test_the_refusal_names_what_is_supported(client):
    detail = upload(client, "payload.exe", b"MZ").json()["detail"]
    assert ".pdf" in detail


def test_an_empty_file_is_refused(client):
    assert upload(client, "empty.txt", b"").status_code == 422


def test_a_file_over_the_size_limit_is_refused(client, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 64)

    response = upload(client, "big.txt", b"x" * 500)

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_cannot_be_indexed_is_not_left_behind(client, tmp_path: Path):
    """A file in data/ that no re-ingest will ever index is worse than no file."""
    response = upload(client, "scan.txt", b"  \n \n  ")

    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_a_failed_upload_does_not_appear_in_the_listing(client):
    upload(client, "scan.txt", b"   ")
    assert client.get("/v1/documents").json()["documents"] == []
