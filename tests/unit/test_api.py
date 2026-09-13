"""API tests.

Components are injected directly rather than built from settings, so the suite needs
no Qdrant, no model download and no API key.
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from rag.api.deps import Components, get_components
from rag.api.main import app
from rag.config import Settings
from rag.generation.generator import Generator
from rag.generation.llm import ScriptedClient
from rag.ingestion.chunking import FixedSizeChunker
from rag.ingestion.embedding import HashEmbedder
from rag.ingestion.loaders import LoadedDocument
from rag.models import DocumentMeta
from rag.retrieval.store import VectorStore

DIM = 32
BODY = (
    "De Morgan laws relate complements of unions to intersections.\n\n"
    "A relation is transitive when a-b and b-c imply a-c.\n\n"
    "Power sets contain every possible subset of a given set."
)


def make_components(answer: str = "De Morgan's laws hold [{cid}].") -> Components:
    embedder = HashEmbedder(dimension=DIM)
    store = VectorStore.connect(":memory:", "api_test", DIM, fingerprint="test:512:64")
    store.ensure_collection()

    meta = DocumentMeta(
        source_path="data/practical.pdf",
        title="Practical File",
        doc_type="pdf",
        content_hash="abc123",
        ingested_at=datetime.now(UTC),
        page_count=1,
    )
    doc = LoadedDocument(BODY, meta, [{"heading": "Page 7", "level": 1, "page": 7, "start": 0}])
    chunks = FixedSizeChunker(chunk_size=40, overlap=8).split(doc)
    store.upsert(chunks, embedder.embed_chunks(chunks))

    text = answer.replace("{cid}", chunks[0].id[:8])
    return Components(
        settings=Settings(embedding_dim=DIM),
        embedder=embedder,
        store=store,
        generator=Generator(ScriptedClient([text])),
    )


def client_with(components: Components) -> TestClient:
    """Override the dependency rather than the module global.

    `TestClient(app)` runs the real lifespan, which would rebuild components from
    settings and try to reach a live Qdrant. Overriding the dependency bypasses that
    entirely, which is what FastAPI provides dependency_overrides for.
    """
    app.dependency_overrides[get_components] = lambda: components
    return TestClient(app)


@pytest.fixture
def client():
    with client_with(make_components()) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_root_reports_the_service(client):
    body = client.get("/").json()
    assert body["service"] == "document-qa-system"


def test_health_reports_ok_when_everything_works(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["dependencies"]["qdrant"].startswith("ok")
    assert body["dependencies"]["llm"].startswith("ok")


def test_health_includes_the_index_fingerprint(client):
    assert client.get("/v1/health").json()["index_fingerprint"]


def test_health_reports_degraded_without_an_llm():
    components = make_components()
    components.generator = None
    components.llm_error = "No Gemini API key."
    with client_with(components) as test_client:
        body = test_client.get("/v1/health").json()
    app.dependency_overrides.clear()
    assert body["status"] == "degraded"
    assert "No Gemini API key" in body["dependencies"]["llm"]


def test_health_stays_200_when_degraded():
    """A monitor must be able to read which dependency failed, not just "down"."""
    components = make_components()
    components.generator = None
    with client_with(components) as test_client:
        assert test_client.get("/v1/health").status_code == 200
    app.dependency_overrides.clear()


def test_query_returns_an_answer(client):
    body = client.post("/v1/query", json={"question": "what are de morgan laws"}).json()
    assert "De Morgan" in body["answer"]
    assert body["answered"]
    assert body["query_id"]


def test_query_returns_citations(client):
    body = client.post("/v1/query", json={"question": "de morgan"}).json()
    assert body["citations"]
    assert body["citations"][0]["chunk_id"]


def test_query_reports_tokens_and_latency(client):
    body = client.post("/v1/query", json={"question": "de morgan"}).json()
    assert body["input_tokens"] > 0
    assert body["latency_ms"] >= 0


def test_query_rejects_an_empty_question(client):
    assert client.post("/v1/query", json={"question": ""}).status_code == 422


def test_query_rejects_an_out_of_range_top_k(client):
    response = client.post("/v1/query", json={"question": "x", "top_k": 500})
    assert response.status_code == 422


def test_query_returns_503_without_an_llm():
    components = make_components()
    components.generator = None
    components.llm_error = "No Gemini API key."
    with client_with(components) as test_client:
        response = test_client.post("/v1/query", json={"question": "anything"})
    app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "Gemini" in response.json()["detail"]


def test_search_returns_hits_without_calling_the_llm(client):
    body = client.post("/v1/search", json={"question": "de morgan", "top_k": 3}).json()
    assert body["hits"]
    assert body["hits"][0]["score"] > 0
    assert body["hits"][0]["text"]


def test_search_respects_top_k(client):
    body = client.post("/v1/search", json={"question": "sets", "top_k": 2}).json()
    assert len(body["hits"]) <= 2


def test_search_works_without_an_llm_configured():
    """Retrieval must stay usable when only the LLM is misconfigured."""
    components = make_components()
    components.generator = None
    with client_with(components) as test_client:
        response = test_client.post("/v1/search", json={"question": "de morgan"})
    app.dependency_overrides.clear()
    assert response.status_code == 200


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/query" in schema["paths"]
    assert "/v1/search" in schema["paths"]
    assert "/v1/health" in schema["paths"]
