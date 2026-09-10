from rag.config import Settings


def test_index_fingerprint_changes_with_embedding_model():
    a = Settings(embedding_model="model-a")
    b = Settings(embedding_model="model-b")
    assert a.index_fingerprint() != b.index_fingerprint()


def test_index_fingerprint_changes_with_chunking():
    a = Settings(chunk_size=512)
    b = Settings(chunk_size=256)
    assert a.index_fingerprint() != b.index_fingerprint()
