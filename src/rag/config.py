"""Application configuration, loaded from environment variables.

Every tunable in the retrieval pipeline lives here rather than inline, so that an
eval run can snapshot the exact configuration that produced its numbers.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["anthropic", "openai", "local"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 64

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 50
    rerank_top_n: int = 5

    # Stores
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    postgres_url: str = "postgresql://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"

    # Cache
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl_seconds: int = 3600

    # Retrieval
    dense_top_k: int = 25
    sparse_top_k: int = 25
    chunk_size: int = 512
    chunk_overlap: int = 64
    enable_hyde: bool = False
    enable_query_rewrite: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 60
    log_level: str = "INFO"

    def index_fingerprint(self) -> str:
        """Identifies the index a given config produces.

        Changing the embedding model or chunking invalidates existing vectors; this
        string is stored with the collection so a mismatch is caught at startup
        rather than showing up as mysteriously bad retrieval.
        """
        return f"{self.embedding_model}:{self.chunk_size}:{self.chunk_overlap}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
