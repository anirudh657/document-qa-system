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
    llm_provider: Literal["gemini", "anthropic", "openai", "local"] = "gemini"
    llm_model: str = "gemini-flash-latest"
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    def llm_key(self) -> str:
        """The API key for the configured provider."""
        return {
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }.get(self.llm_provider, "")

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 64

    # Reranker
    enable_rerank: bool = True
    """Rescore retrieved candidates with a cross-encoder before answering.

    On by default: the golden set shows six of eleven retrieval failures returning the
    page directly beside the correct one, which is a ranking-precision problem. Turn it
    off to measure retrieval without it — the comparison is the point.
    """

    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 25
    """Passages scored by the cross-encoder per query.

    Wider finds more, but every candidate is a forward pass, so this is the main
    latency and cost lever. It also caps what reranking can achieve: a page missing
    from the pool cannot be promoted into the results.
    """

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
    enable_hybrid: bool = True
    """Run BM25 alongside dense retrieval and fuse the results.

    On by default because the two cover each other's blind spots. Turn it off to
    measure what dense retrieval alone achieves — the comparison is the point.
    """

    rrf_k: int = 60
    enable_hyde: bool = False

    enable_query_rewrite: bool = False
    """Expand the query with document terminology before retrieving.

    Off by default: it costs an LLM call per query, so it must earn its place on the
    golden set before being switched on. Targets the four failures whose query
    describes an intent without ever using the term the document is indexed under.
    """

    query_rewrite_cache: str = "eval/cache/rewrites.json"
    """Where expanded queries are cached.

    Not an optimisation. A 32-question eval pass against a ~10 requests-per-minute free
    tier would otherwise take minutes and risk rate-limiting partway through, which
    would change the result for reasons unrelated to the change being measured.
    """

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
