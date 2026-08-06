"""
Configuration loaded from environment variables.
All secrets come from the environment — never hardcoded.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # LLM settings
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500

    # Memory retrieval settings
    episodic_top_k: int = 3          # Max episodic events to retrieve per turn
    semantic_top_k: int = 2          # Max semantic facts to retrieve per turn
    recency_weight_alpha: float = 0.3  # 0 = pure similarity, 1 = pure recency
    recency_days: int = 30           # Only consider events within this window

    # Promotion pipeline settings
    promotion_min_evidence: int = 3  # Min independent events to promote a semantic fact
    semantic_ttl_days: int = 90      # Default TTL for promoted semantic facts

    # Token budget for working memory assembly
    episodic_token_budget: int = 1200
    semantic_token_budget: int = 800

    # Demo / offline mode
    demo_mode: bool = True

    # Vector DB (Qdrant)
    qdrant_url: str = ""             # Empty = use in-memory Qdrant
    qdrant_collection_episodic: str = "episodic_events"
    qdrant_collection_semantic: str = "semantic_facts"

    # Embedding model
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        episodic_top_k=int(os.getenv("EPISODIC_TOP_K", "3")),
        semantic_top_k=int(os.getenv("SEMANTIC_TOP_K", "2")),
        recency_weight_alpha=float(os.getenv("RECENCY_WEIGHT_ALPHA", "0.3")),
        recency_days=int(os.getenv("RECENCY_DAYS", "30")),
        promotion_min_evidence=int(os.getenv("PROMOTION_MIN_EVIDENCE", "3")),
        semantic_ttl_days=int(os.getenv("SEMANTIC_TTL_DAYS", "90")),
        episodic_token_budget=int(os.getenv("EPISODIC_TOKEN_BUDGET", "1200")),
        semantic_token_budget=int(os.getenv("SEMANTIC_TOKEN_BUDGET", "800")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        qdrant_url=os.getenv("QDRANT_URL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
