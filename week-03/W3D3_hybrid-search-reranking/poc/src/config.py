"""Configuration loaded from environment variables.

W3D3 — Hybrid Search & Reranking
All secrets and tuneable parameters live here. Logic lives in hybrid_search_core.py.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # LLM settings
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000

    # Retrieval settings
    bm25_top_k: int = 100          # Candidates fetched from BM25
    dense_top_k: int = 100         # Candidates fetched from dense retriever
    rrf_k: int = 60                # RRF smoothing constant (validated in Cormack 2009)
    fusion_top_n: int = 50         # Candidates passed to reranker after fusion

    # Reranking settings
    reranker_model: str = "ms-marco-MiniLM-L-6-v2"  # FlashRank model name
    reranker_top_k: int = 5        # Final results returned to caller

    # Embedding settings (for live dense retrieval)
    embedding_model: str = "text-embedding-3-small"

    # Runtime flags
    demo_mode: bool = True
    use_reranker: bool = True      # Set False to return RRF results directly


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "100")),
        dense_top_k=int(os.getenv("DENSE_TOP_K", "100")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        fusion_top_n=int(os.getenv("FUSION_TOP_N", "50")),
        reranker_top_k=int(os.getenv("RERANKER_TOP_K", "5")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        use_reranker=os.getenv("USE_RERANKER", "true").lower() == "true",
    )
