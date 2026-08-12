"""
Configuration loaded from environment variables.

All secrets come from the environment — never hardcoded.
Copy .env.example to .env and fill in your values before running in live mode.
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

    # GraphRAG-specific settings
    hop_depth: int = 2               # Maximum graph traversal depth (hops)
    max_nodes_per_traversal: int = 30  # Hard cap on nodes retrieved per query
    min_edge_weight: int = 1         # Minimum edge occurrence count to include
    leiden_gamma: float = 1.0        # Community resolution: higher = smaller communities
    top_k_vector: int = 5            # Number of vector search results to retrieve
    top_k_merged: int = 8            # Number of merged results passed to LLM
    rrf_k: int = 60                  # RRF smoothing constant (default 60, per Cormack 2009)

    # Runtime flags
    demo_mode: bool = True


def load_config() -> Config:
    """
    Load configuration from environment variables with safe defaults.

    demo_mode activates automatically when OPENAI_API_KEY is absent,
    ensuring the PoC always runs without requiring credentials.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        hop_depth=int(os.getenv("HOP_DEPTH", "2")),
        max_nodes_per_traversal=int(os.getenv("MAX_NODES_PER_TRAVERSAL", "30")),
        min_edge_weight=int(os.getenv("MIN_EDGE_WEIGHT", "1")),
        leiden_gamma=float(os.getenv("LEIDEN_GAMMA", "1.0")),
        top_k_vector=int(os.getenv("TOP_K_VECTOR", "5")),
        top_k_merged=int(os.getenv("TOP_K_MERGED", "8")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
    )
