"""
Configuration for W3D5 — Dynamic Skill Selection PoC.
All values come from environment variables — nothing hardcoded.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_tokens: int = 500
    demo_mode: bool = True

    # Skill selection parameters
    top_k: int = 5                        # Max skills injected per turn
    similarity_threshold: float = 0.35    # Minimum cosine score to qualify
    eviction_after_turns: int = 50        # Turns of inactivity before eviction

    # Fallback skill names activated when no candidates pass threshold
    fallback_skills: List[str] = field(
        default_factory=lambda: ["general_response"]
    )


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
        eviction_after_turns=int(os.getenv("EVICTION_AFTER_TURNS", "50")),
    )
