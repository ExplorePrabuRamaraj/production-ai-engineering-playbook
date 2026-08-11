"""
Configuration loaded from environment variables.
All secrets come from the environment — never hardcoded.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    demo_mode: bool = True
    # Token budget for the dynamic conversation suffix (excludes static prefix)
    max_context_tokens: int = 4000
    # Fraction of history eviction that triggers summary compression (0.0–1.0)
    compression_threshold: float = 0.5


def load_config() -> Config:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        # Demo mode activates automatically when no API key is present
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4000")),
        compression_threshold=float(os.getenv("COMPRESSION_THRESHOLD", "0.5")),
    )
