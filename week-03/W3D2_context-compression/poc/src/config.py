"""Configuration loaded from environment variables."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    # Model used for abstractive summarisation (cheaper than main model)
    compression_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    demo_mode: bool = True
    # Maximum tokens allowed in the compressed context (budget)
    token_budget: int = 1000
    # Compression strategy: "extractive", "abstractive", or "hybrid"
    compression_strategy: str = "extractive"
    # Minimum segment size in tokens before compression is applied
    min_segment_tokens: int = 50


def load_config() -> Config:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        compression_model=os.getenv("COMPRESSION_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        token_budget=int(os.getenv("TOKEN_BUDGET", "1000")),
        compression_strategy=os.getenv("COMPRESSION_STRATEGY", "extractive"),
        min_segment_tokens=int(os.getenv("MIN_SEGMENT_TOKENS", "50")),
    )
