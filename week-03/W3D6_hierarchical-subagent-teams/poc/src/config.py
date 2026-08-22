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
    # Hierarchical team settings
    worker_timeout_seconds: int = 30
    worker_max_retries: int = 2
    max_parallel_leads: int = 5


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        # Demo mode activates automatically when no API key is present
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        worker_timeout_seconds=int(os.getenv("WORKER_TIMEOUT_SECONDS", "30")),
        worker_max_retries=int(os.getenv("WORKER_MAX_RETRIES", "2")),
        max_parallel_leads=int(os.getenv("MAX_PARALLEL_LEADS", "5")),
    )
