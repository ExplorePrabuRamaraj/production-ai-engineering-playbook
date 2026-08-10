"""Configuration loaded from environment variables.

W2D1 — Type-Safe Schemas with Pydantic AI
All secrets come from environment variables — never hardcoded.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "openai:gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    demo_mode: bool = True
    # Maximum retry attempts when schema validation fails
    schema_retries: int = 2


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "openai:gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        # Demo mode activates automatically when no API key is present
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        schema_retries=int(os.getenv("SCHEMA_RETRIES", "2")),
    )
