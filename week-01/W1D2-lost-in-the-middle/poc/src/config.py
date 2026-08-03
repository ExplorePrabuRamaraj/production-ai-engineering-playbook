"""Configuration loaded from environment variables."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    context_budget_tokens: int = 4096  # Max tokens to allocate for retrieved documents
    demo_mode: bool = True


def load_config() -> Config:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        context_budget_tokens=int(os.getenv("CONTEXT_BUDGET_TOKENS", "4096")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
    )
