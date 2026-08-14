"""
Configuration loaded from environment variables.

All secrets come from the environment — never hardcoded.
Copy .env.example to .env and fill in your values before running live mode.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"   # Can use a cheaper model for critique
    temperature: float = 0.0
    max_tokens: int = 800
    max_iterations: int = 3             # Hard cap on reflection loop cycles
    demo_mode: bool = True
    confidence_threshold: float = 0.85  # Gate: skip loop if confidence >= this value


def load_config() -> Config:
    """
    Load configuration from environment variables with safe defaults.

    demo_mode activates automatically when OPENAI_API_KEY is absent,
    so the PoC always runs offline without explicit DEMO_MODE=true.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        critic_model=os.getenv("CRITIC_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "800")),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.85")),
    )
