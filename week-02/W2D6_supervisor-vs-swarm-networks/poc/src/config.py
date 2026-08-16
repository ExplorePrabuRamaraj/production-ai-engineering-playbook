"""Configuration loaded from environment variables.

W2D6 -- Supervisor vs. Swarm Networks
All runtime settings are read from env vars so no secrets are hardcoded.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    demo_mode: bool = True
    # Supervisor settings
    supervisor_timeout_seconds: float = 10.0
    max_subtask_retries: int = 2
    # Swarm settings
    swarm_max_hops: int = 5


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        supervisor_timeout_seconds=float(os.getenv("SUPERVISOR_TIMEOUT", "10.0")),
        max_subtask_retries=int(os.getenv("MAX_SUBTASK_RETRIES", "2")),
        swarm_max_hops=int(os.getenv("SWARM_MAX_HOPS", "5")),
    )
