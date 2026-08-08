"""
Configuration loaded from environment variables.
All secrets come from the environment — never hardcoded.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    openai_api_key: str = ""
    judge_model: str = "gpt-4o-mini"
    generator_model: str = "gpt-4o-mini"
    temperature: float = 0.0          # Deterministic judge output
    max_tokens: int = 500
    rubric_version: str = "v1.0"
    max_judge_retries: int = 2
    demo_mode: bool = True


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        generator_model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        rubric_version=os.getenv("RUBRIC_VERSION", "v1.0"),
        max_judge_retries=int(os.getenv("MAX_JUDGE_RETRIES", "2")),
        # Demo mode activates automatically when no API key is present
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
    )
