"""
Configuration loaded from environment variables.
All secrets come from the environment — never hardcoded.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "w3d7-distributed-tracing"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    demo_mode: bool = True
    # Tracing is enabled only when both keys are present and demo_mode is False
    tracing_enabled: bool = False


def load_config() -> Config:
    """
    Load configuration from environment variables.
    Tracing requires both OPENAI_API_KEY and LANGSMITH_API_KEY to be set.
    If either is missing, demo mode activates automatically.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
    force_demo = os.getenv("DEMO_MODE", "false").lower() == "true"

    demo_mode = force_demo or not openai_key

    return Config(
        openai_api_key=openai_key,
        langsmith_api_key=langsmith_key,
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "w3d7-distributed-tracing"),
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        demo_mode=demo_mode,
        # Enable tracing only when LangSmith key is present and not in demo mode
        tracing_enabled=bool(langsmith_key) and not demo_mode,
    )
