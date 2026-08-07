"""
Configuration loaded from environment variables.
All secrets come from the environment — never hardcoded.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    demo_mode: bool = True
    # Risk threshold above which a document routes to the high-risk path
    risk_threshold: float = 0.7
    # Maximum retry attempts before routing to error terminal
    max_retries: int = 3
    # LangSmith tracing (optional — set LANGCHAIN_API_KEY to enable)
    langsmith_api_key: str = ""
    langsmith_project: str = "w1d6-state-graphs-langgraph"


def load_config() -> Config:
    """
    Load configuration from environment variables.
    demo_mode activates automatically when OPENAI_API_KEY is absent,
    so the PoC runs offline without any setup.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        risk_threshold=float(os.getenv("RISK_THRESHOLD", "0.7")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        langsmith_api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        langsmith_project=os.getenv("LANGCHAIN_PROJECT", "w1d6-state-graphs-langgraph"),
    )
