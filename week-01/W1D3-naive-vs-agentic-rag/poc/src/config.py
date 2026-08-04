"""
Configuration loaded from environment variables.
W1D3 — Naive vs. Agentic RAG
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    # Smaller/faster model for decomposition and validation steps
    decomposition_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    demo_mode: bool = True
    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.70
    max_sub_questions: int = 4
    max_reformulation_retries: int = 2


def load_config() -> Config:
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        decomposition_model=os.getenv("DECOMPOSITION_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        # Demo mode activates when no API key is present or DEMO_MODE=true
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.70")),
        max_sub_questions=int(os.getenv("MAX_SUB_QUESTIONS", "4")),
        max_reformulation_retries=int(os.getenv("MAX_REFORMULATION_RETRIES", "2")),
    )
