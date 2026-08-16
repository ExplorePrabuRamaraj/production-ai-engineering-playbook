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
    # Guardrail policy paths
    blocked_input_patterns: List[str] = field(default_factory=list)
    required_output_tokens: List[str] = field(default_factory=list)
    blocked_output_vocab: List[str] = field(default_factory=list)


def load_config() -> Config:
    """
    Load configuration from environment variables.
    demo_mode is True when no API key is present or DEMO_MODE=true is set.
    This ensures the PoC runs safely offline for testing and demonstration.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    demo_mode = not api_key or os.getenv("DEMO_MODE", "false").lower() == "true"

    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        demo_mode=demo_mode,
        # Default blocked input patterns — covers common jailbreak preambles
        # and competitor-mention scenarios demonstrated in the PoC
        blocked_input_patterns=[
            r"ignore previous instructions",
            r"you are now",
            r"in a hypothetical scenario where",
            r"competitor[_\s]?bank",
            r"tell me about rival",
        ],
        # Required tokens that must appear in responses for flagged topics
        required_output_tokens=["[DISCLAIMER]"],
        # Vocabulary that must NOT appear in any outgoing response
        blocked_output_vocab=["you should buy", "i recommend purchasing", "guaranteed returns"],
    )
