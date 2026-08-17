"""
config.py — W3D1 Prompt Distillation
=====================================
Configuration loaded exclusively from environment variables.

No values are hardcoded here. Copy .env.example to .env and fill in
your API key, or set DEMO_MODE=true to run without one.

Usage:
    from config import load_config
    cfg = load_config()
    print(cfg.demo_mode)   # True when no key is present
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Immutable configuration for the Prompt Distillation PoC.

    Fields map 1-to-1 with environment variables documented in .env.example.
    All fields have safe defaults so the PoC runs in demo mode out of the box.
    """

    # LLM credentials
    openai_api_key: str = ""

    # Model selection — gpt-4o-mini balances cost and quality for classification
    model: str = "gpt-4o-mini"

    # Sampling temperature — 0.0 for deterministic classification output
    temperature: float = 0.0

    # Maximum tokens for the LLM response (category label is <=10 tokens)
    max_tokens: int = 16

    # Distillation accuracy guard — student must retain >= this fraction
    # of teacher accuracy to be accepted as a valid compressed prompt
    accuracy_floor: float = 0.90

    # Maximum pruning iterations in the distillation loop
    max_distillation_iterations: int = 5

    # Daily call volume used for cost-savings projection
    daily_calls: int = 6_667  # ~200,000/month

    # Cost per 1M input tokens in USD (gpt-4o-mini, mid-2025 pricing)
    cost_per_1m_tokens: float = 0.15

    # When True: skip all API calls and return pre-computed demo output
    demo_mode: bool = True


def load_config() -> Config:
    """
    Build a Config instance from environment variables.

    Environment variables take precedence over dataclass defaults.
    demo_mode is True when OPENAI_API_KEY is absent OR DEMO_MODE=true.

    Returns:
        A fully populated Config instance.
    """
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    demo_mode: bool = (
        not api_key or os.getenv("DEMO_MODE", "false").lower() == "true"
    )

    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "16")),
        accuracy_floor=float(os.getenv("ACCURACY_FLOOR", "0.90")),
        max_distillation_iterations=int(
            os.getenv("MAX_DISTILLATION_ITERATIONS", "5")
        ),
        daily_calls=int(os.getenv("DAILY_CALLS", "6667")),
        cost_per_1m_tokens=float(os.getenv("COST_PER_1M_TOKENS", "0.15")),
        demo_mode=demo_mode,
    )
