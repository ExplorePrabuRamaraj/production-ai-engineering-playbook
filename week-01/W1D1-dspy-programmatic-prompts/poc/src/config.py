"""
Configuration for W1D1 — DSPy & Programmatic Prompts.
All values are loaded from environment variables.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    openai_api_key: str = field(default="")
    model: str = field(default="gpt-4o-mini")
    temperature: float = field(default=0.0)
    demo_mode: bool = field(default=False)

    def __post_init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("DSPY_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("DSPY_TEMPERATURE", "0.0"))
        demo_flag = os.getenv("DEMO_MODE", "false").lower() == "true"
        # Auto-enable demo mode when no API key is present
        self.demo_mode = demo_flag or not self.openai_api_key
