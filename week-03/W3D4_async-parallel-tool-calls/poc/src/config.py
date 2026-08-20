"""
Configuration loaded from environment variables.
All secrets come from the environment — never hardcoded.
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
    # Async dispatcher settings
    max_concurrent_tools: int = 5      # asyncio.Semaphore limit — tune per downstream API rate limits
    tool_timeout_s: float = 2.0        # Per-tool deadline in seconds
    tool_names: list = field(default_factory=lambda: [
        "get_product_price",
        "get_stock_status",
        "get_shipping_eta",
        "get_user_preferences",
    ])


def load_config() -> Config:
    """Load configuration from environment variables with safe defaults."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return Config(
        openai_api_key=api_key,
        model=os.getenv("MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
        # Demo mode activates when no API key is set OR explicitly requested
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        max_concurrent_tools=int(os.getenv("MAX_CONCURRENT_TOOLS", "5")),
        tool_timeout_s=float(os.getenv("TOOL_TIMEOUT_S", "2.0")),
    )
