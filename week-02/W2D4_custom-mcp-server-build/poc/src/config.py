"""
Configuration loaded from environment variables.
W2D4 — Custom MCP Server Build
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Transport settings
    transport: str = "stdio"          # "stdio" or "sse"
    sse_host: str = "0.0.0.0"
    sse_port: int = 8000

    # Auth (SSE transport only)
    bearer_token: str = ""

    # Server identity
    server_name: str = "document-search-server"
    server_version: str = "1.0.0"

    # Demo / live mode
    demo_mode: bool = True

    # Logging
    log_level: str = "INFO"
    tool_log_path: str = "tool_calls.log"

    # Tool limits
    max_results: int = 20             # Hard cap on top_k to prevent expensive scans


def load_config() -> Config:
    api_key = os.getenv("BEARER_TOKEN", "")
    return Config(
        transport=os.getenv("MCP_TRANSPORT", "stdio"),
        sse_host=os.getenv("SSE_HOST", "0.0.0.0"),
        sse_port=int(os.getenv("SSE_PORT", "8000")),
        bearer_token=api_key,
        server_name=os.getenv("SERVER_NAME", "document-search-server"),
        server_version=os.getenv("SERVER_VERSION", "1.0.0"),
        demo_mode=(
            os.getenv("DEMO_MODE", "false").lower() == "true"
            or not api_key
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        tool_log_path=os.getenv("TOOL_LOG_PATH", "tool_calls.log"),
        max_results=int(os.getenv("MAX_RESULTS", "20")),
    )
