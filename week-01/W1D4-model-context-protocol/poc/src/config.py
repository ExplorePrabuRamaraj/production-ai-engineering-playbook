"""
Configuration loaded from environment variables.
All secrets must be set in .env — never hardcode values here.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    demo_mode: bool = True
    # MCP-specific settings
    mcp_server_name: str = "demo-crm-server"
    mcp_protocol_version: str = "2024-11-05"
    mcp_transport: str = "stdio"          # "stdio" or "http"
    mcp_server_url: str = ""              # Required only when transport = "http"
    mcp_bearer_token: str = ""            # Auth token for HTTP+SSE servers


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
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        demo_mode=not api_key or os.getenv("DEMO_MODE", "false").lower() == "true",
        mcp_server_name=os.getenv("MCP_SERVER_NAME", "demo-crm-server"),
        mcp_protocol_version=os.getenv("MCP_PROTOCOL_VERSION", "2024-11-05"),
        mcp_transport=os.getenv("MCP_TRANSPORT", "stdio"),
        mcp_server_url=os.getenv("MCP_SERVER_URL", ""),
        mcp_bearer_token=os.getenv("MCP_BEARER_TOKEN", ""),
    )
