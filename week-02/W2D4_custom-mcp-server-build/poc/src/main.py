#!/usr/bin/env python3
"""
W2D4 - Custom MCP Server Build
================================
Demonstrates: building a typed MCP server with tool discovery, input validation,
              structured errors, and a dispatcher pattern.
Run:           python src/main.py
Run (demo):    DEMO_MODE=true python src/main.py
"""

import asyncio
import json
import logging
import os
import sys

# Ensure src/ is importable when run from the project root
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from mcp_server_core import dispatch_tool_call, get_tool_definitions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demo mode - simulates the MCP request/response cycle without a live client
# ---------------------------------------------------------------------------
def run_demo() -> None:
    """
    Simulate the three phases of an MCP session:
      1. Tool discovery (tools/list)
      2. Successful tool call (search_documents)
      3. Structured error on invalid arguments (missing required field)
    """
    print("\nMCP Server Demo - document-search-server")
    print("=" * 48)

    # Phase 1: Tool discovery
    tools = get_tool_definitions()
    print(f"\n[tools/list] {len(tools)} tools registered:")
    for t in tools:
        required = t["inputSchema"].get("required", [])
        print(f"  - {t['name']}  required={required}")

    # Phase 2: Valid tool call
    print("\n[tools/call] search_documents - valid request")
    call_args = {"query": "revenue report", "department": "finance", "top_k": 2}
    print(f"  arguments: {json.dumps(call_args)}")
    result = asyncio.run(dispatch_tool_call("search_documents", call_args, demo_mode=True))
    print(f"  result:    {json.dumps(result, indent=4)}")

    # Phase 3: Structured error - missing required field 'department'
    print("\n[tools/call] search_documents - missing required field")
    bad_args = {"query": "budget forecast"}  # 'department' omitted intentionally
    print(f"  arguments: {json.dumps(bad_args)}")
    error_result = asyncio.run(dispatch_tool_call("search_documents", bad_args, demo_mode=True))
    print(f"  result:    {json.dumps(error_result, indent=4)}")

    # Phase 4: Unknown tool - JSON-RPC -32601
    print("\n[tools/call] unknown_tool - not registered")
    unknown_result = asyncio.run(dispatch_tool_call("unknown_tool", {}, demo_mode=True))
    print(f"  result:    {json.dumps(unknown_result, indent=4)}")

    print("\n[OK] Concept demonstrated: MCP server enforces typed contracts,")
    print("   returns structured errors instead of silent failures,")
    print("   and exposes a versioned tool registry agents discover at runtime.")


# ---------------------------------------------------------------------------
# Live mode - runs as a real MCP stdio server (requires mcp SDK)
# ---------------------------------------------------------------------------
def run_live() -> None:
    """
    Run a real MCP server over stdio transport.
    Requires: pip install mcp>=1.0.0
    Connect with any MCP-compatible client (Claude Desktop, LangChain MCP adapter, etc.)
    """
    try:
        from mcp.server import Server
        from mcp.server.models import InitializationOptions
        from mcp.server.stdio import stdio_server
        import mcp.types as types
    except ImportError:
        print("ERROR: 'mcp' package not installed. Run: pip install mcp>=1.0.0")
        print("       Or set DEMO_MODE=true to run without the package.")
        sys.exit(1)

    config = load_config()
    app = Server(config.server_name)

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in get_tool_definitions()
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        result = await dispatch_tool_call(name, arguments, demo_mode=False)
        return [types.TextContent(type="text", text=json.dumps(result))]

    async def serve() -> None:
        init_options = InitializationOptions(
            server_name=config.server_name,
            server_version=config.server_version,
            capabilities=app.get_capabilities(
                notification_options=None,
                experimental_capabilities={}
            ),
        )
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server '%s' started on stdio", config.server_name)
            await app.run(read_stream, write_stream, init_options)

    asyncio.run(serve())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    config = load_config()
    if config.demo_mode:
        print("Running in demo mode (set BEARER_TOKEN to enable live mode).")
        run_demo()
    else:
        run_live()


if __name__ == "__main__":
    main()
