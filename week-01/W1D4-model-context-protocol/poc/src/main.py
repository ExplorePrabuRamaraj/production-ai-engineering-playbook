#!/usr/bin/env python3
"""
W1D4 — Model Context Protocol (MCP) Intro
==========================================
Demonstrates: MCP session lifecycle — init, capability discovery, tool calls,
              structured error handling, and scoped tool sets.

Run (live mode):  python src/main.py
Run (demo mode):  DEMO_MODE=true python src/main.py
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env  # optional — demo mode runs without any API key
"""

import json
import os
import sys
from pathlib import Path

# Make src/ importable when running from the poc-code root
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from mcp_core import MCPServer, run_mcp_session

# ---------------------------------------------------------------------------
# Configuration — all secrets come from environment variables
# ---------------------------------------------------------------------------
config = load_config()
DEMO_MODE = config.demo_mode

# ---------------------------------------------------------------------------
# Demo mode — pre-computed output that mirrors a real MCP session
# ---------------------------------------------------------------------------

def run_demo() -> dict:
    """
    Simulate a complete MCP session using the in-process server.
    No API key or external service required — demonstrates all three MCP
    primitives (Resources, Tools, error handling) offline.
    """
    print("\n[DEMO] Running in DEMO MODE — in-process MCP server, no external calls\n")

    server = MCPServer(
        server_name=config.mcp_server_name,
        protocol_version=config.mcp_protocol_version,
    )

    # Three tool requests that exercise success, error, and write paths
    tool_requests = [
        # Success path: valid account lookup
        {"name": "lookup_account", "arguments": {"customer_id": "CUST-1001", "account_region": "us-east-1"}},
        # Error path: account not found — demonstrates isError:true handling
        {"name": "lookup_account", "arguments": {"customer_id": "CUST-9999", "account_region": "us-east-1"}},
        # Schema validation error: missing required field
        {"name": "lookup_account", "arguments": {"customer_id": "CUST-2042"}},
        # Write tool: update account status
        {"name": "update_account_status", "arguments": {
            "customer_id": "CUST-2042",
            "new_status": "active",
            "reason": "Payment received — account reinstated",
        }},
    ]

    session_result = run_mcp_session(server, tool_requests)
    return {
        "server_name": session_result.server_name,
        "tools_discovered": session_result.tools_discovered,
        "resources_discovered": session_result.resources_discovered,
        "tool_calls": session_result.tool_calls,
    }


# ---------------------------------------------------------------------------
# Live mode — same session logic, same server, same output shape
# (In production you would swap MCPServer for a real transport connection)
# ---------------------------------------------------------------------------

def run_live() -> dict:
    """
    Run the MCP session with a live in-process server.
    In a real deployment, replace MCPServer with a stdio_client or sse_client
    from the MCP Python SDK — the run_mcp_session interface is identical.
    """
    server = MCPServer(
        server_name=config.mcp_server_name,
        protocol_version=config.mcp_protocol_version,
    )

    # Load tool requests from sample_input.json
    sample_path = Path(__file__).parent.parent / "sample_input.json"
    if sample_path.exists():
        payload = json.loads(sample_path.read_text())
        tool_requests = payload.get("tool_requests", [])
    else:
        tool_requests = [
            {"name": "lookup_account", "arguments": {"customer_id": "CUST-3300", "account_region": "eu-west-1"}},
        ]

    session_result = run_mcp_session(server, tool_requests)
    return {
        "server_name": session_result.server_name,
        "tools_discovered": session_result.tools_discovered,
        "resources_discovered": session_result.resources_discovered,
        "tool_calls": session_result.tool_calls,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_session_report(result: dict) -> None:
    """Print the MCP session summary in a human-readable format."""
    print(f"Server:              {result['server_name']}")
    print(f"Tools discovered:    {result['tools_discovered']}")
    print(f"Resources discovered:{result['resources_discovered']}")
    print(f"\nTool call log ({len(result['tool_calls'])} calls):")
    print("-" * 55)
    for i, call in enumerate(result["tool_calls"], 1):
        status = "ERROR" if call["is_error"] else "OK"
        print(f"  [{i}] {call['tool']}  [{status}]  {call['latency_ms']} ms")
        print(f"      Args:   {json.dumps(call['arguments'])}")
        print(f"      Result: {call['result'][:120]}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n[MCP] Model Context Protocol (MCP) Intro Demo")
    print("=" * 55)

    result = run_demo() if DEMO_MODE else run_live()

    print_session_report(result)
    print("[OK] Concept demonstrated: MCP init -> capability discovery -> tool calls -> structured error handling")
    print("\n[doc] See 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
