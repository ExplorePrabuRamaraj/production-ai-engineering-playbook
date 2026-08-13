# W2D4 — Custom MCP Server Build

**Series:** AI Engineering Production Playbook
**Vertical:** MCP & Tool Integration
**Week 2 / Day 4**

## What This Demonstrates

A custom MCP (Model Context Protocol) server that exposes three typed tools to AI agents — with JSON Schema validation, structured error responses, and an audit-log dispatcher — so agents discover capabilities at runtime instead of relying on hardcoded function wrappers.

## Prerequisites

- Python 3.10+
- No API key required — demo mode is available offline

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/your-org/production-ai-engineering-playbook

# 2. Navigate to this day's folder
cd week-02/W2D4_custom-mcp-server-build/poc

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — leave BEARER_TOKEN blank to run in demo mode

# 5. Run
python src/main.py
```

## Demo Mode (No API Key or MCP Client Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
Running in demo mode (set BEARER_TOKEN to enable live mode).

MCP Server Demo — document-search-server
================================================
[tools/list] 3 tools registered:
  • search_documents  required=['query', 'department']
  • get_document  required=['doc_id']
  • list_departments  required=[]

[tools/call] search_documents — valid request
  arguments: {"query": "revenue report", "department": "finance", "top_k": 2}
  result:    {"results": [...], "count": 1, "department": "finance"}

[tools/call] search_documents — missing required field
  arguments: {"query": "budget forecast"}
  result:    {"error": true, "code": -32602, "message": "Field 'department' must be one of ..."}

[tools/call] unknown_tool — not registered
  result:    {"error": true, "code": -32601, "message": "Tool 'unknown_tool' is not registered ..."}

✅ Concept demonstrated: MCP server enforces typed contracts,
   returns structured errors instead of silent failures,
   and exposes a versioned tool registry agents discover at runtime.
```

## Run Tests

```bash
pytest tests/ -v
```

All tests run offline — no external services or API keys required.

## Live Mode (Real MCP stdio Server)

With `BEARER_TOKEN` set, the server starts as a real MCP stdio server compatible with Claude Desktop and MCP-enabled agent frameworks:

```bash
export BEARER_TOKEN=your-token-here
python src/main.py
```

Connect any MCP client (Claude Desktop, LangChain MCP adapter) and the three tools will be discoverable via `tools/list`.

## Project Structure

```
poc/
├── src/
│   ├── main.py              # Entry point — demo and live mode
│   ├── mcp_server_core.py   # Tool definitions, handlers, dispatcher
│   └── config.py            # Config dataclass loaded from env vars
├── tests/
│   └── test_mcp_server.py   # 15 unit tests — all run offline
├── README.md
├── requirements.txt
├── .env.example
├── sample_input.json        # Example MCP session request sequence
└── sample_output.json       # Expected responses for sample_input.json
```

## Key Concepts Demonstrated

- **Tool registry** — `get_tool_definitions()` returns JSON Schema descriptors the LLM uses at inference time to generate valid call arguments
- **Dispatcher pattern** — `dispatch_tool_call()` routes tool names to handlers without exposing handler internals to the client
- **Defense-in-depth validation** — handler-level validation runs after protocol-level JSON Schema validation; catches semantic errors the schema cannot express
- **Structured errors** — all failures return JSON-RPC error objects with machine-readable codes, not empty strings or exceptions
- **Demo/live split** — the same dispatcher works in both modes; only the transport layer and backend calls differ

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [MCP Specification](https://modelcontextprotocol.io/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
