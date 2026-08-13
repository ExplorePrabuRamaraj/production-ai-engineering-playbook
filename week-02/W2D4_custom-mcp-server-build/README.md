# W2D4 — Custom MCP Server Build

> [Week 2](../README.md) · [Playbook](../../README.md)

## Overview

Building a custom MCP (Model Context Protocol) server means implementing the full server contract: typed tool definitions with JSON Schema, a dispatcher that routes tool calls to handlers, defense-in-depth input validation, structured JSON-RPC error responses, and an audit-log for every call. This PoC builds a `document-search-server` with three typed tools that any MCP-compliant agent can discover and call at runtime — without hardcoded function wrappers.

**Vertical:** MCP & Tool Integration | **Week:** 2 | **Day:** 4

---

## Learning Objectives

1. Understand the full MCP server contract — `tools/list`, `tools/call`, and the JSON-RPC error envelope
2. Define typed tools using JSON Schema `inputSchema` with `required` fields, `enum` constraints, and range validation
3. Write description fields that act as routing signals for the LLM — guiding it to call the right tool with the right arguments
4. Implement a dispatcher pattern (`dispatch_tool_call`) that routes tool names to handlers without exposing handler internals to the client
5. Apply defense-in-depth validation: schema-layer validation catches structure errors; handler-layer validation catches semantic errors the schema cannot express
6. Return structured JSON-RPC error objects (`code`, `message`) instead of exceptions or empty strings so agents can recover gracefully
7. Emit structured audit logs on every tool call and understand what transport options (stdio vs. SSE) mean for deployment

---

## Problem Statement

Most teams wire AI agents to internal tools with hardcoded function calls: `if agent_says_search: call_search(args)`. This breaks in three ways. First, the agent has no way to discover what tools exist at runtime — capabilities are invisible until someone reads the code. Second, invalid arguments cause Python exceptions that crash the agent loop instead of returning a recoverable error. Third, there is no audit trail: when an agent calls the wrong tool with wrong arguments in production, there is nothing to debug from.

MCP solves all three problems: the `tools/list` endpoint makes capabilities self-describing, structured error responses let agents recover without crashing, and the dispatcher pattern makes every call auditable.

---

## Prerequisites

- Python 3.10+
- No API key required — demo mode runs fully offline
- [W1D4 — Model Context Protocol Intro](../../week-01/W1D4-model-context-protocol/README.md) recommended

---

## Repository Structure

```
W2D4_custom-mcp-server-build/
├── README.md                               # This file
├── docs/
│   ├── technical-document.md              # 21-section practitioner deep-dive
│   └── mcp-server-layman-scenarios.md     # Business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd                   # Server architecture (Mermaid)
│   └── sequence.mmd                       # Tool call sequence diagram (Mermaid)
└── poc/
    ├── README.md                          # PoC quickstart and expected output
    ├── src/
    │   ├── main.py                        # Entry point — demo and live modes
    │   ├── mcp_server_core.py             # Tool definitions, handlers, dispatcher, audit log
    │   └── config.py                      # Config dataclass loaded from environment
    ├── tests/
    │   └── test_mcp_server.py             # 15 unit tests, all offline-capable
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                  # Example MCP session request sequence
    └── sample_output.json                 # Expected responses for sample_input.json
```

---

## Core Concepts

### 1. Tool Definitions — JSON Schema as LLM Routing Signal

```python
from mcp_server_core import get_tool_definitions

tools = get_tool_definitions()
# Returns a list of tool dicts with name, description, and inputSchema.
# The 'description' field is the LLM's routing signal — it decides which
# tool to call based on this text, not on the function name.
```

Each tool's `inputSchema` is a JSON Schema object with `required` fields, `enum` constraints, and range validation. The `description` field must tell the LLM *when* to use this tool vs. alternatives:

```python
{
    "name": "search_documents",
    "description": (
        "Search the internal document repository by semantic query and department. "
        "Use this when the user needs to find documents, policies, or reports. "
        "Do NOT use this to retrieve a specific document by ID — use get_document instead."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "department": {"type": "string", "enum": ["finance", "legal", "engineering"]},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
        },
        "required": ["query", "department"]
    }
}
```

### 2. Dispatcher Pattern

```python
from mcp_server_core import dispatch_tool_call
import asyncio

result = asyncio.run(dispatch_tool_call(
    tool_name="search_documents",
    arguments={"query": "revenue report", "department": "finance", "top_k": 2},
    demo_mode=True,
))
# {"results": [...], "count": 1, "department": "finance"}
```

`dispatch_tool_call` never raises — it always returns a serialisable dict. Unknown tool names return a JSON-RPC `-32601` error; handler exceptions return `-32603`. This means the agent loop never crashes on a bad tool call.

### 3. Defense-in-Depth Validation

```python
from mcp_server_core import validate_search_arguments

error = validate_search_arguments({"query": "budget forecast"})
# "Field 'department' must be one of ['engineering', 'finance', 'legal'], got ''."
```

Two validation layers run on every call: the JSON Schema layer catches structural errors (missing required fields, wrong types), and the handler-level validator catches semantic errors (empty string queries, out-of-range integers) that JSON Schema cannot express. Both return machine-readable error messages the LLM can use to correct its arguments and retry.

---

## Run the PoC

**Demo mode (no API key or MCP client required):**

```bash
cd week-02/W2D4_custom-mcp-server-build/poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

**Live mode (real MCP stdio server — requires `mcp` package):**

```bash
cp .env.example .env
# Set BEARER_TOKEN in .env to trigger live mode
python src/main.py
# Connect any MCP-compatible client: Claude Desktop, LangChain MCP adapter, etc.
```

**Run tests:**

```bash
pytest tests/ -v
```

---

## Expected Output

```
Running in demo mode (set BEARER_TOKEN to enable live mode).

MCP Server Demo - document-search-server
================================================
[tools/list] 3 tools registered:
  - search_documents  required=['query', 'department']
  - get_document  required=['doc_id']
  - list_departments  required=[]

[tools/call] search_documents - valid request
  arguments: {"query": "revenue report", "department": "finance", "top_k": 2}
  result:    {
      "results": [{"doc_id": "DOC-001", "title": "Q4 Revenue Report", ...}],
      "count": 1,
      "department": "finance"
  }

[tools/call] search_documents - missing required field
  arguments: {"query": "budget forecast"}
  result:    {
      "error": true,
      "code": -32602,
      "message": "Field 'department' must be one of ['engineering', 'finance', 'legal'], got ''."
  }

[tools/call] unknown_tool - not registered
  result:    {
      "error": true,
      "code": -32601,
      "message": "Tool 'unknown_tool' is not registered on this server."
  }

[OK] Concept demonstrated: MCP server enforces typed contracts,
   returns structured errors instead of silent failures,
   and exposes a versioned tool registry agents discover at runtime.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — selects demo or live mode; demo simulates `tools/list` → `tools/call` → error cycle |
| `src/mcp_server_core.py` | `get_tool_definitions()`, `dispatch_tool_call()`, handler functions, `log_tool_call()` |
| `src/config.py` | `Config` dataclass + `load_config()` from environment variables |
| `tests/test_mcp_server.py` | 15 tests covering tool definitions, dispatcher routing, validation, error codes |
| `sample_input.json` | 4-step MCP session: `tools/list`, valid search, document lookup, error case |
| `sample_output.json` | Expected responses for the sample input session |

---

## Configuration

All parameters are loaded from environment variables (see `.env.example`):

| Variable | Default | Effect |
|---|---|---|
| `BEARER_TOKEN` | — | Set to enable live mode. Absent → demo mode activates automatically. |
| `MCP_TRANSPORT` | `stdio` | Transport layer: `stdio` (local subprocess) or `sse` (remote HTTP). |
| `SSE_HOST` | `0.0.0.0` | Host for SSE transport (only used when `MCP_TRANSPORT=sse`). |
| `SSE_PORT` | `8000` | Port for SSE transport. |
| `SERVER_NAME` | `document-search-server` | Server identity reported in `tools/list` and logs. |
| `SERVER_VERSION` | `1.0.0` | Server version string. |
| `DEMO_MODE` | `false` | Force demo mode regardless of `BEARER_TOKEN`. |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `TOOL_LOG_PATH` | `tool_calls.log` | Path for structured tool-call audit logs. |
| `MAX_RESULTS` | `20` | Hard cap on `top_k` to prevent expensive scans. |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section deep-dive on MCP server architecture, transport selection (stdio vs. SSE), JSON Schema tool definitions, dispatcher patterns, and production hardening
- [Layman Scenarios](docs/mcp-server-layman-scenarios.md) — Business scenarios explaining custom MCP servers without ML background

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — MCP server components: tool registry → dispatcher → handlers → backends, with transport and auth layers
- [Sequence Diagram](diagrams/sequence.mmd) — `tools/list` discovery → `tools/call` dispatch → validation → response → audit log

---

## Connection to the Series

| | Day | Topic |
|---|---|---|
| ← Previous | [W2D3 — GraphRAG & Knowledge Graphs](../W2D3_graphrag-knowledge-graphs/README.md) | Graph-enhanced retrieval |
| ← Foundation | [W1D4 — MCP Intro](../../week-01/W1D4-model-context-protocol/README.md) | What MCP is, connecting to existing servers |
| → Next | W2D5 — Reflection & Self-Correction Loops | Agent self-critique and rewrite |

**Why this follows W2D3:** W2D3 improved what the agent retrieves. W2D4 improves *how the agent exposes tools* — shifting from ad-hoc function wrappers to a self-describing, typed, auditable tool contract that any agent framework can consume.

---

## Key References

- MCP Specification: [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
- MCP Python SDK: [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
- JSON-RPC 2.0 error codes: [jsonrpc.org/specification](https://www.jsonrpc.org/specification)

---

## Continue Learning

**Next:** W2D5 — Reflection & Self-Correction Loops *(coming soon)*
