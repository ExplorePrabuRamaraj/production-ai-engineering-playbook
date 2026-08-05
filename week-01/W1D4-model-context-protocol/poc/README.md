# W1D4 — Model Context Protocol (MCP) Intro

**Series:** AI Engineering Production Playbook
**Vertical:** MCP & Tool Integration
**Week 1 / Day 4**

## What This Demonstrates

A complete MCP session lifecycle — protocol initialisation, capability discovery
(tools and resources), tool calls with structured success and error responses,
and schema-validated input handling — all running in-process without any external
service or API key.

The `MCPServer` and `MCPClient` classes mirror the public interface of the
official MCP Python SDK (`modelcontextprotocol/python-sdk`), so the patterns here
transfer directly to production deployments by swapping the in-process server for
a real stdio or HTTP+SSE transport.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-01/W1D4-model-context-protocol/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — skip for demo mode)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY if you want live mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
🚀 Model Context Protocol (MCP) Intro Demo
=======================================================
⚠️  Running in DEMO MODE — in-process MCP server, no external calls

Server:               demo-crm-server
Tools discovered:     ['lookup_account', 'update_account_status']
Resources discovered: ['config://crm/regions']

Tool call log (4 calls):
-------------------------------------------------------
  [1] lookup_account  [OK]  0.14 ms
      Args:   {"customer_id": "CUST-1001", "account_region": "us-east-1"}
      Result: {"customer_id": "CUST-1001", "status": "active", "tier": "gold", ...}

  [2] lookup_account  [ERROR]  0.09 ms
      Args:   {"customer_id": "CUST-9999", "account_region": "us-east-1"}
      Result: Account 'CUST-9999' not found in region 'us-east-1'

  [3] lookup_account  [ERROR]  0.07 ms
      Args:   {"customer_id": "CUST-2042"}
      Result: Invalid params: missing required field(s): ['account_region']

  [4] update_account_status  [OK]  0.11 ms
      Args:   {"customer_id": "CUST-2042", "new_status": "active", ...}
      Result: {"customer_id": "CUST-2042", "previous_status": "suspended", ...}

✅ Concept demonstrated: MCP init → capability discovery → tool calls → structured error handling
```

## Run Tests

```bash
pytest tests/ -v
```

All 20+ tests pass offline — no API key, no network access required.

## File Structure

```
03_poc-code/
├── src/
│   ├── main.py          # Entry point — demonstrates full MCP session
│   ├── mcp_core.py      # MCPServer, MCPClient, run_mcp_session (core protocol logic)
│   └── config.py        # Config dataclass + env loader
├── tests/
│   └── test_mcp_core.py # pytest unit tests (TestDemoMode, TestCoreConcept,
│                        #   TestLiveMode, TestSampleFiles)
├── README.md
├── requirements.txt
├── .env.example
├── sample_input.json    # Example tool request payload
└── sample_output.json   # Expected session output
```

## Key Concepts Demonstrated

| MCP Primitive | Demo File | What It Shows |
|---|---|---|
| **Tool** | `mcp_core.py` → `MCPServer._tools` | Schema-validated callable with `isError` responses |
| **Resource** | `mcp_core.py` → `MCPServer._resources` | Read-only context item returned by URI |
| **Capability negotiation** | `MCPClient.initialize()` | Server declares capabilities at session start |
| **Structured errors** | `ToolCallResult(is_error=True)` | Protocol-level vs execution-level error distinction |
| **Session lifecycle** | `run_mcp_session()` | init → discover → call → shutdown in one function |

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Layman Scenarios](../docs/mcp-layman-scenarios.md)
- [Day README](../README.md)

## References

- MCP Specification: https://modelcontextprotocol.io/specification
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Anthropic MCP announcement: https://www.anthropic.com/news/model-context-protocol
