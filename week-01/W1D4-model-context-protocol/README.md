# W1D4 — Model Context Protocol (MCP) Intro

> Week 1, Day 4 | Vertical: MCP & Tool Integration  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 1](../README.md)

---

## Overview

Every agent needs tools. Before MCP, every agent framework invented its own tool-calling convention — meaning a tool written for LangChain could not be reused in a LlamaIndex pipeline without a rewrite. **Model Context Protocol (MCP)** solves this by defining a shared JSON-RPC 2.0 contract between a host (the LLM application), a client (the protocol implementation), and a server (the process that exposes resources and tools).

The result is a plug-and-play ecosystem: any MCP-compliant server works with any MCP-compliant client without glue code. As of mid-2025, MCP has been adopted by major IDE vendors, cloud platforms, and open-source agent frameworks, making it the de facto standard for agent-to-tool integration.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Explain the three MCP primitives (Resources, Tools, Prompts) and when each applies
2. Distinguish MCP from ad-hoc tool calling in LangChain or raw function calling in OpenAI
3. Implement a minimal MCP server in Python that exposes one resource and one tool
4. Design a production MCP deployment using HTTP+SSE transport with authentication
5. Evaluate the security trade-offs of exposing file system or database access via MCP
6. Apply capability negotiation to limit tool surface area per task type
7. Build a test harness for MCP servers that validates schema contracts without a live LLM

---

## Problem Statement

Before MCP, the tool integration problem was solved differently by every framework — LangChain tools require a custom `run()` method with a manually-written description; OpenAI function calling requires JSON schemas version-controlled separately from the implementation; custom wrappers hard-code schemas in prompts that break silently when upstream APIs evolve.

The compounding failure mode: when a team builds internal tools and then switches from GPT-4 to Claude, they rebuild every integration from scratch. In production, **tool schema drift** — the code changes but the prompt description does not — causes silent agent failures. Published post-mortems report schema drift as responsible for 15–25% of agent reliability incidents.

**MCP formalises the contract.** The server owns its schema; clients fetch the live schema at session start. When an API changes, the developer updates the MCP server schema in one place and every connected agent picks it up on the next session.

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode
- For live mode: an OpenAI API key set in `.env`

---

## Repository Structure

```
W1D4-model-context-protocol/
├── README.md                       # This file
├── docs/
│   ├── technical-document.md       # Full practitioner deep-dive (21 sections)
│   └── mcp-layman-scenarios.md     # Four business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd            # Host → Client → Server topology (Mermaid)
│   └── sequence.mmd                # Full MCP session lifecycle (Mermaid)
└── poc/
    ├── README.md                   # Quick-start and expected output
    ├── src/
    │   ├── main.py                 # Entry point — demonstrates full MCP session lifecycle
    │   ├── mcp_core.py             # MCPServer, MCPClient, run_mcp_session (core protocol)
    │   └── config.py               # Config dataclass + env loader
    ├── tests/
    │   └── test_mcp_core.py        # 20+ pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json           # Example tool request payload
    └── sample_output.json          # Expected session output
```

---

## Core Concept: The Three Primitives

MCP defines three things an MCP server can expose:

| Primitive | Analogy | What It Does |
|---|---|---|
| **Resource** | A read-only dashboard | Provides context the LLM can read — files, configs, DB query results |
| **Tool** | A button the LLM can press | Lets the LLM take an action — call an API, query a DB, write a file |
| **Prompt** | A form template | A server-managed prompt template the client can fetch and use |

### Session Lifecycle

```
initialize → discover (tools/list, resources/list) → operate (tools/call, resources/read) → shutdown
```

### Why Capability Negotiation Matters

During initialisation, both client and server declare which capability categories they support. A server that only provides read-only context advertises `{resources: {}}` but not `{tools: {}}`. This prevents clients from calling capabilities the server does not implement — a common source of silent failures in bespoke frameworks.

### Tool Call with Structured Error Response

```
Client → tools/call(lookup_account, {customer_id, account_region})
Server → {content: [...], isError: false}   # success
Server → {content: [...], isError: true}    # execution error (tool ran, operation failed)
```

Note: `isError: true` (tool ran but failed) is distinct from a JSON-RPC error (protocol failed). The LLM can reason about `isError: true`; it cannot recover from protocol-level errors.

---

## Run the PoC

### Demo Mode (No API Key Required)

```bash
cd poc
pip install -r requirements.txt
python src/main.py
```

Or with an explicit flag:

```bash
DEMO_MODE=true python src/main.py
```

### Live Mode (Requires OpenAI API Key)

```bash
cd poc
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python src/main.py
```

### Run Tests

```bash
cd poc
pytest tests/ -v
# Expected: 20+ passed, 0 failed (no API key needed)
```

---

## Expected Output

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

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — runs a complete MCP session demonstrating all four lifecycle phases |
| `src/mcp_core.py` | `MCPServer` (tool/resource registry + schema validation), `MCPClient` (protocol client), `run_mcp_session()` (full lifecycle) |
| `src/config.py` | `load_config()` — reads env vars including `DEMO_MODE`, `MODEL`, `OPENAI_API_KEY` |
| `tests/test_mcp_core.py` | 20+ tests across `TestDemoMode`, `TestCoreConcept`, `TestLiveMode`, `TestSampleFiles` |
| `sample_input.json` | Example tool request payload |
| `sample_output.json` | Expected session output with all four tool calls |

---

## Configuration

Copy `.env.example` to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL=gpt-4o-mini
TEMPERATURE=0.0
MAX_TOKENS=1000
DEMO_MODE=false   # Set to true to run without an API key
```

Demo mode activates automatically when no `OPENAI_API_KEY` is present.

---

## Technical Documentation

The full practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- JSON-RPC 2.0 message format and the three MCP primitives in detail
- Production security considerations (OWASP LLM07: Insecure Plugin Design, prompt injection mitigations)
- Performance characteristics for stdio and HTTP+SSE transport
- Cost analysis: token overhead of tool schemas at scale
- 10 best practices, 6 anti-patterns, 3 common mistakes
- Production checklist (15 items)
- 21 interview questions ranging from conceptual to architecture

For a jargon-free walkthrough, see [`docs/mcp-layman-scenarios.md`](docs/mcp-layman-scenarios.md).

---

## Architecture Diagrams

Mermaid source files are in [`diagrams/`](diagrams/):

- [`architecture.mmd`](diagrams/architecture.mmd) — Host runtime with embedded LLM connecting to multiple MCP servers via stdio and HTTP+SSE transports
- [`sequence.mmd`](diagrams/sequence.mmd) — Full session lifecycle: initialize → discover → tool call → structured response → shutdown

---

## Connection to the Series

- **W1D1 — DSPy & Programmatic Prompts:** Typed, compiled prompt programs replace hand-crafted strings.
- **W1D2 — Lost in the Middle:** Position-aware context assembly ensures retrieved documents land where attention peaks.
- **W1D3 — Naive vs. Agentic RAG:** Iterative retrieval replaces static context assembly for multi-hop queries.
- **Today — W1D4 Model Context Protocol:** With agentic retrieval in place, MCP formalises how the agent discovers and invokes external tools with a typed, versioned, portable contract.
- **Next — W1D5 Episodic vs. Semantic Memory:** Once tools are standardised, durable memory patterns enable agent continuity across sessions.

---

## Key References

- Anthropic (2024). "Introducing the Model Context Protocol". https://www.anthropic.com/news/model-context-protocol
- MCP Specification (2024). "MCP Core Architecture". https://modelcontextprotocol.io/specification
- MCP Python SDK. https://github.com/modelcontextprotocol/python-sdk
- OWASP (2025). "OWASP Top 10 for LLM Applications 2025". https://owasp.org/www-project-top-10-for-large-language-model-applications/

---

## Continue Learning

**Next:** W1D5 — Episodic vs. Semantic Memory — How agents maintain state across sessions using durable memory patterns.

**Series index:** [Week 1 Overview](../README.md) | [Full Roadmap](../../README.md)
