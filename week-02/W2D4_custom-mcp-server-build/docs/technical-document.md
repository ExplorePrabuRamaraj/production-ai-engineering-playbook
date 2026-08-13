# W2D4 — Custom MCP Server Build
## Technical Deep-Dive: Model Context Protocol — Design, Implementation, and Production Deployment

---

## 1. Overview

The **Model Context Protocol (MCP)** is an open, JSON-RPC 2.0-based standard that defines how AI agents discover and invoke external tools through a uniform, versioned interface. Instead of embedding ad-hoc function wrappers into every agent prompt, MCP separates tool *definition* (server) from tool *invocation* (client), enabling agents to query available capabilities at runtime and call them with typed, validated arguments. Developed and open-sourced by Anthropic in late 2024, MCP has gained rapid adoption across agentic frameworks including Claude, LangChain, and AutoGen. Building a custom MCP server is the production-grade answer to the question: "how do I give my agent access to my internal systems without rewriting the integration layer every time the agent changes?"

---

## 2. Learning Objectives

By the end of this document, you will be able to:

1. **Explain** the MCP protocol architecture and how it differs from ad-hoc function calling
2. **Implement** a custom MCP server with tool registration, input validation, and structured output
3. **Distinguish** between the stdio and SSE transport layers and select the right one for a given deployment
4. **Design** a tool schema that minimises LLM misrouting and malformed-call errors
5. **Evaluate** security risks in MCP server deployments and apply OWASP LLM Top 10 mitigations
6. **Apply** versioning strategies to MCP tool schemas across breaking changes
7. **Build** a production-ready MCP server with authentication, rate limiting, and structured logging
8. **Benchmark** tool call latency and identify bottlenecks in the server-side execution path

---

## 3. Problem Statement

When an AI agent needs to interact with external systems — databases, APIs, internal services — the naive approach is to write Python functions, wrap them in tool-calling descriptors, and hardcode them into the agent's system prompt or framework configuration. This approach breaks in production for four specific reasons:

**Schema drift:** The tool's actual behaviour changes but the descriptor in the prompt does not. The agent generates call arguments that matched the old schema, causing 100% silent failure on the changed fields.

**No discoverability:** Every agent that needs the same tool must duplicate the descriptor. There is no single source of truth. When the tool changes, every agent that uses it must be updated manually.

**No auth boundary:** The tool function runs in the same process as the agent. There is no way to apply per-tool authentication, rate limiting, or audit logging without modifying every integration point.

**Versioning is impossible:** There is no mechanism to run tool-v1 for legacy agents while tool-v2 serves new ones. Schema breaking changes require simultaneous updates across all agent configs.

In production systems processing thousands of agent calls per day, these four failure modes compound. A single schema drift on a high-volume tool can generate thousands of failed calls before it is detected, each consuming tokens and increasing latency.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: Internal Knowledge Base Integration Collapses Under Schema Drift

A financial services company builds an agent that answers analyst questions by calling an internal document search API. The developer hardcodes a function descriptor in the agent's LangChain config: `search_documents(query: str, top_k: int)`. Six weeks later, the document search team adds a mandatory `department` filter parameter to comply with data access policies.

The agent continues calling the old schema. The API now returns a 400 error on every call — but the agent receives no structured error, just an empty result set. The agent hallucinates answers to fill the gap. Analysts receive fabricated citations. The failure takes three weeks to diagnose because there is no structured error surfaced through the ad-hoc wrapper layer.

Impact: 23% of analyst queries during the three-week window received hallucinated responses. Two incorrect regulatory filings were submitted before the issue was caught.

### Scenario B — The Solution: MCP Server Enforces Contract at the Protocol Layer

The same company rebuilds the document search integration as a custom MCP server. The server registers a `search_documents` tool with a JSON Schema that includes `department` as a required field. When the document search API changes its signature, the MCP server's tool schema is updated in one place.

Any agent that calls `search_documents` without `department` receives a structured MCP error: `{"error": {"code": -32602, "message": "Missing required field: department"}}`. The agent's error-handling loop catches this, surfaces it to the operator, and halts rather than hallucinating. The schema change is detected in the first call, not three weeks later.

Impact: Zero hallucinated responses. Schema changes now take 15 minutes to propagate across all six agents that use the tool, compared to days of manual config updates.

---

## 5. Solution Architecture

An MCP deployment has three layers: the **host** (the AI application or agent runtime), the **client** (the MCP client embedded in the host that speaks the protocol), and one or more **servers** (each owning a set of tools).

The host never calls tool logic directly. It queries the MCP server for available tools (`tools/list`), receives JSON Schema definitions for each tool, and passes those schemas to the LLM at inference time. When the LLM decides to call a tool, it generates a JSON arguments object that conforms to the tool's input schema. The host sends this as a `tools/call` request to the MCP server. The server validates the arguments, executes the underlying logic, and returns a structured result.

This separation means the LLM never executes code — it only generates JSON. All execution happens server-side, behind a validated interface.

```
Host (Agent Runtime)
  └── MCP Client
        ├── tools/list  ──►  MCP Server  ──►  Tool Registry
        └── tools/call  ──►  MCP Server  ──►  Dispatcher  ──►  Business Logic
```

See the architecture diagram in `diagrams/architecture.mmd`.

---

## 6. Internal Working Mechanics

### Protocol Layer: JSON-RPC 2.0

MCP messages are JSON-RPC 2.0 objects. Every request has a `method`, optional `params`, and an `id`. Every response has either a `result` or an `error`. The four core MCP methods are:

- `initialize` — handshake; client declares its capabilities, server responds with its own
- `tools/list` — server returns an array of tool definitions (name, description, inputSchema)
- `tools/call` — client invokes a named tool with a JSON arguments object
- `notifications/message` — server sends unsolicited log or progress messages

### Tool Definition Structure

Each tool definition contains:

```json
{
  "name": "search_documents",
  "description": "Search the internal document repository by semantic query.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Natural language search query" },
      "department": { "type": "string", "enum": ["finance", "legal", "engineering"] },
      "top_k": { "type": "integer", "default": 5, "minimum": 1, "maximum": 20 }
    },
    "required": ["query", "department"]
  }
}
```

The `description` field is what the LLM reads at inference time to decide whether to call this tool. Poor descriptions are the leading cause of incorrect tool selection.

### Dispatch and Validation

When a `tools/call` request arrives, the server:
1. Looks up the tool name in its registry — returns error code `-32601` (method not found) if absent
2. Validates the `arguments` object against the tool's `inputSchema` using a JSON Schema validator
3. Executes the registered handler function with the validated arguments
4. Serialises the result as a `content` array (text, image, or resource items)
5. Returns the response or a structured error

### Transport Layer

**stdio transport:** The server runs as a child process. The host communicates over stdin/stdout using newline-delimited JSON. Used for local development and CLI integrations. Never expose over a network socket.

**SSE (Server-Sent Events) transport:** The server runs as an HTTP service. The client connects to an SSE endpoint for server-to-client messages and POSTs to a separate endpoint for client-to-server requests. Supports authentication headers, TLS, and horizontal scaling. Required for any production deployment.

### Session Lifecycle

1. Client sends `initialize` with `protocolVersion` and `clientInfo`
2. Server responds with `serverInfo` and `capabilities`
3. Client sends `initialized` notification
4. Session is open — `tools/list` and `tools/call` requests flow
5. Either party sends `close` to end the session

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd`.

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd`.

---

## 9. Implementation Guide

### Step 1: Install dependencies

```bash
pip install mcp>=1.0.0 pydantic>=2.0.0 httpx>=0.27.0
```

### Step 2: Define the server and register tools

```python
# src/mcp_server_core.py
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types

app = Server("document-search-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_documents",
            description="Search the internal document repository by semantic query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "department": {"type": "string", "enum": ["finance", "legal", "engineering"]},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query", "department"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_documents":
        results = await search_backend(arguments["query"], arguments["department"])
        return [types.TextContent(type="text", text=str(results))]
    raise ValueError(f"Unknown tool: {name}")
```

### Step 3: Configure transport

```python
# src/main.py — stdio transport for local development
import asyncio
from mcp.server.stdio import stdio_server

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, InitializationOptions(...))

asyncio.run(main())
```

### Step 4: Run and verify

```bash
# Demo mode — no external API needed
DEMO_MODE=true python src/main.py

# Expected output:
# MCP Server "document-search-server" started on stdio
# Tools registered: search_documents, get_document, list_departments
# Ready to accept connections.
```

### Step 5: Add SSE transport for production

```python
# src/main.py — SSE transport for production
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette

sse = SseServerTransport("/messages")
starlette_app = Starlette(routes=[...])
```

---

## 10. Benefits and Trade-offs

| Benefit | Trade-off |
|---|---|
| Single source of truth for tool schemas | Adds a network hop for every tool call (typically 2–10 ms on LAN) |
| Schema validated before execution — no silent failures | Server must be running before agents can operate |
| Auth, rate limiting, and audit logging in one place | SSE transport adds operational complexity (service discovery, TLS certs) |
| Tool changes propagate to all agents without agent redeployment | Protocol versioning requires maintaining backward-compatible schemas |
| Enables horizontal scaling of tool execution separately from the LLM layer | stdio transport is limited to single-host deployments |

---

## 11. Performance Characteristics

**Latency:** A `tools/call` over stdio on the same host adds approximately 1–3 ms. Over SSE on a local network, expect 5–15 ms round-trip. For compute-heavy tools (database queries, file processing), the tool execution time dominates; MCP protocol overhead is negligible.

**Throughput:** A single MCP server instance handles concurrent tool calls through Python's asyncio event loop. CPU-bound tools should be offloaded with `asyncio.run_in_executor()` to avoid blocking the event loop. For >500 concurrent tool calls/second, deploy multiple server instances behind a load balancer.

**Memory:** The server process maintains an in-memory tool registry and active session state. A typical registry of 20 tools with session state for 100 concurrent connections requires approximately 50–100 MB RSS.

**Cold-start penalty:** stdio transport incurs a process spawn cost (100–300 ms on typical hardware). Cache the server process across agent runs rather than spawning per request.

---

## 12. Security Considerations

MCP servers are a direct attack surface for OWASP LLM Top 10 risks:

**LLM01 — Prompt Injection via Tool Description:** Attackers who can modify tool `description` fields can instruct the LLM to call unintended tools. Never allow tool descriptions to be set from untrusted user input. Treat tool definitions as code, not data.

**LLM02 — Insecure Output Handling:** Tool handlers that return raw LLM-generated content back into the agent loop without sanitisation create reflected injection. Validate and sanitise all tool output before returning it as `TextContent`.

**LLM06 — Excessive Agency:** MCP servers that expose broad filesystem, shell, or database tools without scoping create excessive agency. Apply the principle of least privilege: each tool should perform exactly one scoped operation.

**Input Validation:** Every tool handler must validate arguments against the declared JSON Schema before execution, even though the MCP SDK performs a first-pass validation. Defense in depth at the handler layer prevents schema-bypass attacks.

**Authentication (SSE transport):** Use bearer token or mTLS for all SSE connections. Never expose an unauthenticated MCP server on a public network. Validate tokens on every request, not just at session initialisation.

**Audit Logging:** Log every `tools/call` with: timestamp, tool name, caller identity, arguments (redacted for PII), execution duration, and result status. This is essential for incident investigation.

---

## 13. Cost Analysis

MCP itself is a protocol — it has no direct API cost. The cost model applies to what tools do:

**Token cost:** Tool definitions (name + description + inputSchema) are injected into the LLM context for every inference call. A registry of 10 tools with detailed schemas adds approximately 500–1,500 tokens per call. At GPT-4o pricing (~$2.50/1M input tokens), a system making 100k calls/day pays ~$0.25–$0.75/day purely for tool schema injection. Prune unused tools from the registry aggressively.

**Compute cost:** Tool execution cost depends entirely on the underlying operation. A tool wrapping a vector search costs ~$0.001/call (Pinecone pricing). A tool calling an external LLM for sub-tasks costs the full inference price of that call.

**Latency vs. cost trade-off:** Caching tool results (for read-only, deterministic tools) reduces both latency and downstream costs. Implement a TTL-based cache at the tool handler level for high-frequency read tools.

---

## 14. Best Practices

1. **One responsibility per tool.** A tool that does two things will be called incorrectly. If a tool fetches and also transforms data, split it into `fetch_record` and `transform_record`. The LLM selects tools based on descriptions — compound tools are described ambiguously.

2. **Write descriptions for the LLM, not for humans.** The `description` field is the LLM's routing signal. Be explicit about what the tool does, what it does NOT do, and when to prefer it over similar tools.

3. **Use JSON Schema `enum` for constrained fields.** If a parameter has a fixed set of valid values, declare them as an `enum`. This eliminates an entire class of malformed-call errors and guides the LLM toward valid inputs.

4. **Return structured errors, not empty results.** When a tool fails, return a JSON-RPC error with a meaningful message. Empty results cause the LLM to hallucinate — structured errors allow the agent's error-handling loop to act.

5. **Version your tool schemas.** When a breaking schema change is required, add a new tool (e.g., `search_documents_v2`) and deprecate the old one with a clear `description` note. Do not silently break existing agents.

6. **Validate at the handler layer, not just at the protocol layer.** The MCP SDK validates structure; your handler must validate semantics (e.g., date ranges, field interdependencies, business rules).

7. **Use SSE transport with TLS for all non-local deployments.** stdio transport over a network socket is not a supported or secure configuration.

8. **Limit tool registry size.** Every tool in the registry is injected into the LLM context. More than 20 tools in a single registry degrades routing accuracy. Use multiple specialised servers instead of one mega-server.

9. **Implement health and readiness endpoints.** Production agents must know if the MCP server is available before routing calls to it. A `/health` endpoint that verifies backend connectivity prevents silent degradation.

10. **Log all tool calls with structured fields.** Use JSON-structured logs with consistent fields (tool_name, caller_id, duration_ms, status) to enable alerting and dashboarding on tool call health.

---

## 15. Anti-Patterns

### The Mega-Server
**What it looks like:** A single MCP server that registers 30+ tools covering database access, file operations, email sending, and external API calls.
**Why it fails:** The LLM receives a 3,000-token tool registry on every call. Routing accuracy degrades significantly above 20 tools as similar-sounding tool names create ambiguity.
**Fix:** Split into domain-specific servers (data-server, comms-server, file-server). Each agent connects only to the servers it needs.

### The God Tool
**What it looks like:** A `run_query` tool that accepts a raw SQL string and executes it against the production database.
**Why it fails:** The LLM will generate SQL that drops tables, exposes PII, or runs full table scans. This is OWASP LLM06 (Excessive Agency) at its worst.
**Fix:** Replace with scoped tools: `search_customers(name, department)`, `get_order_history(order_id)`. Never pass raw query strings from the LLM to a database.

### Swallowed Errors
**What it looks like:** A tool handler that catches all exceptions and returns an empty string on failure.
**Why it fails:** The LLM sees an empty result and cannot distinguish between "no results found" and "tool crashed." It hallucinates to fill the gap.
**Fix:** Return a structured error response. The agent's error handling loop can then retry, escalate, or halt with a diagnostic message.

### Schema-Less Registration
**What it looks like:** A tool registered with `inputSchema: {}` (any object accepted) because the developer "will add the schema later."
**Why it fails:** Without a schema, the LLM cannot generate valid arguments reliably. Malformed arguments arrive at the handler with no upfront validation. "Later" never comes in production.
**Fix:** Define the full JSON Schema before registering any tool. Treat schema definition as part of the tool's specification, not an afterthought.

### stdio in Production
**What it looks like:** A stdio-transport MCP server deployed on a production host, with multiple remote agents connecting via SSH tunnels or reverse proxies.
**Why it fails:** stdio is designed for local subprocess communication. It has no authentication, no session isolation, and no horizontal scaling. One crashed process takes down all connected agents.
**Fix:** Use SSE transport with TLS and bearer token authentication for any multi-agent or remote deployment.

---

## 16. Common Mistakes

**Mistake 1: Using the same description for similar tools**
- Symptom: The LLM randomly alternates between two tools that should serve distinct purposes
- Root cause: Both tool descriptions say "retrieves information about X" without distinguishing their scope
- Fix: Rewrite descriptions to explicitly state the negative: "Use this for X. Do NOT use this for Y — use `tool_b` instead."

**Mistake 2: Forgetting to handle the `initialize` handshake in custom transports**
- Symptom: Client connects but never receives tool definitions; calls time out
- Root cause: Custom transport implementations skip the mandatory `initialize`/`initialized` exchange
- Fix: Always implement the full session lifecycle. Use the official MCP SDK rather than writing raw JSON-RPC transport code.

**Mistake 3: Blocking the asyncio event loop in tool handlers**
- Symptom: Server processes tool calls sequentially; latency spikes under concurrent load
- Root cause: Tool handlers call synchronous I/O (database drivers, file reads) directly in async functions
- Fix: Wrap synchronous calls with `await asyncio.run_in_executor(None, sync_function, args)` to move them to a thread pool.

---

## 17. Production Checklist

- [ ] All tools have complete JSON Schema definitions with `required` fields declared
- [ ] Tool descriptions clearly differentiate each tool's scope and exclusions
- [ ] Input validation implemented at the handler layer (not only at the protocol layer)
- [ ] SSE transport with TLS configured for all non-localhost deployments
- [ ] Bearer token or mTLS authentication enabled on the SSE endpoint
- [ ] All tool calls logged with: timestamp, tool name, caller ID, duration, status
- [ ] Structured error responses (not empty strings) returned on handler failure
- [ ] Health and readiness endpoints implemented and registered with service discovery
- [ ] Tool registry pruned to under 20 tools per server instance
- [ ] Domain-specific servers used instead of a single mega-server
- [ ] No raw SQL, shell commands, or file paths accepted from LLM-generated arguments
- [ ] PII fields identified and redacted from audit logs
- [ ] Schema versioning strategy documented and tested with backward-compatible changes
- [ ] asyncio event loop profiled — no blocking calls in tool handler async functions
- [ ] Load test performed at 2x expected peak concurrent call rate

---

## 18. References

[1] Anthropic (2024). "Model Context Protocol Specification." modelcontextprotocol.io. https://modelcontextprotocol.io/specification

[2] Anthropic (2024). "MCP Python SDK." GitHub. https://github.com/modelcontextprotocol/python-sdk

[3] OWASP (2023). "OWASP Top 10 for Large Language Model Applications." owasp.org. https://owasp.org/www-project-top-10-for-large-language-model-applications/

[4] JSON Schema (2020). "JSON Schema: A Media Type for Describing JSON Documents." json-schema.org. https://json-schema.org/specification

[5] Frostig, R. et al. (2024). "Tool Use and Function Calling in Large Language Models." arXiv:2402.07867. https://arxiv.org/abs/2402.07867

[6] FastMCP (2024). "FastMCP: The Fast, Pythonic Way to Build MCP Servers." GitHub. https://github.com/jlowin/fastmcp

---

## 19. Summary

Custom MCP servers solve the schema drift, discoverability, and auth-boundary failures that plague ad-hoc function-calling integrations in production AI systems. By defining tools as versioned, typed contracts in a server that agents discover at runtime, teams can update integrations without touching agent code, apply uniform security controls at the protocol layer, and get structured errors instead of silent hallucination cascades. The protocol overhead is negligible (2–15 ms per call); the operational complexity of SSE transport is real but manageable with standard service deployment patterns. The investment pays off at the first breaking API change in a tool your agents depend on.

---

## 20. Exercises

**Beginner:** Run the PoC in demo mode (`DEMO_MODE=true python src/main.py`) and observe the tool discovery and call flow in the terminal output. Modify `sample_input.json` to call a different registered tool.

**Intermediate:** Add a fourth tool to `mcp_server_core.py` — `get_document(doc_id: str)` — with a full JSON Schema definition. Observe how the tool list output changes. Add a unit test for the new tool handler.

**Advanced:** Replace the demo transport with a real SSE transport. Deploy the MCP server as a FastAPI service. Write an MCP client (using the `mcp` Python SDK) that connects to it, lists tools, and calls `search_documents`. Verify the structured error response when `department` is omitted.

**Expert:** Implement a tool schema versioning strategy. Register both `search_documents` (v1, no `department` field) and `search_documents` (v2, `department` required). Use the `description` field to route agents to the correct version. Benchmark the token cost difference between the two registries using `tiktoken`.

**Research:** Read the MCP specification at modelcontextprotocol.io/specification and identify one capability declared in the spec that is not yet implemented in the Python SDK. Describe what use case it would enable and what the implementation gap is.

---

## 21. Interview Questions

**Conceptual:**
1. Explain to a backend engineer who has never worked with AI agents why MCP exists. What specific problem does it solve that regular REST APIs do not?
2. What is the difference between a `tools/list` response and a `tools/call` request in MCP? At what point in the inference lifecycle does each occur?

**Technical:**
3. What happens when an MCP server receives a `tools/call` request for a tool name that is not registered? What JSON-RPC error code should it return and why?
4. A tool handler calls a synchronous database driver inside an `async def` function. What failure mode does this cause under load, and how do you fix it without replacing the database driver?

**Design:**
5. You need to give an agent access to 35 distinct internal tools across four business domains. How do you architect the MCP server layer to avoid routing accuracy degradation?
6. How would you design a schema versioning strategy for a `send_email` tool that needs to add a required `priority` field without breaking the 12 agents already using the v1 schema?

**Trade-off:**
7. When should you use stdio transport instead of SSE transport? What are the specific constraints that make each choice correct?
8. A product manager asks you to put all available company tools into a single MCP server so agents have "access to everything." What are the production risks of this approach and what do you propose instead?

**Debugging:**
9. An agent is calling the wrong tool 30% of the time — it calls `search_documents` when it should call `get_document`. You cannot change the agent's system prompt. What is your diagnostic process and what is the most likely fix?
10. An MCP server running over SSE works correctly in development but returns connection timeouts in production after exactly 30 seconds. What is the most likely cause, and what three things do you check first?
