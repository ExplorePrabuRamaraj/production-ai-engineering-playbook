# W1D4 — Model Context Protocol (MCP) Intro
## AI Engineering Production Playbook — Week 1, Day 4

**Vertical:** MCP & Tool Integration
**Series Position:** Day 4 of 28
**Prerequisites:** W1D3 (Naive vs. Agentic RAG), basic familiarity with JSON-RPC

---

## 1. Overview

Model Context Protocol (MCP) is an open protocol introduced by Anthropic in November 2024 that standardises how LLM-powered host applications connect to external data sources and tools. Before MCP, every agent framework invented its own tool-calling convention — meaning a tool written for LangChain could not be reused in a LlamaIndex pipeline without a rewrite. MCP solves this by defining a shared JSON-RPC 2.0 contract between a **host** (the LLM application), a **client** (the protocol implementation embedded in the host), and a **server** (the process that exposes resources and tools). The result is a plug-and-play ecosystem: any MCP-compliant server can be consumed by any MCP-compliant client without glue code. As of mid-2025, MCP has been adopted by major IDE vendors, cloud platforms, and open-source agent frameworks, making it the de facto standard for agent-to-tool integration.

---

## 2. Learning Objectives

By the end of this document you should be able to:

1. **Explain** the three MCP primitives (Resources, Tools, Prompts) and when each applies
2. **Distinguish** MCP from ad-hoc tool calling in LangChain or raw function calling in OpenAI
3. **Implement** a minimal MCP server in Python that exposes one resource and one tool
4. **Design** a production MCP deployment using HTTP+SSE transport with authentication
5. **Evaluate** the security trade-offs of exposing file system or database access via MCP
6. **Apply** capability negotiation to limit tool surface area per task type
7. **Build** a test harness for MCP servers that validates schema contracts without a live LLM
8. **Benchmark** the latency overhead introduced by MCP vs. direct function calls

---

## 3. Problem Statement

Every agent needs tools. Agents that can only reason without acting are limited to text generation; the moment you add web search, database access, code execution, or API calls, you have introduced a tool integration problem.

Before MCP, that problem was solved differently by every framework:

- **LangChain** tools require a Python class with a `run()` method and a manually-written description string that the framework injects into the system prompt.
- **OpenAI function calling** requires a JSON schema per function, defined by the developer, version-controlled separately from the actual function implementation, and re-uploaded whenever the API changes.
- **Custom agent wrappers** hard-code tool schemas in prompts, which break silently when upstream APIs evolve.

The compounding failure mode: when a team builds three internal tools, integrates with two external services, and then switches from GPT-4 to Claude — they rebuild every integration from scratch. There is no portability.

In production this translates to concrete costs: **tool schema drift** (the code changes but the prompt description does not) causes the model to call tools with wrong argument shapes. The failure is silent — the model gets a malformed response and either hallucinates recovery or produces a wrong answer. Teams at scale report that schema drift is responsible for 15–25% of agent reliability incidents (based on published post-mortems from LangChain and Anthropic engineering blogs).

The second failure mode is **trust boundary collapse**: without a standard protocol, developers bolt authentication onto each tool individually, some tools get auth and some do not, and a prompt-injection attack against one unprotected tool can cascade through the entire tool set.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: Fragile Bespoke Tool Integration

A fintech company builds a customer service agent powered by GPT-4o. The agent needs three tools: a CRM lookup (Salesforce API), a transaction history query (internal PostgreSQL), and a case escalation trigger (Zendesk API). Each tool is implemented as a LangChain `Tool` object with a manually-written description.

Six months later, Salesforce adds a new required parameter `account_region` to the contact lookup endpoint. The developer updates the Python function but forgets to update the tool description string in the prompt. The model continues calling the tool without `account_region`. The Salesforce SDK raises a 400 error on every call. The agent handles the error by fabricating a response — "I couldn't find any account details" — which is technically accurate but misleading.

Measured impact: the error goes undetected for 11 days. During that window, 340 support tickets are incorrectly routed because the agent could not verify account status. The root cause is not the API change — it is the absence of a contract between the tool definition and the tool implementation.

### Scenario B — The Solution: MCP-Standardised Tool Discovery

The same fintech company rebuilds the CRM integration as an MCP server. The server implements the `tools/list` endpoint, which returns the current tool schema at runtime. The MCP client calls `tools/list` before every agent session to fetch the live schema. When Salesforce adds `account_region`, the MCP server developer updates the schema in one place — the server's tool definition. The client fetches the updated schema on the next call without any prompt changes.

Additionally, the MCP server enforces input validation on the server side: if `account_region` is missing, the server returns a structured `MCP error` response with `code: -32602` (Invalid params) rather than a raw HTTP 400. The agent's error handler receives a typed error and can ask the user for clarification.

Measured improvement: schema drift incidents drop to zero in the three months following migration. Tool call success rate improves from 87% to 98.4%. Because the MCP server owns its own auth (Bearer token validated on the server), the prompt injection surface area decreases — the model never sees credentials directly.

---

## 5. Solution Architecture

MCP introduces a three-role model: **Host**, **Client**, and **Server**.

The **Host** is the application that contains or manages the LLM — Claude Desktop, an IDE plugin, or a custom agent runtime. The host creates and manages one or more **Clients**. Each Client maintains a 1:1 stateful connection to a single **Server**. The Server is a process (local or remote) that exposes capabilities: Resources (read-only context), Tools (executable functions), and Prompts (reusable prompt templates).

Communication uses **JSON-RPC 2.0** as the message format. Two transport options exist:

1. **stdio transport**: The host spawns the server as a subprocess and communicates via stdin/stdout. Ideal for local tools (file system, local databases, desktop apps). No network overhead. Simple process lifecycle management.
2. **HTTP + Server-Sent Events (SSE) transport**: The server runs as an HTTP service. The client sends requests via HTTP POST and receives streaming responses via SSE. Required for remote servers, multi-client deployments, and cloud-hosted tools.

The lifecycle of an MCP session has four phases:

1. **Initialisation**: Client sends `initialize` with its protocol version and capabilities. Server responds with its capabilities.
2. **Discovery**: Client calls `resources/list`, `tools/list`, and `prompts/list` to learn what the server offers.
3. **Operation**: Client makes requests — `resources/read`, `tools/call`, `prompts/get` — as the LLM needs them.
4. **Shutdown**: Client sends `shutdown` or closes the transport cleanly.

The architecture diagram below shows a production deployment with multiple MCP servers behind a host runtime.

---

## 6. Internal Working Mechanics

### JSON-RPC 2.0 Message Format

Every MCP message is a JSON-RPC 2.0 object. Requests include a method name and params; responses include a result or error object.

```json
// Tool call request
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tools/call",
  "params": {
    "name": "lookup_account",
    "arguments": {
      "customer_id": "CUST-8823",
      "account_region": "us-east-1"
    }
  }
}

// Tool call response
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"account_id\": \"CUST-8823\", \"status\": \"active\", \"tier\": \"gold\"}"
      }
    ],
    "isError": false
  }
}
```

### The Three Primitives in Detail

**Resources** are identified by a URI (e.g., `file:///data/config.json`, `db://customers/recent`). The client calls `resources/read` with the URI; the server returns the content. Resources are read-only from the model's perspective — the model requests context, it does not mutate state through a resource.

**Tools** are the mutation surface. A tool has a name, a description (used by the LLM to decide when to call it), and a JSON Schema defining its input parameters. The server validates inputs against the schema before execution. The LLM chooses which tool to call and what arguments to pass based solely on the description and schema — it never sees implementation code.

**Prompts** are server-defined prompt templates that the client can request by name. This allows a team to centralise prompt management: update the prompt on the MCP server and all connected clients pick it up automatically without redeployment.

### Capability Negotiation

During initialisation, both client and server declare which capability categories they support. A server that only provides read-only context might advertise `{resources: {}}` but not `{tools: {}}`. A client that does not support prompt templates ignores `{prompts: {}}` from the server. This prevents a client from accidentally calling a capability the server does not implement — a common source of silent failures in bespoke frameworks.

### Sampling (Server-Initiated LLM Calls)

MCP includes an optional `sampling` capability that allows a server to request an LLM completion from the host. This enables the server to enrich its responses with LLM reasoning — for example, a document-processing server that calls the host's LLM to summarise a retrieved file before returning it to the agent. This is a powerful but dangerous pattern: it creates recursive LLM calls that must be bounded.

### Error Handling

MCP defines a standard error code set extending JSON-RPC 2.0:
- `-32700`: Parse error (malformed JSON)
- `-32600`: Invalid request (missing required fields)
- `-32601`: Method not found (unknown MCP method)
- `-32602`: Invalid params (schema validation failure)
- `-32603`: Internal error (server-side execution failure)

Critically, tool execution errors (the tool ran but produced a bad result) are returned as `isError: true` in the result, not as a JSON-RPC error. This distinction matters: a JSON-RPC error means the protocol failed; `isError: true` means the tool ran but the operation did not succeed. The LLM can reason about `isError: true` responses; it cannot recover from protocol-level errors.

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd`.

```
artifacts/W1D4_mcp-intro/02_technical-doc/diagrams/architecture.mmd
```

The diagram shows: Host (LLM runtime + MCP clients) connecting to three MCP servers (File System Server via stdio, CRM Server via HTTP+SSE, Code Execution Server via stdio), with the LLM sitting inside the host and making tool/resource requests through the client layer.

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd`.

```
artifacts/W1D4_mcp-intro/02_technical-doc/diagrams/sequence.mmd
```

The diagram shows the full MCP session lifecycle: initialise → discover capabilities → LLM requests tool call → client routes to server → server validates and executes → structured response returned to LLM.

---

## 9. Implementation Guide

### Step 1: Install the MCP Python SDK

```bash
pip install mcp>=1.0.0
```

### Step 2: Define a minimal MCP server

```python
# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("crm-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lookup_account",
            description="Look up a customer account by ID. Returns account status and tier.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer account ID"},
                    "account_region": {"type": "string", "enum": ["us-east-1", "eu-west-1"]}
                },
                "required": ["customer_id", "account_region"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "lookup_account":
        # In production: call actual CRM API
        account = {"status": "active", "tier": "gold"}
        return [types.TextContent(type="text", text=str(account))]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
```

### Step 3: Connect a client

```python
# client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            result = await session.call_tool(
                "lookup_account",
                {"customer_id": "CUST-8823", "account_region": "us-east-1"}
            )
            print(f"Result: {result.content}")
```

### Step 4: Run the PoC

```bash
# Run the demo (no API key needed)
DEMO_MODE=true python src/main.py

# Run with real MCP SDK
python src/main.py

# Run tests
pytest tests/ -v
```

### Step 5: Verify the schema contract

The PoC in `03_poc-code/` demonstrates in-process MCP simulation: an `MCPServer` class exposes tools via `list_tools()` and `call_tool()`, and an `MCPClient` class calls them through the same interface a real MCP client would use. This isolates the protocol contract from transport details, making it testable offline.

---

## 10. Benefits & Trade-offs

| Benefit | Trade-off |
|---|---|
| Schema is server-owned — updates propagate automatically to all clients | Server must be running and reachable; adds operational complexity vs. inline function definitions |
| Protocol portability — any MCP-compliant client works with any MCP-compliant server | Requires all tool developers to learn and implement the MCP spec |
| Structured error codes enable agent-side error reasoning | JSON-RPC overhead adds 1–3 ms per call vs. direct in-process function calls |
| Capability negotiation prevents silent feature mismatches | Stateful sessions complicate horizontal scaling — session affinity required |
| Centralised auth on the server reduces prompt injection surface | Server-side validation is a new failure mode; misconfigured schemas cause tool call failures |

---

## 11. Performance Characteristics

**Latency overhead (stdio transport):** 1–3 ms per tool call for the JSON-RPC serialisation/deserialisation round trip. For tools that execute in < 50 ms, this is negligible. For high-frequency agents (> 10 tool calls/second), the overhead accumulates.

**Latency overhead (HTTP+SSE transport):** Adds network RTT to the above. On a local network, expect 5–15 ms. Over the internet, budget 50–200 ms per call depending on geography.

**Memory footprint:** The MCP Python SDK adds approximately 15–25 MB to the host process. Each server subprocess adds its own process overhead (40–80 MB for a typical Python MCP server).

**Throughput scaling:** stdio transport is inherently single-client per server process. For multi-agent scenarios, use HTTP+SSE transport behind a load balancer, or spawn per-agent server processes (heavy but isolated). The MCP spec does not mandate connection pooling — implement it at the transport layer.

**Cold start penalty:** For stdio servers, the host spawns a subprocess on first connection. Cold start time depends on server initialisation (Python import time: 200–800 ms for typical server packages). Mitigate with process pooling or keep-alive connections.

**References:** MCP specification performance notes (modelcontextprotocol.io/specification), Anthropic MCP announcement blog (anthropic.com/news/model-context-protocol, November 2024).

---

## 12. Security Considerations

MCP significantly improves the agent security posture compared to ad-hoc tool integration, but introduces its own attack surface.

**OWASP LLM Top 10 — LLM07: Insecure Plugin Design** is the primary applicable risk. MCP servers are effectively LLM plugins. The mitigation is schema-level input validation (the MCP server validates all inputs against the declared JSON Schema before execution) combined with least-privilege tool design (each tool does exactly one thing and returns exactly the data needed).

**OWASP LLM Top 10 — LLM01: Prompt Injection** remains relevant. A malicious resource (e.g., a file returned by `resources/read`) can contain instructions designed to manipulate the LLM's next action. Mitigate by treating all resource content as untrusted data: pass it to the LLM as quoted content, not as system prompt material.

**Authentication and authorisation:** The MCP spec does not mandate an auth mechanism — it is transport-dependent. For HTTP+SSE transport, use OAuth 2.0 Bearer tokens or mTLS. For stdio transport, process isolation is the auth boundary (only the host process can spawn the server).

**Tool scope creep:** Every tool exposed to the LLM is a potential execution path for prompt injection. Follow the principle of least privilege: expose only the tools required for the current task. Use MCP's capability negotiation to register different tool sets for different agent roles.

**Secrets management:** MCP servers frequently need credentials to call downstream APIs. Never pass credentials through MCP messages. Load secrets from environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault) at server startup, outside the MCP message flow.

**Sampling capability abuse:** If a server uses the `sampling` capability to request LLM completions from the host, it can potentially manipulate the LLM by crafting adversarial prompts in the sampling request. Disable sampling in servers that do not require it. Validate sampling requests against an allowlist of known-safe prompt patterns.

---

## 13. Cost Analysis

**Token cost impact:** MCP does not directly affect token cost — the LLM still receives tool descriptions and produces tool call arguments. The cost determinant is the number and verbosity of tool schemas injected into context. A server with 20 tools and verbose JSON Schema descriptions can add 800–1,500 tokens to every request. At $0.15/1M tokens (gpt-4o-mini), 1,000 requests/day adds $0.12–0.23/day — negligible unless your tool count grows unbounded.

**Mitigation:** Use MCP's capability negotiation to expose only task-relevant tools. Instead of loading all 20 tools for every request, register a `customer_support` tool set (5 tools) and a `billing` tool set (3 tools) and select the appropriate set per request. This reduces schema injection cost by 60–80%.

**Compute cost:** For stdio servers, each agent session spawns one subprocess per connected server. At scale (1,000 concurrent agents), 1,000 subprocesses per server type consume significant memory. Budget 40–80 MB per Python MCP server process × number of concurrent sessions.

**HTTP+SSE server cost:** A single MCP HTTP server handling 1,000 requests/second requires approximately 2–4 vCPUs and 2 GB RAM for a typical Python FastAPI implementation. At cloud rates ($0.048/vCPU-hour on AWS), this is ~$70–140/month for a production deployment.

---

## 14. Best Practices

1. **Own your schema on the server side.** Define tool JSON Schemas in the MCP server, not in the LLM's system prompt. The schema is the contract — it should live with the implementation.

2. **Validate all tool inputs on the server before execution.** Never trust the LLM's argument generation to be schema-compliant. Use `jsonschema` or Pydantic to validate before calling downstream APIs.

3. **Return structured errors, not raw exceptions.** Map all execution errors to MCP's `isError: true` result with a human-readable error message the LLM can reason about. Never let a raw stack trace reach the LLM.

4. **Use capability negotiation per task, not per session.** Build a tool registry that maps task types to tool subsets. A `read_only_query` task type should never receive write tools.

5. **Instrument every tool call with structured logging.** Log: tool name, input arguments (sanitised), execution time, result type (success/error), and the agent session ID. This is the foundation of agent observability.

6. **Pin the MCP protocol version in your initialisation handshake.** Specify `protocolVersion: "2024-11-05"` (or the current spec version) in your `initialize` request. Breaking changes in future versions will be version-gated.

7. **Test your MCP server without a live LLM.** Write tests that call `list_tools()` and `call_tool()` directly against the server implementation. The server should be fully testable without spinning up a model.

8. **For production HTTP+SSE servers, implement health checks.** Add a `/health` endpoint that verifies the server's downstream dependencies (database, external APIs) are reachable. The MCP host should check server health before routing agent requests.

9. **Limit resource URI scope.** If a server exposes file system resources, constrain the root path. Never allow `file:///` as a root — always scope to `file:///data/allowed-path/`. Validate all requested URIs against an allowlist.

10. **Document tool descriptions as if they are API docs.** The LLM chooses tools based solely on the description. Ambiguous descriptions cause wrong tool selections. Write descriptions that specify: what the tool does, what it does NOT do, and when to prefer it over similar tools.

---

## 15. Anti-Patterns

### The Tool Dump
**What it looks like:** Registering every available tool in every agent session — 30+ tools exposed regardless of the task.
**Why it fails:** The LLM's tool selection accuracy degrades as the tool count increases. Studies on function-calling models show accuracy drops 15–40% when tool count exceeds 20. More tools also means more tokens injected into context on every request.
**What to do instead:** Build a tool registry with named subsets. Select the minimal tool set per task type during session initialisation.

### The Chatty Server
**What it looks like:** Breaking a single logical operation into many small tool calls — e.g., separate tools for `get_account_id`, `get_account_status`, `get_account_tier` instead of one `lookup_account` tool.
**Why it fails:** Each tool call is a latency hit (LLM inference to generate arguments + network RTT + execution). Three calls instead of one triples the per-operation latency.
**What to do instead:** Design tools at the business operation level, not the data field level. Return complete objects the LLM needs.

### Schema in the Prompt
**What it looks like:** Describing tool parameters in the system prompt as freeform text in addition to (or instead of) the MCP tool schema.
**Why it fails:** The two descriptions drift independently. When they conflict, the LLM behaviour is undefined — it may follow the prompt description or the schema, inconsistently.
**What to do instead:** Remove all tool documentation from the system prompt. The MCP schema is the single source of truth.

### Unscoped Resource Access
**What it looks like:** Registering a file system resource server with `file:///` as the root, giving the LLM access to the entire file system.
**Why it fails:** A prompt injection attack embedded in any retrieved file can instruct the LLM to read `/etc/passwd` or write to configuration files.
**What to do instead:** Scope all resource URIs to a specific allowed directory. Validate every resource request against an allowlist before execution.

### Silent Tool Failures
**What it looks like:** Catching exceptions in tool implementations and returning empty strings or `null` instead of proper MCP error responses.
**Why it fails:** The LLM interprets an empty response as a valid (if empty) result, not as an error. It may proceed with incorrect assumptions rather than requesting clarification or retrying.
**What to do instead:** Return `isError: true` with a descriptive error message. The LLM can then reason about the failure and decide the appropriate recovery strategy.

### No Version Pinning
**What it looks like:** Not specifying `protocolVersion` in the `initialize` handshake, relying on the SDK's default.
**Why it fails:** When the MCP spec releases a breaking change, your client and server may negotiate different versions silently, causing subtle protocol errors.
**What to do instead:** Always specify the protocol version explicitly. Pin both client and server to the same version in your deployment manifests.

---

## 16. Common Mistakes

**Mistake 1: Returning tool results as raw JSON strings instead of structured content.**
- Symptom: The LLM says "I received a result but cannot parse it" or ignores tool output.
- Root cause: Tool returns `json.dumps(result)` as a plain string. The LLM receives escaped JSON inside a JSON string — double-encoded and unreadable.
- Fix: Return a `TextContent` object with the pre-formatted string, or return `structured` content with `type: "json"`. Let the MCP SDK handle serialisation.

**Mistake 2: Blocking the event loop in async MCP servers.**
- Symptom: Tool calls time out intermittently. Agent sessions stall after 10–30 seconds.
- Root cause: A `@app.call_tool()` handler performs a synchronous database query inside an `async` function using a blocking driver (e.g., `psycopg2` instead of `asyncpg`).
- Fix: Use `asyncio.to_thread()` to wrap synchronous calls, or switch to an async driver. The MCP server's event loop must not be blocked by tool execution.

**Mistake 3: Not handling `initialize` failures.**
- Symptom: The agent silently produces no tool calls. Tool calls work in development but not in production.
- Root cause: The MCP server fails to initialise (missing dependency, wrong port) but the client does not raise an error — it proceeds with an empty capability set and the LLM receives no tools.
- Fix: Assert that the `initialize` response contains at least the expected capability categories before proceeding. Treat capability negotiation failure as a fatal startup error.

---

## 17. Production Checklist

- [ ] MCP server exposes `list_tools()`, `list_resources()` and all tools/resources are documented
- [ ] All tool inputs validated with JSON Schema (server-side) before execution
- [ ] All tool errors returned as `isError: true` with human-readable messages (no raw exceptions)
- [ ] Tool set scoped per task type — not all tools exposed in every session
- [ ] MCP server has a health check endpoint verifying downstream dependencies
- [ ] Secrets loaded from environment variables or secrets manager — not in MCP message flow
- [ ] All resource URIs scoped to an allowlist — no unrestricted file system or database access
- [ ] Protocol version pinned in `initialize` handshake for both client and server
- [ ] Structured logging on every tool call: tool name, args (sanitised), duration, result type
- [ ] Unit tests call `list_tools()` and `call_tool()` without a live LLM (offline test suite)
- [ ] For HTTP+SSE: Bearer token auth enforced on all tool call endpoints
- [ ] For stdio: server process runs under a dedicated low-privilege service account
- [ ] Sampling capability disabled on any server that does not require it
- [ ] Tool description quality reviewed — descriptions specify what the tool does AND does not do
- [ ] Cold start time measured and documented — process pooling in place if > 500 ms

---

## 18. References

[1] Anthropic (2024). "Introducing the Model Context Protocol". Anthropic Blog. https://www.anthropic.com/news/model-context-protocol

[2] Model Context Protocol Specification (2024). "MCP Core Architecture". https://modelcontextprotocol.io/specification

[3] Model Context Protocol Python SDK (2024). GitHub repository. https://github.com/modelcontextprotocol/python-sdk

[4] OWASP (2025). "OWASP Top 10 for LLM Applications 2025". https://owasp.org/www-project-top-10-for-large-language-model-applications/

[5] Anthropic (2024). "MCP: An open standard for connecting AI assistants to the systems where data lives". GitHub. https://github.com/modelcontextprotocol

[6] JSON-RPC Working Group (2013). "JSON-RPC 2.0 Specification". https://www.jsonrpc.org/specification

[7] LangChain Blog (2024). "Building production agents: lessons from tool integration at scale". https://blog.langchain.dev

---

## 19. Summary

MCP solves the most under-appreciated problem in agent engineering: the absence of a standard contract between an LLM host and the tools it uses. Without this contract, every team reinvents tool integration, schema drift accumulates silently, and switching models or frameworks means rebuilding from scratch. By defining three primitives — Resources, Tools, and Prompts — over JSON-RPC 2.0, MCP gives both tool producers and tool consumers a shared language. The result is composability: an MCP server written today works with any MCP client written tomorrow, regardless of the LLM provider. The protocol is deliberately minimal — it does not mandate a deployment model, an auth mechanism, or an orchestration layer — which means it fits into existing architectures without imposing a new framework. The production considerations are real: tool scope control, structured error handling, and capability negotiation are not optional — they are what separates a reliable agent from a fragile demo.

---

## 20. Exercises

**Beginner:** Run `DEMO_MODE=true python src/main.py` and examine the output. Identify which three primitives are demonstrated. Add a second tool to the demo server with a different input schema.

**Intermediate:** Modify `mcp_core.py` to add input validation using `jsonschema`. Call the tool with an invalid argument and observe how the structured error response is formatted. Verify the LLM-facing error message is human-readable.

**Advanced:** Implement a minimal HTTP+SSE MCP server using `FastAPI` and the MCP Python SDK's `sse_server` transport. Connect the existing `MCPClient` to the HTTP server instead of the stdio simulation. Measure the additional latency introduced by the HTTP transport.

**Expert:** Build a tool registry that maps three task types (`read_query`, `write_operation`, `admin`) to different tool subsets. Implement a session initialisation flow that selects the appropriate tool set based on a `task_type` parameter. Benchmark tool call accuracy (using a simple test LLM or mock) with full tool exposure vs. scoped tool exposure.

**Research:** Read the MCP specification at modelcontextprotocol.io/specification, specifically the `sampling` capability section. Identify one security risk in server-initiated sampling that is not covered in Section 12 of this document. Propose a mitigation and write a test that would detect the vulnerability.

---

## 21. Interview Questions

1. **Conceptual:** Explain the difference between an MCP Resource and an MCP Tool to a product manager who has never worked with LLMs.

2. **Technical:** What is the difference between a JSON-RPC protocol error (e.g., code `-32602`) and a tool execution error (`isError: true` in the result)? Why does the distinction matter for agent error recovery?

3. **Design:** You are building an agent that needs access to 50 different internal tools across finance, HR, and IT systems. How would you use MCP's capability negotiation to prevent the LLM from being overwhelmed by irrelevant tools? Describe the architecture.

4. **Trade-off:** When would you choose stdio transport over HTTP+SSE transport for an MCP server? Name two scenarios where each is the correct choice and explain the deciding factor.

5. **Debugging:** An agent is making tool calls that consistently fail with `isError: true` and the message "Invalid params: missing required field 'account_region'". The developer swears the code passes `account_region`. What are three possible root causes you would investigate first?

6. **Security:** A red team finds that by placing a specific phrase in a document processed by your MCP file-reading tool, they can cause your agent to call a privileged write tool it should not have access to. Which OWASP LLM Top 10 category does this fall under? What MCP-level controls would you add?

7. **Architecture:** Your MCP server processes 5,000 tool calls per second at peak load. The server is written in Python and uses stdio transport. What are the bottlenecks, and how would you re-architect for this scale?

8. **Implementation:** You need to add authentication to an existing MCP server that uses HTTP+SSE transport. The server currently has no auth. Walk through the changes you would make to both the server and the client, and what happens when an unauthenticated request arrives.

9. **Conceptual:** MCP includes a `Prompts` primitive that lets servers expose prompt templates. In what production scenario would storing prompts on an MCP server be preferable to storing them in the application code?

10. **Trade-off:** A colleague argues that MCP adds unnecessary complexity — "just use OpenAI function calling, it works fine." Describe two production failure modes that function calling cannot prevent but MCP addresses, and one scenario where function calling is genuinely the better choice.
