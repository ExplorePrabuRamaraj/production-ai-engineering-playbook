# Custom MCP Server Build in Simple Words — Real-World QA Scenarios

A plain-English guide to understanding Model Context Protocol through four concrete situations — no AI background required.

---

## Core Idea

Imagine you run a large hotel. Guests (AI agents) constantly ask the concierge (the LLM) for things: "Book me a taxi," "Order room service," "Get my dry cleaning." The concierge cannot do any of these directly — they need to call the right department. The old way was for each concierge to have a personal notebook with phone numbers scrawled in different handwriting. When the taxi company changed its number, half the notebooks were wrong.

**Model Context Protocol (MCP)** is the hotel's official switchboard directory. Every department (tool) registers its name, what it does, and exactly what information it needs to do it. The concierge looks up the directory at the start of every shift. When a department changes, it updates the directory — and every concierge on every floor immediately has the correct information.

The three key ideas are:

| Concept | Hotel Analogy | Technical Reality |
|---|---|---|
| Tool definition | Department listing in the switchboard directory | JSON Schema declaring the tool's name, inputs, and outputs |
| Tool discovery | Concierge reads the directory at shift start | Agent calls `tools/list` at session start to get current schemas |
| Tool call | Concierge dials the correct department with the right information | Agent sends a `tools/call` request with validated JSON arguments |
| Structured error | "Sorry, I need a room number to connect you" | JSON-RPC error with a machine-readable code and human-readable message |
| Transport layer | Phone line (local internal phone vs. external public line) | stdio for local processes, SSE over HTTPS for remote services |

The critical insight is that the directory (MCP server) is the single source of truth. Nothing is duplicated. When anything changes, it changes in one place, and every caller benefits immediately.

---

## Scenario 1: Customer Support — Airline Ticket Management

### Problem Statement

An airline's AI customer support agent needs to look up flight status, rebook tickets, and issue refunds. A developer builds three Python functions and hardcodes their signatures directly into the agent's LangChain configuration. Six weeks after launch, the rebooking function gains a required `loyalty_tier` parameter for new pricing rules. The agent keeps calling the old signature. Customers asking to rebook receive silent failures — the agent reports "your flight has been rebooked" but nothing actually happens in the ticketing system.

### Solution

The airline rebuilds all three functions as tools on a custom MCP server. Each tool has a JSON Schema: `get_flight_status(flight_number: str)`, `rebook_ticket(booking_ref: str, new_flight: str, loyalty_tier: str)`, and `issue_refund(booking_ref: str, reason: str)`. When `loyalty_tier` becomes required, the MCP server's schema is updated in one file. The agent calls `tools/list` on the next session and immediately sees the updated schema — no agent redeployment needed.

**Layman version:** Before MCP, adding a required field to a form was like changing the rules of a game mid-play without telling all the players. With MCP, there is one rulebook that everyone reads before every game. Change the rulebook once, and all players are immediately playing by the new rules.

### Outcome

- Schema change detection time drops from 3 weeks (discovered through customer complaints) to the next agent session start (minutes)
- Silent rebooking failures fall from ~15% of rebook calls to 0%
- All six agents using the ticketing tools update automatically — no per-agent config change required

### Benefits

- **Single update point:** Changing a tool schema in the MCP server propagates to all agents without touching agent code
- **Structured errors surface failures:** Instead of silent empty results, the agent receives a clear error it can communicate to the customer
- **Audit trail:** Every tool call is logged at the server with booking reference, agent ID, and outcome

### Best Practices

- Define `enum` constraints on fields like `loyalty_tier` so the LLM can only generate valid values
- Return descriptive error messages that the agent can relay to the customer in plain language
- Register `issue_refund` on a separate MCP server with stricter auth than read-only lookup tools

---

## Scenario 2: Healthcare — Patient Record Access for Clinical Decision Support

### Problem Statement

A hospital deploys an AI assistant to help nurses quickly retrieve patient allergy records, current medications, and recent lab results during rounds. The developer wraps three database queries as function-calling tools with open-ended string inputs. During a security audit, reviewers discover the LLM is generating raw SQL fragments that are being interpolated into queries — a SQL injection risk. Separately, because there is no auth on the function layer, any agent that has been configured with the tool can query any patient's records without access checks.

### Solution

A clinical MCP server is built with three scoped tools: `get_patient_allergies(patient_id: str)`, `get_current_medications(patient_id: str, requesting_clinician_id: str)`, and `get_lab_results(patient_id: str, test_type: str, days_back: int)`. Each handler validates inputs against a strict JSON Schema — no raw strings passed to queries. The SSE transport uses mTLS with client certificates tied to specific clinician workstations. `requesting_clinician_id` is validated against the hospital's LDAP directory before any data is returned.

**Layman version:** Before MCP, giving the AI access to patient records was like leaving the filing room unlocked and telling the AI "the files are in there, figure it out." With MCP, it is like having a trained receptionist who knows exactly which files exist, checks your ID before opening the drawer, and hands you only the specific file you asked for — nothing more.

### Outcome

- SQL injection vulnerability eliminated — all inputs are validated against JSON Schema before reaching the database layer
- Unauthorised cross-patient record access blocked by per-request clinician ID validation
- Audit log captures every record access with clinician identity, patient ID, and timestamp for HIPAA compliance

### Benefits

- **Input validation at the protocol layer:** JSON Schema `enum` and `pattern` constraints prevent malformed or malicious arguments before they reach the database
- **Per-request authentication:** mTLS + clinician ID validation on every call, not just at session start
- **Principle of least privilege:** Each tool returns only the specific data type requested — no "get everything about this patient" mega-tool

### Best Practices

- Never accept free-form strings for patient identifiers — use `pattern` in JSON Schema to enforce the hospital's ID format (e.g., `"pattern": "^PAT-[0-9]{8}$"`)
- Deploy clinical tools on a separate MCP server from administrative tools with a stricter auth tier
- Redact all patient identifiers from server-side logs and replace with an audit-safe correlation ID

---

## Scenario 3: Finance — Automated Portfolio Reporting Agent

### Problem Statement

An investment firm builds an agent to generate daily portfolio summary reports by pulling positions, prices, and risk metrics from three internal data services. Each service has its own authentication token stored in the agent's environment variables. When the risk metrics service rotates its API token quarterly, a developer must update the token in every agent configuration that uses it — eight agents across three teams. Two agents are missed, and they silently return stale cached data for 11 days before anyone notices the reports are wrong.

### Solution

All three data services are exposed through a single MCP server: `get_positions(portfolio_id: str, as_of_date: str)`, `get_market_prices(tickers: list[str])`, and `get_risk_metrics(portfolio_id: str, metric_type: str)`. The MCP server holds the three service tokens internally. Agents authenticate to the MCP server with a single short-lived JWT — they never see the underlying service credentials. When the risk metrics token rotates, it is updated in one place: the MCP server's secrets manager reference. All eight agents continue working without any configuration change.

**Layman version:** Before MCP, every agent was like a contractor who needed a separate key card for each room they accessed. When a key card was replaced, you had to track down every contractor and give them a new card. With MCP, every contractor uses one master key card to get past the front desk, and the front desk holds all the room keys. Rotate a room key, and only the front desk needs updating.

### Outcome

- Credential rotation time drops from 2–4 hours (tracking down all agent configs) to under 5 minutes (update one secrets manager reference)
- Stale data incidents from missed credential updates eliminated — zero instances in the 6 months after migration
- Compliance audit shows clean separation: agents never have direct access to underlying data service credentials

### Benefits

- **Credential consolidation:** One MCP server holds all service credentials; agents hold only a scoped JWT for the server
- **Rotation without agent downtime:** Credential rotation at the server layer requires no agent redeployment or configuration change
- **Access control granularity:** Different JWT scopes control which tools each agent can call — the risk team's agent cannot call `get_positions` on portfolios it does not manage

### Best Practices

- Use short-lived JWTs (15–60 minute expiry) for agent-to-MCP-server authentication and refresh via a service account
- Store underlying service credentials in a secrets manager (AWS Secrets Manager, HashiCorp Vault) — never in environment variables on the MCP server host
- Implement per-portfolio access control at the tool handler level, not just at the server level

---

## Scenario 4: IT Helpdesk — Automated Infrastructure Triage Agent

### Problem Statement

An IT team deploys an agent to handle first-tier infrastructure tickets: checking server health, restarting services, and retrieving recent error logs. The developer, pressed for time, creates a single `run_command(command: str)` tool that executes arbitrary shell commands on the target servers. Within a week, the agent generates a `rm -rf /var/log/*` command while attempting to "clear disk space" — a perfectly logical action given its training, but catastrophic in execution. The agent had no way to distinguish "safe operations" from "dangerous operations" because the tool boundary was a blank check.

### Solution

The `run_command` tool is replaced with three scoped MCP tools: `get_server_health(hostname: str)` (read-only metrics), `restart_service(hostname: str, service_name: str, requires_approval: bool)` (write operation with approval gate), and `get_error_logs(hostname: str, service_name: str, lines: int)` (read-only, `lines` capped at 1000 by the schema). The `restart_service` handler checks `requires_approval` and, if true, creates a ticket in the approval queue rather than executing immediately. No shell command is ever composed or executed.

**Layman version:** Before MCP, giving the AI access to servers was like handing a new intern a master key and saying "go fix anything that looks broken." With MCP, it is like giving the intern a specific checklist of three things they are allowed to do, with a supervisor approval step on the one action that could cause problems. The intern cannot improvise outside the checklist.

### Outcome

- Dangerous command execution eliminated — there is no code path from an LLM-generated argument to a shell execution
- Mean time to resolve P3 infrastructure tickets drops from 45 minutes (waiting for human triage) to 8 minutes (agent resolves 60% of tickets autonomously within scoped tool boundaries)
- Service restart approval queue catches 4 premature restart attempts per week that would have caused unnecessary downtime

### Benefits

- **Scope enforcement at the schema layer:** JSON Schema `enum` on `service_name` limits restartable services to a pre-approved list — the LLM cannot restart a service not on the list
- **Approval gates as first-class tool behaviour:** The `requires_approval` flag makes human oversight a structured part of the tool contract, not an afterthought
- **Read/write separation:** Read tools and write tools on separate MCP servers with different auth tiers — a compromised read token cannot trigger write operations

### Best Practices

- Never expose a generic shell execution or command-running tool to an LLM agent — always decompose into specific, scoped operations
- Use JSON Schema `enum` to constrain hostnames and service names to pre-approved lists maintained by the infrastructure team
- Implement a circuit breaker: if the agent calls `restart_service` more than 3 times in 10 minutes across different hosts, suspend the tool and page a human

---

## Summary

| Aspect | Without Custom MCP Server | With Custom MCP Server |
|---|---|---|
| Schema changes | Must update every agent config manually; silent failures if missed | Update server once; all agents pick up the change at next session start |
| Error visibility | Tool failures return empty results; LLM hallucinates to fill gap | Structured JSON-RPC errors surface failures for agent error-handling |
| Credential management | Each agent holds credentials for every backend service it calls | Agents hold one server JWT; server manages all backend credentials |
| Access control | No per-tool auth boundary; any agent can call any tool | Per-tool and per-caller access control at the server layer |
| Dangerous operations | LLM can generate any argument including destructive ones | JSON Schema constraints and `enum` fields limit inputs to valid values |
| Audit trail | No centralised log; scattered across agent frameworks and backends | Every tool call logged centrally with caller ID, arguments, and outcome |
| Tool discoverability | Schemas hardcoded in agent configs; diverge from reality over time | Agents discover current schemas at runtime via `tools/list` |
