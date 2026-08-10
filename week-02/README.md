# Week 2 — Intermediate Patterns

> Part of the [Production AI Engineering Playbook](../README.md)

---

## Objective

Build on the Week 1 foundations with intermediate production patterns. Week 2 goes deeper on each vertical — moving from first-principles understanding to applied, production-ready techniques: stronger output contracts, cost-efficient context management, graph-enhanced retrieval, custom tool servers, agent self-correction, multi-agent network topologies, and deterministic safety guardrails.

---

## Topics Covered

Week 2 maps one intermediate topic per day across the five production AI engineering verticals:

| Day | Topic | Vertical | Status |
|---|---|---|---|
| [W2D1](W2D1_type-safe-schemas-pydantic-ai/) | Type-Safe Schemas (Pydantic AI) | Prompt Engineering & Schemas | ✅ Complete |
| W2D2 | KV Caching & Token Trimming | Context Engineering & Tokens | 🔜 Coming |
| W2D3 | GraphRAG & Knowledge Graphs | Advanced RAG | 🔜 Coming |
| W2D4 | Custom MCP Server Build | MCP & Tool Integration | 🔜 Coming |
| W2D5 | Reflection & Self-Correction Loops | Agent Memory & Capabilities | 🔜 Coming |
| W2D6 | Supervisor vs. Swarm Networks | Multi-Agent Orchestration | 🔜 Coming |
| W2D7 | Deterministic Guardrails (NeMo) | Production Evals & Guardrails | 🔜 Coming |

---

## Learning Outcomes

After completing Week 2, you will be able to:

- Enforce strict data contracts at the LLM output boundary using Pydantic schemas with automatic retry
- Reduce inference cost and latency using KV cache prefix reuse and context trimming strategies
- Enhance retrieval accuracy on entity-rich queries using graph-based knowledge structures
- Build and deploy a custom MCP server that exposes typed tools to any MCP-compliant agent
- Implement reflection and self-correction loops that catch and fix agent reasoning errors in-flight
- Design multi-agent network topologies — supervisor-controlled hierarchies vs. peer-to-peer swarms
- Apply deterministic safety guardrails using NeMo Guardrails to enforce policy compliance at runtime

---

## Daily Lessons

### [W2D1 — Type-Safe Schemas (Pydantic AI)](W2D1_type-safe-schemas-pydantic-ai/)

**Vertical:** Prompt Engineering & Schemas  
**Core problem:** LLM output is a string — a 2–5% malformed-output rate in production means thousands of broken records per day. Rule-based parsing is fragile and untestable.  
**Solution:** Define output schemas as Pydantic `BaseModel` subclasses with enum constraints, field-level validators, and `extra="forbid"`. Pydantic AI injects the schema into the system prompt and retries automatically with a correction hint on `ValidationError`.

**Key concepts:** `ReviewAnalysis`, `SupportTicketTriage`, `Sentiment`/`UrgencyLevel` enums, `extra="forbid"`, `field_validator`, validator error messages as model instructions, `Agent(result_type=..., retries=...)`  
**Status:** ✅ Complete — [Start here →](W2D1_type-safe-schemas-pydantic-ai/README.md)

---

### W2D2 — KV Caching & Token Trimming

**Vertical:** Context Engineering & Tokens  
**Core problem:** Repeated large system prompts and static context are re-tokenised and processed from scratch on every request, wasting compute and increasing latency.  
**Solution:** KV cache prefix reuse amortises the cost of static context across requests; token trimming strategies remove low-value context before it reaches the model.

**Status:** 🔜 Coming soon

---

### W2D3 — GraphRAG & Knowledge Graphs

**Vertical:** Advanced RAG  
**Core problem:** Flat vector retrieval misses entity relationships — queries that require understanding how concepts connect (not just which documents contain them) return poor results from a standard embedding index.  
**Solution:** Graph-enhanced retrieval encodes relationships between entities explicitly, enabling multi-hop traversal and relationship-aware ranking.

**Status:** 🔜 Coming soon

---

### W2D4 — Custom MCP Server Build

**Vertical:** MCP & Tool Integration  
**Core problem:** Most teams use pre-built MCP servers. Building a custom server for internal tools requires understanding the full MCP server contract — tool schema, input validation, structured error responses, and transport selection.  
**Solution:** Implement a production-grade custom MCP server from scratch using the Python SDK, with schema-validated tools, health checks, and HTTP+SSE transport.

**Status:** 🔜 Coming soon

---

### W2D5 — Reflection & Self-Correction Loops

**Vertical:** Agent Memory & Capabilities  
**Core problem:** Agents make reasoning errors silently — wrong tool selection, incomplete answers, internal contradictions — and there is no mechanism to catch and fix them before the response is returned.  
**Solution:** Reflection loops add a self-critique step after generation: the agent evaluates its own output against a rubric and rewrites if the score falls below threshold.

**Status:** 🔜 Coming soon

---

### W2D6 — Supervisor vs. Swarm Networks

**Vertical:** Multi-Agent Orchestration  
**Core problem:** Multi-agent systems require a coordination model: either a central supervisor that delegates and aggregates, or a peer-to-peer swarm where agents communicate directly. Choosing the wrong topology causes bottlenecks, coordination failures, or unpredictable emergent behaviour.  
**Solution:** Explicit topology selection — supervisor for hierarchical tasks with clear subtask boundaries; swarm for parallel tasks requiring dynamic negotiation.

**Status:** 🔜 Coming soon

---

### W2D7 — Deterministic Guardrails (NeMo)

**Vertical:** Production Evals & Guardrails  
**Core problem:** LLM-as-a-Judge (W1D7) is probabilistic — it can miss policy violations and produces different verdicts on identical inputs. High-stakes applications require deterministic safety enforcement that fires on every request.  
**Solution:** NeMo Guardrails adds a rule-based safety layer that intercepts inputs and outputs, enforcing defined policies deterministically before any LLM call reaches the user.

**Status:** 🔜 Coming soon

---

## Prerequisites

- Python 3.10+
- Basic familiarity with LLM APIs (OpenAI, Anthropic, or equivalent)
- Week 1 completed (or familiarity with the core concepts)

All PoC demos run offline in `DEMO_MODE=true` — no API key required.
