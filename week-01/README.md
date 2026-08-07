# Week 1 — AI Engineering Foundations

> Part of the [Production AI Engineering Playbook](../README.md)

---

## Objective

Establish the core vocabulary and tooling for production AI engineering. By the end of Week 1, you will understand the fundamental patterns that make AI systems reliable in production — typed prompt programs, context-aware retrieval, agent memory, and automated evaluation — and have runnable code demonstrating each.

---

## Topics Covered

Week 1 maps one topic per day across the five production AI engineering verticals:

| Day | Topic | Vertical | Status |
|---|---|---|---|
| [W1D1](W1D1-dspy-programmatic-prompts/) | DSPy & Programmatic Prompts | Prompt Engineering & Schemas | ✅ Complete |
| [W1D2](W1D2-lost-in-the-middle/) | "Lost in the Middle" Decay | Context Engineering & Tokens | ✅ Complete |
| [W1D3](W1D3-naive-vs-agentic-rag/) | Naive vs. Agentic RAG | Advanced RAG | ✅ Complete |
| [W1D4](W1D4-model-context-protocol/) | Model Context Protocol (MCP) Intro | MCP & Tool Integration | ✅ Complete |
| [W1D5](W1D5-agent-memory/) | Episodic vs. Semantic Memory | Agent Memory & Capabilities | ✅ Complete |
| [W1D6](W1D6-langgraph-state-graphs/) | State Graphs (LangGraph) | Multi-Agent Orchestration | ✅ Complete |
| W1D7 | LLM-as-a-Judge Evals | Production Evals & Guardrails | 🔜 Coming |

---

## Learning Outcomes

After completing Week 1, you will be able to:

- Replace hand-crafted prompt strings with typed, optimizable DSPy programs
- Identify and mitigate "lost in the middle" accuracy decay in long-context LLM calls
- Explain when naive RAG fails and when agentic RAG patterns are warranted
- Integrate external tools with LLMs via the Model Context Protocol
- Design agent memory systems using episodic and semantic memory patterns
- Model multi-step agent workflows as typed state graphs with LangGraph
- Implement LLM-as-a-Judge evaluation pipelines to measure output quality automatically

---

## Daily Lessons

### [W1D1 — DSPy & Programmatic Prompts](W1D1-dspy-programmatic-prompts/)

**Vertical:** Prompt Engineering & Schemas  
**Core problem:** A hand-crafted prompt that works today will silently fail next quarter — and you won't know why.  
**Solution:** DSPy replaces prompt strings with typed Python programs that compile to optimal prompts automatically using `BootstrapFewShot` and the teleprompter.

**Key concepts:** Signature, Predict, ChainOfThought, BootstrapFewShot, teleprompter.compile()  
**Status:** ✅ Complete — [Start here →](W1D1-dspy-programmatic-prompts/README.md)

---

### [W1D2 — "Lost in the Middle" Decay](W1D2-lost-in-the-middle/)

**Vertical:** Context Engineering & Tokens  
**Core problem:** LLMs reliably recall information at the start and end of a context window — accuracy degrades for content buried in the middle. Naive document stuffing causes silent accuracy drops.  
**Solution:** LiTM-aware document ordering places the highest-relevance documents at context boundaries (positions 0 and N-1) where transformer attention peaks, recovering up to 26% of mean effective retrieval score over naive ordering.

**Key concepts:** U-shaped attention, primacy bias, recency bias, middle dead zone, LiTM-aware interleaving, effective attention score  
**Status:** ✅ Complete — [Start here →](W1D2-lost-in-the-middle/README.md)

---

### [W1D3 — Naive vs. Agentic RAG](W1D3-naive-vs-agentic-rag/)

**Vertical:** Advanced RAG  
**Core problem:** Single-retrieval RAG fails on multi-hop questions that require synthesizing information from multiple sources. Naive RAG fetches by surface similarity and has no awareness of whether the retrieved evidence actually answers the query.  
**Solution:** Agentic RAG decomposes the query into atomic sub-questions, retrieves targeted evidence per sub-question, validates each result against a similarity threshold, reformulates and retries on failure, then synthesises a final answer with inline citations.

**Key concepts:** QueryDecomposer, ChunkRetriever, evidence validation, reformulation retry, multi-hop retrieval, retrieval-as-tool-call  
**Status:** ✅ Complete — [Start here →](W1D3-naive-vs-agentic-rag/README.md)

---

### [W1D4 — Model Context Protocol (MCP) Intro](W1D4-model-context-protocol/)

**Vertical:** MCP & Tool Integration  
**Core problem:** Unstructured tool descriptions bleed into prompts and degrade reliability. Ad-hoc tool wiring breaks silently when upstream APIs change — schema drift causes 15–25% of agent reliability incidents.  
**Solution:** MCP formalises the tool contract between your LLM and external systems with a typed, versioned, JSON-RPC 2.0 interface. The server owns its schema; every connected client fetches the live schema at session start — one update propagates everywhere.

**Key concepts:** Resources, Tools, Prompts, capability negotiation, `isError` vs. JSON-RPC errors, stdio vs. HTTP+SSE transport  
**Status:** ✅ Complete — [Start here →](W1D4-model-context-protocol/README.md)

---

### [W1D5 — Episodic vs. Semantic Memory](W1D5-agent-memory/)

**Vertical:** Agent Memory & Capabilities  
**Core problem:** Naive context stuffing (full conversation history in every prompt) collapses in production — context overflow, cost explosion ($235/day at 500 conversations), and no cross-session recall.  
**Solution:** A dual-memory architecture separates episodic memory (time-stamped, user-scoped events with hybrid similarity+recency retrieval) from semantic memory (validated, generalised facts promoted asynchronously via a gated pipeline). Replaces ~8,000 tokens of raw history with ~2,000 tokens of relevant context — a 78% token cost reduction.

**Key concepts:** EpisodicMemory, SemanticMemory, PromotionPipeline, MemoryRouter, recency-weighted retrieval, token budget assembly, injection-safe context delimiters  
**Status:** ✅ Complete — [Start here →](W1D5-agent-memory/README.md)

---

### [W1D6 — State Graphs (LangGraph)](W1D6-langgraph-state-graphs/)

**Vertical:** Multi-Agent Orchestration  
**Core problem:** LLM agents without explicit state management produce non-deterministic, unauditable behaviour — failures mid-workflow lose all intermediate work, and there is no principled place to pause for human input.  
**Solution:** LangGraph models workflows as typed state graphs: each node is a pure function on `DocumentReviewState`, conditional edges route by state value at runtime, and a checkpointer persists every node transition — enabling mid-run recovery, resume, and human-in-the-loop interrupts.

**Key concepts:** `TypedDict` state schema, pure node functions, `add_conditional_edges`, `route_by_risk`, `interrupt_before`, `MemorySaver`/`SqliteSaver` checkpointers, `recursion_limit` guard  
**Status:** ✅ Complete — [Start here →](W1D6-langgraph-state-graphs/README.md)

---

### W1D7 — LLM-as-a-Judge Evals

**Vertical:** Production Evals & Guardrails  
**Core problem:** Human evaluation doesn't scale. Without automated quality measurement, accuracy regressions go undetected until users report them.  
**Solution:** LLM-as-a-Judge enables continuous, automated quality scoring — with calibration against human labels to keep scores meaningful.

**Status:** 🔜 Coming soon

---

## Prerequisites

- Python 3.10+
- Basic familiarity with LLM APIs (OpenAI, Anthropic, or equivalent)
- `pip` and a virtual environment tool

All PoC demos run offline in `DEMO_MODE=true` — no API key required.
