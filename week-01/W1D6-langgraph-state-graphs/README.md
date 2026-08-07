# W1D6 — State Graphs (LangGraph)

> Week 1, Day 6 | Vertical: Multi-Agent Orchestration  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 1](../README.md)

---

## Overview

LLM agents without explicit state management produce non-deterministic, unauditable behaviour. When the agent's "state" lives only in the context window, there is no way to pause mid-execution, resume after a failure, or inspect exactly which step produced a wrong output.

**LangGraph** solves this by making agent workflows a first-class graph. Every step is a named node — a pure function that reads from a typed shared state and returns only the fields it changes. Edges between nodes are either static (always run next) or conditional (router decides at runtime). The graph is compiled once and then invoked with a checkpointer, giving you persistence, resumability, and human-in-the-loop interrupts for free.

The PoC demonstrates this with a document triage workflow: a contract document flows through `ingest → classify → route → [auto_process | human_approval] → finalise`, with every state transition visible, logged, and testable without a live LLM.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Define a `TypedDict` state schema that is the single source of truth for all fields flowing through a graph
2. Implement node functions as pure state transformers — receive full state, return only the changed fields
3. Build a conditional edge router that maps state values to target node names
4. Compile a `StateGraph` with a checkpointer and invoke it with a `thread_id` for persistence
5. Simulate a human-in-the-loop interrupt using `interrupt_before` and graph resume
6. Explain the difference between a static edge, a conditional edge, and a terminal node
7. Test every node function in complete isolation without the LangGraph runtime

---

## Problem Statement

Stateless agents fail in three predictable ways in production:

- **No auditability** — when an agent produces a wrong answer, there is no record of which step made the bad decision; debugging requires replaying the entire run
- **No resumability** — a failure mid-workflow (network error, rate limit, tool timeout) loses all intermediate work; the entire workflow restarts from scratch
- **No human oversight** — for high-stakes decisions (contract approval, financial transactions, medical triage), there is no principled place to pause execution and wait for human input

The root cause is treating agent workflow as a single monolithic LLM call rather than a structured computation. State graphs solve all three: each node transition is checkpointed, failed runs resume from the last successful node, and `interrupt_before` pauses the graph at a named node until an external actor submits a decision.

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode (full graph simulation runs offline)
- For live mode: an OpenAI API key and `pip install langgraph langchain-openai`
- Optional: a LangSmith API key for graph tracing (`LANGCHAIN_API_KEY`)

---

## Repository Structure

```
W1D6-langgraph-state-graphs/
├── README.md                         # This file
├── docs/
│   ├── technical-document.md         # Full practitioner deep-dive (21 sections)
│   └── langgraph-layman-scenarios.md # Business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd              # State graph topology (Mermaid)
│   └── sequence.mmd                  # Document triage execution flow (Mermaid)
└── poc/
    ├── README.md                     # Quick-start and expected output
    ├── src/
    │   ├── main.py                   # Entry point — runs the graph in demo or live mode
    │   ├── state_graph_core.py       # DocumentReviewState, all nodes, router, build_graph
    │   └── config.py                 # Config dataclass + env loader
    ├── tests/
    │   └── test_state_graph.py       # pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json             # Contract document input
    └── sample_output.json            # Expected graph output (high-risk path)
```

---

## Core Concept: The Document Triage State Graph

### State Schema

A single `TypedDict` is the shared state for the entire graph. Every node reads from it and returns only the fields it changes — LangGraph merges partial updates automatically.

```python
class DocumentReviewState(TypedDict):
    document_text: str          # Raw input — set at invocation
    clauses: List[str]          # Extracted clauses — set by ingest node
    risk_score: float           # 0.0–1.0 — set by classify node
    risk_label: str             # "low_risk" | "high_risk" — set by classify node
    flags: List[str]            # Risk keywords found — set by classify node
    human_approved: Optional[bool]  # None until human_approval node runs
    summary: Optional[str]      # Final output — set by finalise node
    retry_count: int            # Guards against infinite loops
    error: Optional[str]        # Set by error_terminal on failure
```

### Graph Topology

```
START
  │
  ▼
ingest_document          # extract clauses from raw text
  │
  ▼
classify_risk            # score risk, set flags
  │
  ▼ route_by_risk() ─────────────────────────┐
  │  risk_label == "low_risk"                │ risk_label == "high_risk"
  ▼                                          ▼
auto_process             # automated summary  human_approval  # pause for human
  │                                          │
  └──────────────────┬───────────────────────┘
                     ▼
               finalise_document   # generate final summary
                     │
                     ▼
                    END

 (error_terminal → END on retry_count > MAX_RETRIES)
```

### Why Pure Node Functions Matter

Each node has the signature `(state: DocumentReviewState) -> dict`. It receives the full state and returns only the fields it changes. This means:
- **Testable in isolation** — call `classify_risk(mock_state)` in a unit test with no graph or LLM
- **Composable** — swap any node without touching others; the state schema is the contract
- **Auditable** — every state transition is a logged, named event

### Conditional Edge vs. Static Edge

```python
# Static edge — always runs
builder.add_edge("ingest", "classify")

# Conditional edge — router decides at runtime
builder.add_conditional_edges(
    "classify",
    route_by_risk,                          # returns "low_risk" or "high_risk"
    {"low_risk": "auto_process",
     "high_risk": "human_approval"},
)
```

---

## Run the PoC

### Demo Mode (No API Key Required)

```bash
cd poc
pip install -r requirements.txt
python src/main.py
```

### Live Mode (Requires OpenAI API Key + LangGraph)

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
# All tests pass offline — no API key, no langgraph install required
```

---

## Expected Output

```
  State Graphs (LangGraph) Demo
==================================================
  Input document (607 chars):
  "This agreement shall indemnify the contracting party against all liability..."

  Running in DEMO MODE — langgraph simulation (no API call)

  Node: ingest      -> clauses extracted: 6
  Node: classify    -> risk_score=0.86, label=high_risk
  Router            -> high_risk path selected
  Node: human_approval
  [HUMAN-IN-THE-LOOP] High-risk document flagged for review.
  Flags: ['indemnify', 'liability', 'arbitration', 'warrant', 'termination for cause', 'liquidated damages']
  (Demo: auto-approving to complete workflow)
  Node: finalise    -> summary generated

  Output:
  risk_score       : 0.86
  risk_label       : high_risk
  flags            : ['indemnify', 'liability', 'arbitration', 'warrant', 'termination for cause', 'liquidated damages']
  human_approved   : True
  clauses_extracted: 6
  summary          : Document review complete. Risk score: 0.86 (high_risk). Flags raised: indemnify,
                     liability, arbitration, warrant, termination for cause, liquidated damages.
                     Approved by human reviewer. Clauses reviewed: 6.
  model            : demo

  Concept demonstrated: typed shared state flows through a conditional state graph —
  low-risk documents auto-process, high-risk documents pause for human approval.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads `sample_input.json`, runs demo or live graph, prints structured output |
| `src/state_graph_core.py` | `DocumentReviewState` (schema), all node functions, `route_by_risk` (router), `build_graph` (LangGraph compiler), `run_demo_graph` (offline simulation) |
| `src/config.py` | `Config` dataclass + `load_config()` — reads `RISK_THRESHOLD`, `MAX_RETRIES`, `MODEL`, `LANGCHAIN_API_KEY` from environment |
| `tests/test_state_graph.py` | Full offline test suite: node isolation tests, router logic, state merging, error terminal, sample file schema |
| `sample_input.json` | Contract document with indemnity, arbitration, liquidated damages, and termination clauses |
| `sample_output.json` | Expected output: risk_score=0.86, high_risk label, 6 flags, human_approved=true |

---

## Configuration

Copy `.env.example` to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL=gpt-4o-mini
TEMPERATURE=0.0
MAX_TOKENS=500
DEMO_MODE=false          # Set to true to force demo mode with API key present
RISK_THRESHOLD=0.7       # Score above which documents route to human_approval
MAX_RETRIES=3            # Retries before routing to error_terminal node

# LangSmith tracing (optional — set to enable graph observability)
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=w1d6-state-graphs-langgraph
```

Demo mode activates automatically when no `OPENAI_API_KEY` is present.

---

## Technical Documentation

The full practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- Why stateless agents fail and how checkpointing addresses each failure mode
- Full `StateGraph` construction: nodes, static edges, conditional edges, `START`/`END`
- Human-in-the-loop patterns: `interrupt_before`, state injection on resume
- Checkpointer options: `MemorySaver` (in-process), `SqliteSaver` (single-node), `PostgresSaver` (production)
- LangSmith tracing integration for graph observability
- Security considerations: state schema injection, recursion limit guards, node isolation
- Cost and latency analysis: per-node LLM call overhead vs. monolithic approach
- 10 best practices, 5 anti-patterns, production checklist

For a jargon-free walkthrough, see [`docs/langgraph-layman-scenarios.md`](docs/langgraph-layman-scenarios.md).

---

## Architecture Diagrams

Mermaid source files are in [`diagrams/`](diagrams/):

- [`architecture.mmd`](diagrams/architecture.mmd) — Full state graph topology: nodes, conditional edges, both execution paths, and the error terminal
- [`sequence.mmd`](diagrams/sequence.mmd) — Document triage execution: ingest → classify → route decision → high-risk path with human-in-the-loop → finalise

---

## Connection to the Series

- **W1D1 — DSPy & Programmatic Prompts:** Typed, compiled prompt programs — the same philosophy of replacing magic strings with typed structures applied to prompts.
- **W1D2 — Lost in the Middle:** Context position affects accuracy — state graphs ensure the right context is injected at the right node, not dumped into a single prompt.
- **W1D4 — Model Context Protocol:** MCP formalises tool contracts; LangGraph formalises the workflow that calls those tools.
- **W1D5 — Episodic vs. Semantic Memory:** Memory gives agents continuity across sessions; state graphs give agents structure within a session.
- **Today — W1D6 State Graphs (LangGraph):** Typed state + conditional edges + checkpointing makes multi-step agent workflows inspectable, testable, and resumable.
- **Next — W1D7 LLM-as-a-Judge Evals:** With structured workflows in place, automated evaluation measures whether each node's output meets quality thresholds.

---

## Key References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- LangGraph Conceptual Guide — State Graphs: https://langchain-ai.github.io/langgraph/concepts/
- LangSmith (graph tracing): https://docs.smith.langchain.com/
- LangChain Blog — "Building production agents with LangGraph": https://blog.langchain.dev

---

## Continue Learning

**Next:** W1D7 — LLM-as-a-Judge Evals — Automated quality scoring that scales beyond human review and catches regressions before users do.

**Series index:** [Week 1 Overview](../README.md) | [Full Roadmap](../../README.md)
