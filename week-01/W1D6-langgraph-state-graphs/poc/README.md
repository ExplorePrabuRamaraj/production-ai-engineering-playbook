# W1D6 — State Graphs (LangGraph)

**Series:** AI Engineering Production Playbook
**Vertical:** Multi-Agent Orchestration
**Week 1 / Day 6**

---

## What This Demonstrates

A document triage workflow built as a LangGraph state graph: typed shared state flows through multiple nodes, a conditional edge routes low-risk documents to auto-processing and high-risk documents to a human approval step, and a checkpointer persists state so runs can survive crashes and resume mid-workflow.

---

## The Core Concept in One Paragraph

A state graph represents a multi-step agentic workflow as a directed graph. Each node is a Python function that reads the shared state, does one job, and returns only the fields it changed. Edges between nodes are either static (always go to the same next node) or conditional (a router function inspects state and picks the target). A checkpointer serialises state after every node, enabling mid-run recovery and human-in-the-loop pause/resume patterns. This PoC demonstrates all three: typed state, conditional routing, and a simulated human approval interrupt.

---

## Prerequisites

- Python 3.10+
- OpenAI API key — optional; demo mode runs fully offline without one

---

## Quickstart

```bash
# 1. Navigate to this folder
cd week-01/W1D6-langgraph-state-graphs/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (API key optional)
cp .env.example .env
# Edit .env — leave OPENAI_API_KEY blank to use demo mode

# 4. Run
python src/main.py
```

---

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
  State Graphs (LangGraph) Demo
==================================================
  Input document (583 chars):
  "This agreement shall indemnify the contracting party against all liability..."

  Running in DEMO MODE — langgraph simulation (no API call)

  Node: ingest       -> clauses extracted: 6
  Node: classify     -> risk_score=0.86, label=high_risk
  Router             -> high_risk path selected
  Node: human_approval
  [HUMAN-IN-THE-LOOP] High-risk document flagged for review.
  Flags: ['indemnify', 'liability', 'arbitration', 'warrant', 'termination for cause', 'liquidated damages']
  (Demo: auto-approving to complete workflow)
  Node: finalise     -> summary generated

  Output:
  risk_score       : 0.86
  risk_label       : high_risk
  flags            : ['indemnify', 'liability', ...]
  human_approved   : True
  clauses_extracted: 6
  summary          : Document review complete. Risk score: 0.86 (high_risk)...
  model            : demo

  Concept demonstrated: typed shared state flows through a conditional state
  graph — low-risk documents auto-process, high-risk documents pause for human approval.
```

---

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — every external call is mocked.

---

## Try the Low-Risk Path

Lower the risk threshold to 0.9 to force the low-risk routing path:

```bash
RISK_THRESHOLD=0.9 DEMO_MODE=true python src/main.py
```

---

## File Structure

```
poc/
├── src/
│   ├── main.py               # Entry point — run this file
│   ├── state_graph_core.py   # State schema, node functions, router, graph builder
│   └── config.py             # Config dataclass loaded from environment variables
├── tests/
│   └── test_state_graph.py   # pytest unit tests (offline, all mocked)
├── README.md
├── requirements.txt
├── .env.example
├── sample_input.json          # Example high-risk legal contract fragment
└── sample_output.json         # Expected output for the sample input
```

---

## Key Design Decisions

- **Node functions are pure.** Each node reads state, returns a partial update dict, and has no side effects outside that contract. This makes every node independently unit-testable without the graph runtime.
- **`run_demo_graph` in state_graph_core.py** executes the same node functions in topological order without requiring langgraph to be installed, so the PoC is fully runnable in a clean Python environment.
- **`retry_count` is always in state.** Any graph with a loop must have an explicit counter and a bounded exit condition — this PoC demonstrates the pattern even though the demo path does not exercise the retry loop.
- **`route_by_risk` always returns a valid key.** Missing or unexpected `risk_label` values default to `"low_risk"` — no `KeyError` in production.

---

## Extending This PoC

1. Replace the keyword-based `classify_risk` node with a real LLM call using `langchain-openai`.
2. Attach a `SqliteSaver` checkpointer and observe that restarting the process mid-run resumes from the last completed node.
3. Implement a real `interrupt_before=["human_approval"]` by compiling the graph with that flag and calling `graph.stream()` to detect the `Interrupt` event.
4. Add a retry loop: if `finalise_document` returns an empty summary, route back to `classify` with `retry_count` incremented.

---

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Day README](../README.md)
- [LangGraph Official Docs](https://langchain-ai.github.io/langgraph/)
