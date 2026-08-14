# W2D5 — Reflection & Self-Correction Loops

> [Week 2](../README.md) · [Playbook](../../README.md)

## Overview

A reflection loop adds a self-critique step between generation and response: the agent produces a draft, evaluates it against a structured rubric, revises only the failing sections, and repeats — until all criteria pass or a hard iteration cap is reached. This PoC implements a three-node **Generate → Critique → Revise** loop that terminates deterministically and always returns the best draft it has, never hanging indefinitely.

**Vertical:** Agent Memory & Capabilities | **Week:** 2 | **Day:** 5

---

## Learning Objectives

1. Understand why agents produce reasoning errors silently and why no downstream check catches them without an explicit self-critique step
2. Design a rubric-driven critique prompt that returns structured per-criterion `pass/fail` judgements rather than prose feedback
3. Implement `generate_node()`, `critique_node()`, and `revise_node()` as pure, independently testable functions
4. Wire the three nodes into `run_reflection_loop()` with a hard `max_iterations` cap that guarantees termination
5. Build a targeted revision prompt that fixes only failing criteria without regressing passing sections
6. Interpret `ReflectionState`: `exited_at_cap`, `history`, and `critique` for routing and debugging
7. Apply the confidence gate pattern — skip the loop entirely when the task is routine — to avoid unnecessary latency

---

## Problem Statement

Agents make reasoning errors silently. A wrong tool selection, an incomplete answer, or a factual claim that contradicts the context — none of these raise an exception. The agent returns the flawed output, the user sees it, and there is no mechanism that caught the failure before it escaped.

In production, this manifests as hallucinated citations that pass review, summaries that omit required elements, and responses that violate stated format constraints. Without a self-critique step, the only safety net is human review — which does not scale.

The reflection loop inserts a programmatic quality gate: every draft is evaluated against a rubric before it reaches the caller. If it fails, a targeted revision pass fixes only the failing sections. If it still fails after `MAX_ITERATIONS` cycles, it exits with an `exited_at_cap=True` flag that can route the output to human review.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs fully offline)
- [W1D5 — Episodic vs. Semantic Memory](../../week-01/W1D5-agent-memory/README.md) recommended

---

## Repository Structure

```
W2D5_reflection-self-correction-loops/
├── README.md                                           # This file
├── docs/
│   ├── technical-document.md                          # 21-section practitioner deep-dive
│   └── reflection-self-correction-loops-layman-scenarios.md   # Business scenarios
├── diagrams/
│   ├── architecture.mmd                               # Loop architecture (Mermaid)
│   └── sequence.mmd                                   # Iteration-by-iteration sequence (Mermaid)
└── poc/
    ├── README.md                                      # PoC quickstart and expected output
    ├── src/
    │   ├── main.py                                    # Entry point — demo and live modes
    │   ├── reflection_core.py                         # Generate, Critique, Revise nodes +
    │   │                                              #   run_reflection_loop, DEFAULT_RUBRIC
    │   └── config.py                                  # Config dataclass loaded from environment
    ├── tests/
    │   └── test_reflection.py                         # 20+ unit tests, all offline-capable
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                              # Example task input
    └── sample_output.json                             # Expected 2-iteration output
```

---

## Core Concepts

### 1. The Rubric — the Most Important Engineering Artefact

```python
from reflection_core import DEFAULT_RUBRIC

# DEFAULT_RUBRIC contains 3 criteria checked on every draft:
# [factual_accuracy, completeness, constraint_compliance]
#
# Each criterion has:
#   name                — identifier used in logs and iteration_log
#   check               — the yes/no question the critic must answer
#   revision_instruction — what the revise node must fix if this fails
```

The rubric is the contract between what the agent produces and what the system requires. Engineering the rubric — not the loop mechanics — is where the quality improvement actually comes from. Vague criteria produce vague critiques; binary yes/no criteria with concrete revision instructions produce reliable targeted rewrites.

### 2. Three Nodes

```python
from reflection_core import generate_node, critique_node, revise_node

# Generate — produce initial draft
draft = generate_node(task, client, model="gpt-4o-mini", max_tokens=800)

# Critique — evaluate draft against rubric, returns CritiqueResult
critique = critique_node(draft, rubric=DEFAULT_RUBRIC, iteration=1, client=client, model="gpt-4o-mini")
# critique.all_passed  → bool
# critique.failing_criteria()  → list[CriterionResult]
# critique.summary()  → "Iteration 1: FAIL (2/3 criteria passed)"

# Revise — rewrite only failing sections (leaves passing sections unchanged)
revised = revise_node(draft, critique, client, model="gpt-4o-mini", max_tokens=800)
```

`critique_node` returns structured JSON — one `{"name", "passed", "revision_instruction"}` object per criterion. If the critic returns malformed JSON, all criteria are treated as failing (safe default) and the raw output is logged for debugging.

`revise_node` builds a targeted prompt that lists only the failing criteria and explicitly instructs the model not to modify passing sections. This prevents the regression-by-full-rewrite anti-pattern, where a revision that fixes one criterion accidentally breaks another.

### 3. Orchestrator with Hard Termination Cap

```python
from reflection_core import run_reflection_loop, DEFAULT_RUBRIC

state = run_reflection_loop(
    task="Summarise the top 3 risks of deploying LLMs in production...",
    client=client,
    model="gpt-4o-mini",
    critic_model="gpt-4o-mini",  # can be a cheaper model
    max_tokens=800,
    max_iterations=3,            # hard cap — loop always terminates
    rubric=DEFAULT_RUBRIC,
)

# state.draft          → final (or best) draft
# state.critique       → last CritiqueResult
# state.iteration      → number of cycles used
# state.exited_at_cap  → True if loop hit cap without all criteria passing
# state.history        → list of (iteration, draft_length, critique_summary) snapshots
```

The loop exits on the first iteration where all criteria pass, or at `max_iterations` — whichever comes first. `exited_at_cap=True` is a routing signal: route to human review rather than returning to the user automatically.

---

## Run the PoC

**Demo mode (no API key required):**

```bash
cd week-02/W2D5_reflection-self-correction-loops/poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

**Live mode (OpenAI API key required):**

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python src/main.py
```

**Run tests:**

```bash
pytest tests/ -v
```

---

## Expected Output

```
Reflection & Self-Correction Loops Demo
==================================================
WARNING: Running in DEMO MODE -- output is pre-computed (no API call made)

Task:       Summarise the top 3 risks of deploying LLMs in production...
Model:      demo
Status:     PASS
Iterations: 2
Latency:    0 ms

--- Iteration Log ---
  Iteration 1: FAIL (2/3 criteria passed)
    Failing criteria: completeness
  Iteration 2: PASS (3/3 criteria passed)

--- Final Draft ---
- Hallucination risk: LLMs confidently produce false outputs; validate every
  factual claim before surfacing to users (Anthropic, 2024).
- Prompt injection: adversarial inputs can hijack agent behaviour; sanitise
  all user-controlled content before interpolation.
- Cost unpredictability: unbounded context growth spikes token spend; enforce
  input and output length limits at the gateway.

Concept demonstrated: Generate -> Critique -> Revise loop with hard termination cap.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — selects demo or live mode, formats and prints result |
| `src/reflection_core.py` | `generate_node`, `critique_node`, `revise_node`, `run_reflection_loop`, `DEFAULT_RUBRIC`, `build_critique_prompt`, `build_revision_prompt` |
| `src/config.py` | `Config` dataclass + `load_config()` from environment variables |
| `tests/test_reflection.py` | 20+ tests across all nodes, rubric evaluation, prompt builders, and state tracking |
| `sample_input.json` | Example task with multi-constraint requirements |
| `sample_output.json` | Expected 2-iteration output: iteration 1 FAIL → iteration 2 PASS |

---

## Configuration

All parameters are loaded from environment variables (see `.env.example`):

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for live mode. Absent → demo mode activates automatically. |
| `MODEL` | `gpt-4o-mini` | Model for `generate_node` and `revise_node`. |
| `CRITIC_MODEL` | `gpt-4o-mini` | Model for `critique_node`. Can be set to a cheaper model — critique is simpler than generation. |
| `TEMPERATURE` | `0.0` | Temperature for all nodes. Keep at 0 for deterministic critique. |
| `MAX_TOKENS` | `800` | Max tokens per generation or revision call. |
| `MAX_ITERATIONS` | `3` | Hard cap on loop cycles. Loop always exits at or before this count. |
| `CONFIDENCE_THRESHOLD` | `0.85` | Confidence gate: if confidence ≥ this value, skip the loop and return directly. |
| `DEMO_MODE` | `false` | Force demo mode regardless of API key. |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section deep-dive on reflection loop design, rubric engineering, prompt injection mitigations, and production deployment patterns
- [Layman Scenarios](docs/reflection-self-correction-loops-layman-scenarios.md) — Business scenarios explaining self-correction loops without ML background

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — Confidence gate → Generate → Critique → Revise loop with hard cap and human-review routing
- [Sequence Diagram](diagrams/sequence.mmd) — Iteration-by-iteration flow: draft production → per-criterion evaluation → targeted revision → termination conditions

---

## Connection to the Series

| | Day | Topic |
|---|---|---|
| ← Previous | [W2D4 — Custom MCP Server Build](../W2D4_custom-mcp-server-build/README.md) | Typed tool contracts for agents |
| ← Foundation | [W1D5 — Episodic vs. Semantic Memory](../../week-01/W1D5-agent-memory/README.md) | Agent memory architecture |
| → Next | W2D6 — Supervisor vs. Swarm Networks | Multi-agent coordination topologies |

**Why this follows W2D4:** W2D4 ensured tool calls return structured, validated results. W2D5 adds the layer above — ensuring the agent catches and fixes its own reasoning errors *after* generation, before the response reaches the user. Together, they close the two failure modes that matter most: bad tool inputs (W2D4) and bad generated outputs (W2D5).

---

## Key References

- Madaan et al. (2023). SELF-REFINE: Iterative Refinement with Self-Feedback. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- LangGraph Documentation: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)

---

## Continue Learning

**Next:** W2D6 — Supervisor vs. Swarm Networks *(coming soon)*
