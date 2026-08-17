# W3D1 — Prompt Distillation

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

Production LLM prompts grow over time — edge-case rules, defensive instructions, and labelled examples get appended after each incident until a 200-token prompt becomes 1,800 tokens. Every extra token is charged on every call. Prompt distillation compresses a verbose "teacher" prompt into a minimal "student" prompt that retains ≥90% of the teacher's accuracy, reducing token cost by 60%+ without retraining the model. This PoC implements a greedy sentence-pruning distillation loop with a configurable accuracy floor, scores each candidate against a labelled eval set, and projects the resulting monthly and annual cost savings at your call volume.

---

## Learning Objectives

1. Understand why production prompts bloat and quantify the token cost of accumulated defensive instructions
2. Implement `build_teacher_prompt()` and `build_student_prompt()` to create a verbose/minimal prompt pair for the same task
3. Implement `score_prompt_candidate()` to evaluate any prompt against a labelled eval set and return an accuracy float
4. Implement `distill_prompt()` — the greedy sentence-pruning loop that removes the longest sentence at each iteration while accuracy stays above the floor
5. Use `compute_token_savings()` to project monthly and annual cost savings from a given token reduction at a given call volume
6. Know when to stop: understand what "load-bearing sentences" are and why the accuracy floor is the correct stopping criterion
7. Understand how this greedy heuristic relates to production optimisers like DSPy's `MIPROv2` and where the heuristic breaks down

---

## Problem Statement

At 200,000 calls/month, a 1,800-token system prompt costs ~$54/month in input tokens on gpt-4o-mini — before any user content is counted. That overhead was added incrementally: each production incident prompted an extra rule or example, and no one removed instructions that became redundant after model updates. The result is a prompt full of load-bearing sentences mixed with dead weight. Without a systematic way to measure which sentences can be removed without accuracy loss, engineers either leave the bloated prompt in place (paying the ongoing cost) or remove sentences by intuition (risking silent accuracy regressions). Prompt distillation solves this by making the trade-off explicit and measurable.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — all demos run offline with `DEMO_MODE=true`)
- Familiarity with [W1D1 — DSPy & Programmatic Prompts](../../week-01/W1D1-dspy-programmatic-prompts/README.md) provides useful context

---

## Repository Structure

```
W3D1_prompt-distillation/
├── README.md                              # This file
├── docs/
│   ├── technical-document.md              # 21-section practitioner deep-dive
│   └── prompt-distillation-layman-scenarios.md
├── diagrams/
│   ├── architecture.mmd                   # Teacher → student distillation pipeline
│   └── sequence.mmd                       # Per-iteration pruning sequence
└── poc/
    ├── README.md                          # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                        # Entry point — demo + live mode
    │   ├── distillation_core.py           # All distillation logic (pure functions)
    │   └── config.py                      # Config dataclass + env loader
    ├── tests/
    │   └── test_distillation.py           # 17 unit tests across 4 test classes
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                  # Legal NDA document classification task
    └── sample_output.json                 # Pre-computed demo result
```

---

## Core Concepts

### Teacher vs. student prompts

The teacher prompt is over-specified — it contains exhaustive rules, numbered constraints, and labelled examples. The student prompt strips everything except the essential category list and output constraint:

```python
# distillation_core.py
_TEACHER_PROMPT_BODY = """
You are a legal document classification assistant with expertise in contract law.
Your task is to classify the document into exactly ONE of the following categories:
  - NDA, SaaS, Employment, IP, Refund, General
Classification rules:
  1. Read the entire document before deciding.
  2. If confidentiality clauses constitute more than 50% of the substantive content, classify as NDA.
  ...  (10 rules + 3 labelled examples)
"""

_STUDENT_PROMPT_BODY = """
Classify the document into one category: NDA, SaaS, Employment, IP, Refund, or General.
Output only the category name.
"""
```

### Greedy pruning loop

`distill_prompt()` removes the longest sentence at each iteration. If accuracy after removal stays at or above the `accuracy_floor`, the removal is accepted; otherwise the sentence is load-bearing and the loop stops:

```python
# distillation_core.py
def distill_prompt(teacher_prompt, eval_examples, call_llm_fn,
                   accuracy_floor=0.90, max_iterations=5) -> dict:
    sentences = [s.strip() for s in teacher_prompt.split(".") if s.strip()]
    current_prompt = teacher_prompt
    for _ in range(max_iterations):
        target = max(
            [s for s in sentences if s in current_prompt], key=len
        )
        pruned = current_prompt.replace(target + ".", "").strip()
        if score_prompt_candidate(pruned, eval_examples, call_llm_fn) >= accuracy_floor:
            current_prompt = pruned   # removal accepted
        else:
            break                     # load-bearing sentence — stop
    return {"student_prompt": current_prompt, "token_reduction_pct": ..., ...}
```

### Cost-savings projection

```python
# distillation_core.py
savings = compute_token_savings(
    teacher_tokens=1_800,
    student_tokens=640,
    daily_calls=6_667,          # ~200k/month
    cost_per_1m_tokens=0.15,    # gpt-4o-mini, mid-2025
)
# → monthly_savings_usd: 2.90, annual_savings_usd: 35.40
```

### Scoring a prompt candidate

`score_prompt_candidate()` accepts an injectable `call_llm_fn` so the function stays side-effect free and fully testable offline:

```python
# distillation_core.py
def score_prompt_candidate(prompt, eval_examples, call_llm_fn) -> float:
    correct = sum(
        1 for ex in eval_examples
        if call_llm_fn(prompt, ex["input"]).strip().upper() == ex["label"].upper()
    )
    return correct / len(eval_examples)
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-03/W3D1_prompt-distillation/poc
pip install -r requirements.txt
cp .env.example .env
DEMO_MODE=true python src/main.py
```

### Live mode

```bash
# Edit .env and set OPENAI_API_KEY=your-key
python src/main.py
```

### Tests

```bash
pytest tests/ -v
# Expected: 17 tests pass, 0 fail, 0 network calls
```

---

## Expected Output

```
Prompt Distillation Demo
==================================================

Task       : classify_document
Input text : This Non-Disclosure Agreement ("Agreement") is entered into as of...

[DEMO MODE] Running with pre-computed output — no API key required.

  Teacher prompt tokens : 1800
  Student prompt tokens : 640
  Token reduction       : 64.4%
  Teacher accuracy      : 96.2%
  Student accuracy      : 95.4%
  Accuracy delta        : -0.8%
  Monthly savings       : $2.90 (at 200,000 calls/month, gpt-4o-mini pricing)
  Model                 : demo
  Latency (ms)          : 0

Concept demonstrated: A 1800-token teacher prompt distilled into a
  640-token student prompt with <1pp accuracy delta on held-out eval.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads config, runs demo or live mode, prints distillation metrics |
| `src/distillation_core.py` | `build_teacher_prompt()`, `build_student_prompt()`, `score_prompt_candidate()`, `distill_prompt()`, `compute_token_savings()`, `run_distillation_demo()`, `run_distillation_live()` |
| `src/config.py` | `Config` dataclass + `load_config()` with accuracy floor, iteration cap, and cost parameters |
| `tests/test_distillation.py` | 4 test classes, 17 tests: prompt building, scoring, distillation loop, cost projection |
| `sample_input.json` | Legal NDA document classification task |
| `sample_output.json` | Pre-computed demo result: 1800 → 640 tokens, -0.8pp accuracy delta, $2.90/month savings |
| `.env.example` | All environment variable defaults |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `MODEL` | `gpt-4o-mini` | LLM model for classification scoring |
| `DEMO_MODE` | `false` | Set `true` to skip all API calls |
| `ACCURACY_FLOOR` | `0.90` | Minimum student accuracy fraction (0.0–1.0) |
| `MAX_DISTILLATION_ITERATIONS` | `5` | Max pruning rounds before stopping |
| `DAILY_CALLS` | `6667` | Projected calls/day for cost savings calculation (~200k/month) |
| `COST_PER_1M_TOKENS` | `0.15` | Input token price USD/1M (gpt-4o-mini, mid-2025) |
| `TEMPERATURE` | `0.0` | LLM temperature — keep at 0 for deterministic classification |
| `MAX_TOKENS` | `16` | Max tokens per LLM response (category label is ≤10 tokens) |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/prompt-distillation-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — Teacher → student distillation pipeline
- [Sequence Diagram](diagrams/sequence.mmd) — Per-iteration greedy pruning sequence

---

## Connection to the Series

**Previous:** [W2D1 — Type-Safe Schemas (Pydantic AI)](../../week-02/W2D1_type-safe-schemas-pydantic-ai/README.md) — the Week 2 entry point for Prompt Engineering & Schemas, covering output contracts at the LLM boundary.

**Next:** [W3D2 — Context Compression](../README.md) — compress conversation history and retrieved context rather than the system prompt itself.

**Series arc:** [W1D1 — DSPy & Programmatic Prompts](../../week-01/W1D1-dspy-programmatic-prompts/README.md) introduced programmatic prompt construction. W2D1 added type-safe output schemas. W3D1 closes the loop: once prompts are working correctly and output is validated, systematically shrink them to reduce cost while preserving accuracy. The W3D1 greedy heuristic is a stepping stone toward DSPy's `MIPROv2` Bayesian optimiser, which searches over the full instruction + few-shot candidate space.

---

## Key References

- Khattab, O. et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." arXiv:2310.03714
- [DSPy MIPROv2 optimizer documentation](https://dspy.ai/deep-dive/optimizers/miprov2/)
- [OpenAI pricing reference](https://openai.com/api/pricing/)

---

## Continue Learning

**Next:** [W3D2 — Context Compression](../README.md)

Return to [Week 3 overview](../README.md) to explore all advanced techniques.
