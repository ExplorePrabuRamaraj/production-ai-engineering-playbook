# W2D5 - Reflection & Self-Correction Loops

**Series:** AI Engineering Production Playbook
**Vertical:** Agent Memory & Capabilities
**Week 2 / Day 5**

---

## What This Demonstrates

A three-node Generate -> Critique -> Revise loop that checks its own output against a structured rubric before returning it to the caller. The loop terminates when all rubric criteria pass or a hard iteration cap is reached — whichever comes first.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs fully offline)

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/your-org/production-ai-engineering-playbook

# 2. Navigate to this folder
cd week-02/W2D5_reflection-self-correction-loops/poc

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API key (or leave blank for demo mode)

# 5. Run
python src/main.py
```

---

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode returns pre-computed output that mirrors a two-iteration correction scenario — no network call is made.

---

## Run Tests

```bash
pytest tests/ -v
```

All 20+ tests pass offline. No API key required.

---

## How the Loop Works

```
User Request
     |
     v
[Confidence Gate] ----high confidence----> Return direct (skip loop)
     |
   low confidence
     |
     v
[Generate Node]  -- produces initial draft
     |
     v
[Critique Node]  -- evaluates draft against rubric (binary pass/fail per criterion)
     |
     +-- all pass --> Return validated output
     |
     +-- failures remain + under iteration cap --> [Revise Node] --> back to Critique
     |
     +-- failures remain + at iteration cap --> Return best draft with partial_pass flag
```

---

## Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — run this file |
| `src/reflection_core.py` | Core loop logic: Generate, Critique, Revise nodes + ReflectionState |
| `src/config.py` | Config dataclass loaded from environment variables |
| `tests/test_reflection.py` | 20+ unit tests covering all four test classes |
| `sample_input.json` | Example task input |
| `sample_output.json` | Expected output structure |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (empty) | OpenAI API key — leave blank for demo mode |
| `MODEL` | `gpt-4o-mini` | Model used for generate and revise nodes |
| `CRITIC_MODEL` | `gpt-4o-mini` | Model used for critique node (can differ from MODEL) |
| `MAX_ITERATIONS` | `3` | Hard cap on loop cycles before returning partial result |
| `MAX_TOKENS` | `800` | Maximum tokens per generation or revision call |
| `DEMO_MODE` | `false` | Set to `true` to run without API key |

---

## Expected Output (Demo Mode)

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

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Layman Scenarios](../docs/reflection-self-correction-loops-layman-scenarios.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [LinkedIn Post / Day Overview](../README.md)

---

## References

- Madaan et al. (2023). SELF-REFINE. arXiv:2303.17651
- Shinn et al. (2023). Reflexion. arXiv:2303.11366
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Guardrails AI Documentation: https://www.guardrailsai.com/docs
