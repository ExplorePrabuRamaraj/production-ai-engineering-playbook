# W1D1 — DSPy & Programmatic Prompts

> Part of the [AI Engineering Production Playbook](../../../../README.md) — Week 1, Day 1

## What This Demo Shows

| Concept | What it does |
|---|---|
| **Signature** | Defines a typed I/O contract (`question → rationale, answer`) in code, not prose |
| **ChainOfThought** | Forces the model to produce explicit reasoning before committing to an answer |
| **BootstrapFewShot** | Compiles optimal few-shot examples automatically from a labelled training set |

**Key insight:** DSPy treats prompts as compiler outputs, not hand-crafted strings. `teleprompter.compile()` generates optimal few-shot prompts from data — no manual prompt iteration required.

---

## Quick Start — Demo Mode (No API Key Required)

```bash
# 1. Navigate to this folder
cd 03_poc-code

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run in demo mode
DEMO_MODE=true python src/main.py
```

**Expected output:**
```
🚀 DSPy Programmatic Prompts Demo
==================================
⚠️  Running in demo mode (no API key). Output is pre-computed.

Query 1: What is the difference between few-shot and zero-shot prompting?
  Reasoning : Zero-shot prompting asks the model to perform a task without any examples...
  Answer    : Zero-shot uses no examples and relies on pre-trained knowledge alone...
  Demos used: 2 bootstrapped examples

...

✅ Concept demonstrated: DSPy separates program logic from prompt text,
   enabling automatic optimization via BootstrapFewShot teleprompter.
```

---

## Live Mode (Requires OpenAI API Key)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 2. Run live (dspy-ai must be installed)
python src/main.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

All 22 tests should pass without any API key.

---

## File Structure

```
03_poc-code/
├── src/
│   ├── main.py          Entry point — runs the full demo
│   ├── dspy_core.py     Signature, predictors, teleprompter (mock + real DSPy)
│   └── config.py        Config from environment variables
├── tests/
│   └── test_dspy.py     pytest unit tests (22 tests)
├── diagrams/            → see ../../02_technical-doc/diagrams/
├── docs/                → see ../../02_technical-doc/technical-document.md
├── requirements.txt
├── .env.example
├── sample_input.json    Example input accepted by main.py
└── sample_output.json   Expected output for the sample input
```

---

## Architecture at a Glance

```
User Query
    │
    ▼
Signature (typed I/O contract: question → rationale, answer)
    │
    ▼
ChainOfThought Predictor
    │        ▲
    │        │  BootstrapFewShot teleprompter
    │        │  compiles few-shot demos
    ▼        │  from labelled trainset
LM Backend (OpenAI / Local)
    │
    ▼
Typed Prediction (rationale: str, answer: str)
```

---

## References

- Khattab et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
- DSPy GitHub: [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
