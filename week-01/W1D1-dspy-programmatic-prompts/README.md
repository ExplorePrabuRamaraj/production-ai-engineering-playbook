# W1D1 — DSPy & Programmatic Prompts

> Week 1, Day 1 | Vertical: Prompt Engineering & Schemas  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 1](../README.md)

---

## Overview

Hand-crafted prompt strings are the "magic numbers" of AI engineering — they work until a model update, a format change, or a new edge case silently breaks them. **DSPy** (Declarative Self-improving Language Programs, Stanford NLP) replaces prompt strings with typed Python programs that compile to optimal prompts automatically.

**Research result:** BootstrapFewShot compilation improves exact-match accuracy by +8–25% across standard NLP benchmarks without changing a line of program logic (Khattab et al., arXiv:2310.03714).

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Define a DSPy `Signature` that replaces a hand-written prompt string with a typed input/output contract
2. Distinguish between `Predict`, `ChainOfThought`, and `ReAct` predictors and choose the right one for a given task
3. Explain what `teleprompter.compile()` does and when to run it
4. Run `BootstrapFewShot` against a labelled training set to produce a compiled, few-shot-optimized program
5. Identify the anti-patterns that indicate a team is using DSPy as a prompt-engineering wrapper rather than a program compiler
6. Apply DSPy's `Assert` and `Suggest` constraints to enforce output validity at runtime

---

## Problem Statement

Production prompt engineering has four compounding failure modes:

1. **Model update fragility** — A provider update changes output formatting; your hand-crafted parsing logic breaks silently
2. **No type system** — Prompt strings have no declared input/output schema; callers pass wrong types and receive plausible-looking wrong answers
3. **No optimization path** — Improving a prompt requires manual A/B testing; there is no programmatic way to compile better prompts from data
4. **Logic/text coupling** — Business logic is embedded in prompt prose; testing requires a live LLM call on every assertion

DSPy eliminates all four failure modes by treating prompt generation as a compiler problem, not a text editing problem.

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode
- For live mode: an OpenAI API key set in `.env`

---

## Repository Structure

```
W1D1-dspy-programmatic-prompts/
├── README.md                    # This file
├── docs/
│   └── technical-document.md   # 21-section deep-dive
├── diagrams/
│   ├── architecture.mmd         # 5-layer DSPy architecture (Mermaid)
│   └── sequence.mmd             # Inference vs. compilation paths (Mermaid)
└── poc/
    ├── README.md                # Quick-start and file structure guide
    ├── src/
    │   ├── main.py              # Entry point — demo and live dispatch
    │   ├── dspy_core.py         # Signatures, predictors, BootstrapFewShot
    │   └── config.py            # Config loaded from environment variables
    ├── tests/
    │   └── test_dspy.py         # 16 pytest unit tests (no API key needed)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json
    └── sample_output.json
```

---

## Technical Documentation

The full 21-section practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- Internal mechanics of `Predict`, `ChainOfThought`, and `BootstrapFewShot`
- Architecture and sequence diagrams (also rendered as Mermaid source in `diagrams/`)
- Performance benchmarks from the DSPy paper (+8 EM on HotpotQA, +12 on GSM8K, +25 on 2WikiMultihopQA)
- Security considerations (OWASP LLM01 prompt injection, LLM06 sensitive data in compiled programs, LLM09 overreliance on format validation)
- Cost analysis across workload sizes (1k / 10k / 100k requests per day)
- 10 best practices and 5 named anti-patterns
- 13-item production checklist
- 10 interview questions from Conceptual through Architecture and Production

---

## Architecture Diagrams

Mermaid source files are in [`diagrams/`](diagrams/):

- [`architecture.mmd`](diagrams/architecture.mmd) — Five-layer DSPy system: Input Layer → Signature → Module → Optimizer Layer → LM Backend
- [`sequence.mmd`](diagrams/sequence.mmd) — Two annotated paths: Inference (zero-shot predict) and Compilation (BootstrapFewShot with trainset)

---

## Run the PoC

### Demo Mode (No API Key Required)

```bash
cd poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

### Live Mode (Requires OpenAI API Key)

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
# All 22 tests pass without any API key
```

---

## Expected Output

```
🚀 DSPy Programmatic Prompts Demo
======================================
⚠️  Running in demo mode (no API key). Output is pre-computed.

Query 1: What is the difference between few-shot and zero-shot prompting?
  Reasoning : Zero-shot prompting asks the model to perform a task without any examples...
  Answer    : Zero-shot uses no examples and relies on pre-trained knowledge alone...
  Demos used: 2 bootstrapped examples

...

✅ Concept demonstrated: DSPy separates program logic from prompt text,
   enabling automatic optimization via BootstrapFewShot teleprompter.

Key DSPy abstractions shown:
  • Signature        — typed I/O contract replacing raw prompt strings
  • ChainOfThought   — predictor that requires explicit reasoning steps
  • BootstrapFewShot — teleprompter that compiles optimal few-shot demos
```

---

## Further Reading

- Khattab et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
- [DSPy GitHub — stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- [DSPy Documentation](https://dspy.ai)
- Full technical deep-dive: [`docs/technical-document.md`](docs/technical-document.md)

---

## Continue Learning

**Next:** W1D2 — "Lost in the Middle" Decay — How LLM accuracy degrades for information buried in the middle of long context windows, and the retrieval ordering strategies that prevent it.

**Series index:** [Week 1 Overview](../README.md) | [Full Roadmap](../../README.md)
