# W1D7 — LLM-as-a-Judge Evals

**Series:** AI Engineering Production Playbook
**Vertical:** Production Evals & Guardrails
**Week 1 / Day 7**

## What This Demonstrates

A second LLM acts as an automated quality scorer for the output of a first LLM,
evaluating each response against a versioned rubric with per-criterion scores,
structured JSON verdicts, and automatic routing to a human review queue when
confidence is low or any criterion fails.

## The Problem It Solves

Rule-based checks (regex, keyword matching, length gates) cannot detect semantic
failures — responses that are correctly formatted but factually wrong, policy-violating,
or critically incomplete. At production inference volume, human review of every response
is not economically viable. LLM-as-a-Judge bridges that gap.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-01/W1D7-llm-as-a-judge/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or leave blank for demo mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
LLM-as-a-Judge Evals Demo
==================================================
User prompt:        What is your return policy for opened software?
Candidate response: You can return any item within 30 days for a full refund.
Reference material: Opened software is non-refundable. ...

Running in DEMO MODE — output is pre-computed (no API call made)

--- Judge Verdict ---
Overall: FAIL (confidence: high)
  relevance: 3/3
  accuracy: 1/3 — Response states all items are refundable but reference excludes opened software.
  completeness: 2/3 — Missing the opened software exclusion.

Routes to human review: True
Parse attempts: 1

Concept demonstrated: A second LLM evaluates response quality against a calibrated rubric.
```

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — no API key required. Expected: 20+ tests, all green.

## Project Structure

```
poc/
├── src/
│   ├── main.py           # Entry point — run this file
│   ├── judge_core.py     # Rubric definitions, prompt builder, verdict parser
│   └── config.py         # Config dataclass loaded from environment variables
├── tests/
│   └── test_judge.py     # pytest unit tests (4 test classes, 20+ tests)
├── README.md
├── requirements.txt
├── .env.example
├── sample_input.json     # Example evaluation request
└── sample_output.json    # Expected judge verdict for the sample input
```

## Key Design Decisions

**Sentinel delimiters in the judge prompt**
Candidate responses are wrapped in `<<<BEGIN_RESPONSE>>>` / `<<<END_RESPONSE>>>` blocks.
This mitigates prompt injection — a malicious response cannot override judge instructions
because the judge is instructed to treat sentinel-delimited content as inert data.

**Rubric versioning**
Every verdict stores the rubric version used to produce it. Changing the rubric without
bumping the version breaks score comparability across time. The `RUBRICS` dict in
`judge_core.py` is the single source of truth; add new versions there, never mutate existing ones.

**temperature=0.0 for judge calls**
Deterministic judge output is mandatory for auditable production pipelines. Stochastic
scores cannot be compared across runs or used as reliable trend signals.

**Retry with correction hint**
When the judge returns malformed JSON (max 2 retries), a correction hint is appended to
the message history before the next call. If all retries fail, the request is routed to
the human review queue rather than silently dropped.

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (empty) | OpenAI API key — blank activates demo mode |
| `JUDGE_MODEL` | `gpt-4o-mini` | Model used as the judge |
| `MODEL` | `gpt-4o-mini` | Generator model (extended live mode) |
| `TEMPERATURE` | `0.0` | Judge temperature — keep at 0.0 |
| `RUBRIC_VERSION` | `v1.0` | Rubric version key in `judge_core.RUBRICS` |
| `MAX_JUDGE_RETRIES` | `2` | Retries on malformed JSON output |
| `DEMO_MODE` | `false` | Set `true` to skip API calls |

## Extending This PoC

**Add a new rubric criterion:**
Edit `RUBRICS["v1.0"]` in `judge_core.py` — or create a new `"v2.0"` entry if production
verdicts already exist under `v1.0` (never mutate a deployed rubric version).

**Add pairwise comparison:**
Create a `build_pairwise_prompt(prompt, response_a, response_b, rubric_version)` function
in `judge_core.py` that returns a verdict with `"preference": "A"|"B"|"tie"`.

**Calibration batch runner:**
Add `scripts/run_calibration.py` that loads a JSONL file of `{prompt, response, human_verdict}`
records, runs the judge on each, and prints precision/recall/F1 for the `fail` class.

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Layman Scenarios](../docs/llm-as-a-judge-evals-layman-scenarios.md)
- [Day README](../README.md)

## References

- Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv:2306.05685
- DeepEval: https://github.com/confident-ai/deepeval
- LangSmith Evaluation: https://docs.smith.langchain.com/evaluation
