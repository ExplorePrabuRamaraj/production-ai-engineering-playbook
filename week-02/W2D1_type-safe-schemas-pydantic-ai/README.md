# W2D1 — Type-Safe Schemas with Pydantic AI

> Week 2, Day 1 | Vertical: Prompt Engineering & Schemas  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 2](../README.md)

---

## Overview

When you ask an LLM to return JSON, you get a string. A 2–5% malformed-output rate in production means thousands of broken records per day — silently. Rule-based string parsing treats the LLM as a reliable serialiser; it is not.

**Pydantic AI** enforces a strict data contract at the LLM output boundary. Output schemas are defined as Pydantic `BaseModel` subclasses with enum constraints, field-level validators, and `extra="forbid"`. When the model returns output that fails validation, the agent automatically retries with the error message as a correction hint — turning a silent failure into a self-healing extraction pipeline.

The PoC demonstrates two schemas: `ReviewAnalysis` (product review extraction) and `SupportTicketTriage` (support ticket routing), both running offline in demo mode without an API key.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Define LLM output schemas as Pydantic `BaseModel` subclasses with enum constraints and field-level validators
2. Explain why `extra="forbid"` is the correct default for all LLM output schemas
3. Write validator error messages as model instructions (not internal codes) so they are meaningful on retry
4. Configure a Pydantic AI `Agent` with a `result_type` schema and `retries` count
5. Distinguish point-in-time validation (field validators) from schema-level constraints (enum, type, required)
6. Reason about the 2–5% malformed-output floor and where Pydantic validation fits in a defence-in-depth output pipeline

---

## Problem Statement

LLM output is a string. Even with `response_format={"type": "json_object"}` or structured outputs enabled, the model can:

- Return a valid JSON object with the wrong field names or types
- Omit required fields entirely
- Invent enum values outside the declared set (`"sentiment": "mixed"` instead of `"positive" | "negative" | "neutral"`)
- Exceed field-length constraints silently

Without a validation layer, these failures propagate into application logic. A routing field set to `"urgent"` instead of `"high"` is never caught; the ticket goes to the wrong queue. A confidence score of `1.5` passes JSON parsing but corrupts downstream analytics.

The standard fix — writing `if response.get("sentiment") in ("positive", "negative", "neutral")` throughout application code — is fragile, inconsistent, and untestable. **Pydantic models are the fix**: one canonical schema definition that is reused for validation, serialisation, API documentation, and test fixtures.

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode
- For live mode: an OpenAI API key set in `.env`

---

## Repository Structure

```
W2D1_type-safe-schemas-pydantic-ai/
├── README.md                                        # This file
├── docs/
│   ├── technical-document.md                        # Full practitioner deep-dive (21 sections)
│   └── type-safe-schemas-pydantic-ai-layman-scenarios.md  # Business scenarios, no ML background needed
├── diagram/
│   ├── architecture.mmd                             # Validation pipeline architecture (Mermaid)
│   └── sequence.mmd                                 # Agent → validate → retry lifecycle (Mermaid)
└── poc/
    ├── README.md                                    # Quick-start and expected output
    ├── src/
    │   ├── main.py                                  # Entry point — runs demo or live extraction
    │   ├── pydantic_schemas_core.py                 # Schema definitions + demo fixtures
    │   └── config.py                                # Config dataclass + env loader
    ├── tests/
    │   └── test_pydantic_schemas.py                 # pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                            # Product review + support ticket inputs
    └── sample_output.json                           # Expected structured output for both schemas
```

---

## Core Concept: Schema-Enforced Extraction

### The Two Schemas

**`ReviewAnalysis`** — product review extraction:

```python
class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")   # rejects any undeclared fields

    sentiment:   Sentiment      # Enum: "positive" | "negative" | "neutral"
    confidence:  float          # Validated: 0.0 ≤ value ≤ 1.0, rounded to 3dp
    key_topics:  list[str]      # Validated: at least one topic required
    summary:     str            # Validated: max 150 characters
```

**`SupportTicketTriage`** — support ticket routing:

```python
class SupportTicketTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    urgency:          UrgencyLevel   # Enum: "low" | "medium" | "high"
    department:       str            # Required, no default
    refund_involved:  bool           # Strict boolean
    one_line_summary: str            # Validated: max 100 characters
```

### Why `extra="forbid"` Matters

Without it, the model can return `{"sentiment": "positive", "mood": "happy", ...}` — the extra `mood` field is silently ignored. With `extra="forbid"`, any undeclared field raises a `ValidationError`, forcing the retry loop. It also provides a partial defence against prompt injection via output: a response that tries to inject extra fields into the schema is rejected outright.

### Validators Are Instructions, Not Codes

```python
@field_validator("confidence")
@classmethod
def confidence_must_be_in_range(cls, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError("confidence must be a decimal between 0.0 and 1.0")
    return round(v, 3)
```

The error message `"confidence must be a decimal between 0.0 and 1.0"` is fed verbatim to the LLM on retry. Write it as a model instruction, not an internal code like `"CONF_OUT_OF_RANGE"`.

### The Pydantic AI Agent

```python
review_agent = Agent(
    model="openai:gpt-4o-mini",
    result_type=ReviewAnalysis,     # schema injected into system prompt automatically
    system_prompt="Extract sentiment, confidence, key topics, and summary...",
    retries=2,                      # retry with correction hint on ValidationError
)
result = await review_agent.run(review_text)
validated: ReviewAnalysis = result.data   # guaranteed to satisfy the schema
```

---

## Run the PoC

### Demo Mode (No API Key Required)

```bash
cd poc
pip install -r requirements.txt
python src/main.py
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
# All tests pass offline — no API key needed
```

---

## Expected Output

```
Type-Safe Schemas with Pydantic AI — W2D1 Demo
==================================================
Input (review): Battery lasts 3 full days on a single charge. The build quality feels genuinely prem...
Input (ticket): I was charged twice for order #88231 placed on August 3rd. My credit card shows two ...

  Running in DEMO MODE — output is pre-computed (no API call made)

Review Analysis:
{
  "sentiment": "positive",
  "confidence": 0.92,
  "key_topics": ["battery life", "build quality", "value for money"],
  "summary": "Reviewer is highly satisfied with battery endurance and build, considers it good value."
}

Ticket Triage:
{
  "urgency": "high",
  "department": "billing",
  "refund_involved": true,
  "one_line_summary": "Customer charged twice for the same order, requesting immediate refund."
}

Model: demo

Concept demonstrated: LLM output validated against Pydantic schemas
  with field-level validators and automatic retry on failure.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads `sample_input.json`, runs demo or live extraction, prints structured output |
| `src/pydantic_schemas_core.py` | `Sentiment`, `UrgencyLevel` enums; `ReviewAnalysis`, `SupportTicketTriage` schemas with validators; `DEMO_REVIEW_OUTPUT`, `DEMO_TICKET_OUTPUT` fixtures |
| `src/config.py` | `Config` + `load_config()` — reads `MODEL`, `SCHEMA_RETRIES`, `TEMPERATURE` from environment |
| `tests/test_pydantic_schemas.py` | Schema validation, field validator behaviour, enum coercion, `extra="forbid"` rejection, parametrised valid inputs, live mode (mocked), sample file integrity |
| `sample_input.json` | Product review (battery/build quality) + support ticket (duplicate charge) |
| `sample_output.json` | Expected output: `sentiment=positive`, `urgency=high`, `refund_involved=true` |

---

## Configuration

Copy `.env.example` to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL=openai:gpt-4o-mini    # Pydantic AI model string format: "provider:model-name"
TEMPERATURE=0.0
MAX_TOKENS=500
SCHEMA_RETRIES=2             # Retries with correction hint on ValidationError
DEMO_MODE=false              # Set to true to run without an API key
```

Demo mode activates automatically when no `OPENAI_API_KEY` is present.

---

## Technical Documentation

The full practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- Why JSON mode and structured outputs are not sufficient without Pydantic validation
- Designing schemas for LLM reliability: enum coverage, field granularity, constraint precision
- Retry mechanics: how Pydantic AI constructs correction hints from `ValidationError` messages
- `extra="forbid"` as a defence against schema drift and partial prompt injection
- Cost analysis: validation retry overhead (typically <5% of total calls)
- Comparison with alternatives: OpenAI structured outputs, Instructor, Outlines, Guidance
- Production checklist, anti-patterns, and 21 interview questions

For a jargon-free walkthrough, see [`docs/type-safe-schemas-pydantic-ai-layman-scenarios.md`](docs/type-safe-schemas-pydantic-ai-layman-scenarios.md).

---

## Architecture Diagrams

Mermaid source files are in [`diagram/`](diagram/):

- [`architecture.mmd`](diagram/architecture.mmd) — Validation pipeline: LLM output → Pydantic parse → field validators → `extra="forbid"` check → retry loop → validated object
- [`sequence.mmd`](diagram/sequence.mmd) — Agent lifecycle: system prompt injection → LLM call → ValidationError → correction hint → retry → `result.data`

---

## Connection to the Series

- **W1D1 — DSPy & Programmatic Prompts:** DSPy compiles prompt programs; Pydantic AI enforces the output contract those programs must satisfy.
- **W1D7 — LLM-as-a-Judge Evals:** The judge's structured verdict (`JudgeVerdict`) is itself a Pydantic model — the same validation patterns apply.
- **W1D4 — Model Context Protocol:** MCP tool schemas are JSON Schema; Pydantic models are the Python-side equivalent for structured LLM output.
- **Today — W2D1 Type-Safe Schemas:** Adding Pydantic validation at the output boundary closes the reliability gap that prompt engineering alone cannot fill.
- **Next — W2D2 KV Caching & Token Trimming:** With reliable structured output, the next challenge is controlling inference cost at scale.

---

## Key References

- Pydantic AI Documentation: https://ai.pydantic.dev/
- Pydantic v2 Field Validators: https://docs.pydantic.dev/latest/concepts/validators/
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs

---

## Continue Learning

**Next:** W2D2 — KV Caching & Token Trimming — How to reduce inference cost and latency by reusing prefix KV cache and trimming low-value context before it reaches the model.

**Series index:** [Week 2 Overview](../README.md) | [Full Roadmap](../../README.md)
