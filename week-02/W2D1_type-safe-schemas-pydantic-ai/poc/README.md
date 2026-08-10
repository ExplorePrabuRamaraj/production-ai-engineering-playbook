# W2D1 — Type-Safe Schemas with Pydantic AI

**Series:** AI Engineering Production Playbook
**Vertical:** Prompt Engineering & Schemas
**Week 2 / Day 1**

## What This Demonstrates

How to enforce a strict data contract at the LLM output boundary using Pydantic AI — so that malformed, missing, or wrongly-typed fields are caught and retried automatically before they reach application logic.

## The Problem It Solves

When you ask an LLM to return JSON, you get a string. A 2–5% malformed-output rate in production means thousands of broken records per day — silently. This PoC demonstrates how to:

- Define output schemas as Pydantic `BaseModel` subclasses
- Enforce enum constraints on categorical fields
- Apply field-level validators with model-directed error messages
- Trigger automatic retry with a targeted correction hint on validation failure
- Run the full pipeline offline with pre-computed demo output

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode available without any key)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-02/W2D1_type-safe-schemas-pydantic-ai/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API key — or leave blank for demo mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
Type-Safe Schemas with Pydantic AI — W2D1 Demo
==================================================
Input (review): Battery lasts 3 full days on a single charge...
Input (ticket): I was charged twice for order #88231...

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

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — no API key required. The test suite covers:

- Demo mode output schema correctness
- Field validator behaviour (confidence range, summary length, extra fields rejected)
- Enum coercion for sentiment and urgency fields
- Parametrised tests across multiple valid input combinations
- Live mode with mocked Pydantic AI agent
- Sample file integrity validation

## Project Structure

```
poc/
├── src/
│   ├── main.py                    # Entry point — run this file
│   ├── pydantic_schemas_core.py   # Schema definitions + demo fixtures
│   └── config.py                  # Config dataclass + env loader
├── tests/
│   └── test_pydantic_schemas.py   # pytest unit tests (all offline)
├── README.md                      # This file
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
├── sample_input.json              # Example review + ticket input
└── sample_output.json             # Expected structured output
```

## Key Schemas

**ReviewAnalysis** — extracts from product review text:

| Field | Type | Constraint |
|---|---|---|
| `sentiment` | `Sentiment` enum | positive / negative / neutral |
| `confidence` | `float` | 0.0 – 1.0, rounded to 3dp |
| `key_topics` | `list[str]` | At least 1 topic |
| `summary` | `str` | Max 150 characters |

**SupportTicketTriage** — extracts from support ticket text:

| Field | Type | Constraint |
|---|---|---|
| `urgency` | `UrgencyLevel` enum | low / medium / high |
| `department` | `str` | Required, no default |
| `refund_involved` | `bool` | Strict boolean |
| `one_line_summary` | `str` | Max 100 characters |

Both models use `model_config = ConfigDict(extra="forbid")` to reject any field not declared in the schema.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `""` | API key — leave blank for demo mode |
| `MODEL` | `openai:gpt-4o-mini` | Pydantic AI model string |
| `SCHEMA_RETRIES` | `2` | Max retries on validation failure |
| `DEMO_MODE` | `false` | Force demo mode regardless of API key |

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Layman Scenarios](../docs/type-safe-schemas-pydantic-ai-layman-scenarios.md)
- [Architecture Diagram](../diagram/architecture.mmd)
- [Sequence Diagram](../diagram/sequence.mmd)
- [Day README](../README.md)

## References

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic v2 Field Validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
