# W3D7 — Distributed Tracing (LangSmith)

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

LLM pipelines are opaque by default: a latency spike or accuracy regression in production has no obvious root cause without tracing through each model call, retrieval step, and tool invocation. Without observability, debugging means adding print statements and re-running — expensive and unreliable. This PoC instruments a 5-step RAG pipeline (retrieve → rerank → assemble → generate → validate) with LangSmith distributed tracing: every step emits a child span capturing its inputs, outputs, token count, and latency, and a deterministic validator attaches a grounding score to the root span after generation. In demo mode the full pipeline runs offline with pre-computed mock data to explore the span structure without any API key.

---

## Learning Objectives

1. Understand why LLM pipelines need distributed tracing and what per-span input/output capture enables that logs alone cannot
2. Implement the 5-step RAG pipeline — `retrieve_documents()`, `rerank_documents()`, `assemble_context()`, `generate_answer`, `validate_answer()` — as individually traceable functions
3. Apply the `@traceable` LangSmith decorator with correct `run_type` values (`"retriever"`, `"chain"`, `"llm"`) to emit typed child spans to the run tree
4. Implement `validate_answer()` — a deterministic grounding check that computes word-overlap ratio between the answer and retrieved documents, preventing hallucination over silent `None` values
5. Understand `PipelineResult` and `RetrievedDocument` as the structured output contracts that make span outputs machine-readable in the LangSmith trace explorer
6. Configure `tracing_enabled` correctly — tracing requires both `OPENAI_API_KEY` and `LANGCHAIN_API_KEY`; missing either should activate demo mode automatically, not raise an unhandled exception
7. Know what each span type reveals: `retriever` spans surface recall quality, `chain` spans show context assembly truncation, `llm` spans expose token cost and latency, validator spans isolate grounding failures from generation failures

---

## Problem Statement

A production RAG support chatbot handles 50,000 queries/day. After a model update, user satisfaction drops 12% over three days. The engineering team has logs showing answer latency and error rates, but cannot determine whether the regression is in retrieval (wrong documents returned), reranking (correct documents ranked too low), context assembly (key sentences truncated), generation (model ignoring the context), or the validator (false negatives). Without per-span tracing, diagnosing the root cause requires reproducing individual queries offline, instrumenting each step manually, and re-running — a process that takes days. With distributed tracing, the team opens the LangSmith trace explorer, filters by grounding score below 0.5, and immediately sees that the `assemble_context` span is truncating the electronics return policy document — a `max_chars` threshold that was not updated when document lengths increased after the model update. Fix time: 20 minutes.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs entirely offline)
- LangSmith API key (optional — tracing is a no-op without it; demo mode explores span structure without any key)
- Familiarity with [W1D7 — LLM-as-a-Judge Evals](../../week-01/W1D7-llm-as-a-judge/README.md) and [W2D7 — Deterministic Guardrails (NeMo)](../../week-02/W2D7_deterministic-guardrails-nemo/README.md) provides the production evals arc that W3D7 advances

---

## Repository Structure

```
W3D7_distributed-tracing-langsmith/
├── README.md                                      # This file
├── docs/
│   ├── technical-document.md                      # 21-section practitioner deep-dive
│   └── distributed-tracing-layman-scenarios.md    # Business scenarios without ML background
├── diagrams/
│   ├── architecture.mmd                           # 5-span RAG pipeline trace architecture
│   └── sequence.mmd                               # Per-step span emission sequence
└── poc/
    ├── README.md                                  # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                                # Entry point — demo + live mode with @traceable
    │   ├── tracing_core.py                        # Pipeline functions + typed data structures
    │   └── config.py                              # Config dataclass + env loader
    ├── tests/
    │   └── test_tracing.py                        # 18 unit tests across 4 test classes (all offline)
    ├── requirements.txt
    ├── sample_input.json                          # Example query input
    └── sample_output.json                         # Expected output: 5 spans, 87 tokens, 18.4ms
```

---

## Core Concepts

### `RetrievedDocument` and `PipelineResult` — span output contracts

Typed dataclasses ensure every span emits structured, machine-readable output rather than raw strings. The `last_updated` field on `RetrievedDocument` enables stale-document detection directly in the trace explorer:

```python
# tracing_core.py
@dataclass
class RetrievedDocument:
    doc_id: str
    content: str
    score: float       # 0.0–1.0 relevance score — visible per-document in the span output
    source: str = ""   # filename for audit trail
    last_updated: str = ""   # ISO date — surface stale docs in the trace UI

@dataclass
class PipelineResult:
    run_id: str
    answer: str
    retrieved_docs: list[RetrievedDocument]
    tokens_used: int
    latency_ms: float
    model: str = "demo"
    spans_captured: int = 0   # how many child spans were emitted
```

### `@traceable` — per-step span emission

In live mode, the `@traceable` decorator wraps each pipeline step. The `run_type` field tells LangSmith how to render the span — `"retriever"` for document fetch, `"llm"` for generation, `"chain"` for everything else:

```python
# main.py (live mode)
from langsmith import traceable

@traceable(run_type="retriever", name="retrieve_documents")
def _retrieve(q: str):
    return retrieve_documents(q, top_k=3)

@traceable(run_type="chain", name="rerank_documents")
def _rerank(q: str, docs):
    return rerank_documents(q, docs)

@traceable(run_type="chain", name="assemble_context")
def _assemble(docs):
    return assemble_context(docs)

@traceable(run_type="llm", name="generate_answer")
def _generate(context: str, question: str) -> dict:
    resp = client.chat.completions.create(model=cfg.model, messages=[...])
    return {"answer": resp.choices[0].message.content, "tokens": resp.usage.total_tokens}

@traceable(run_type="chain", name="validate_answer")
def _validate(answer: str, docs):
    return validate_answer(answer, docs)
```

### `validate_answer()` — deterministic grounding check

A word-overlap heuristic checks that the answer does not introduce content absent from the retrieved documents. This is a separate span — validator bugs are visible independently of LLM generation bugs, a common source of confusion in production:

```python
# tracing_core.py
def validate_answer(answer: str, docs: list[RetrievedDocument]) -> dict:
    doc_words    = set(" ".join(d.content.lower() for d in docs).split())
    answer_words = set(answer.lower().split())
    overlap      = len(answer_words & doc_words) / max(len(answer_words), 1)
    return {
        "passed":        overlap >= 0.3,
        "overlap_ratio": round(overlap, 3),
        "reason":        "Sufficient overlap" if overlap >= 0.3 else "Answer diverges from context",
    }
```

### `run_demo_pipeline()` — offline span structure exploration

Demo mode chains all 5 steps with local mock documents and counts spans manually. The output structure is identical to live mode — the only difference is no API calls are made and `model` is `"demo"`:

```python
# tracing_core.py
def run_demo_pipeline(query: str) -> PipelineResult:
    run_id = str(uuid.uuid4())
    docs        = retrieve_documents(query, top_k=3);     spans += 1  # retriever span
    ranked_docs = rerank_documents(query, docs);           spans += 1  # reranker span
    context     = assemble_context(ranked_docs);           spans += 1  # assemble_context span
    answer      = "<pre-computed answer>";                 spans += 1  # llm_call span
    validation  = validate_answer(answer, ranked_docs);   spans += 1  # validate_answer span
    return PipelineResult(run_id=run_id, answer=answer, spans_captured=5, ...)
```

---

## Run the PoC

### Demo mode (no API keys required)

```bash
cd week-03/W3D7_distributed-tracing-langsmith/poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

### Live mode (full tracing in LangSmith)

```bash
export OPENAI_API_KEY=your-openai-api-key
export LANGCHAIN_API_KEY=your-langsmith-api-key
export LANGCHAIN_PROJECT=w3d7-distributed-tracing
export LANGCHAIN_TRACING_V2=true
python src/main.py
# Trace URL is printed — click to see the full 5-span run tree in LangSmith
```

### Tests

```bash
pytest tests/ -v
# 18 tests across 4 classes — all pass offline
```

---

## Expected Output

```
Distributed Tracing (LangSmith) Demo
=============================================

⚠️  Running in DEMO MODE — output is pre-computed (no API calls made)

Query: What is the return window for electronics, and how long do refunds take?

Answer:         Based on the return policy, standard items can be returned within
                30 days. Electronics must be returned within 15 days in original
                packaging. Refunds are processed within 5 business days.
Run ID:         3f8a1c2d-b47e-4d91-a830-e6c2f9d05a1b
Spans captured: 5
Tokens used:    87
Latency:        18.4 ms
Model:          demo

✅ Concept demonstrated: 5-span RAG pipeline with per-step input/output capture.
   In live mode, open LangSmith to see the full run tree.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — runs demo or live pipeline; applies `@traceable` decorators in live mode; prints run ID, span count, and trace URL |
| `src/tracing_core.py` | `RetrievedDocument`, `PipelineResult`, `DEMO_DOCUMENTS`, `retrieve_documents()`, `rerank_documents()`, `assemble_context()`, `validate_answer()`, `run_demo_pipeline()` |
| `src/config.py` | `Config` dataclass + `load_config()` with `tracing_enabled` flag (requires both API keys), LangSmith project name |
| `tests/test_tracing.py` | 18 tests across 4 classes: `TestDemoMode` (schema, answer, model, spans, run_id), `TestCoreConcept` (8 tests), `TestLiveMode` (2 mocked), `TestSampleFiles` (3 tests) |
| `sample_input.json` | Example query: electronics return window + refund timing |
| `sample_output.json` | Pre-computed `PipelineResult`: 5 spans, 87 tokens, 18.4ms, full span tree |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith key — required for span emission; tracing is a no-op without it |
| `LANGCHAIN_PROJECT` | `w3d7-distributed-tracing` | LangSmith project name shown in the trace explorer |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangChain automatic callback tracing |
| `MODEL` | `gpt-4o-mini` | OpenAI model for live mode generation |
| `DEMO_MODE` | `false` | Set `true` to force demo mode even if API keys are present |
| `TEMPERATURE` | `0.0` | LLM temperature — keep at 0 for deterministic answers |
| `MAX_TOKENS` | `500` | Max tokens per generation call |

Copy variable names into a `.env` file (never commit to source control) and load with `python-dotenv` or `export` in your shell.

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/distributed-tracing-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — 5-span RAG pipeline trace architecture
- [Sequence Diagram](diagrams/sequence.mmd) — Per-step span emission sequence

---

## Connection to the Series

**Previous:** [W3D6 — Hierarchical Subagent Teams](../W3D6_hierarchical-subagent-teams/README.md) — builds the pipeline that W3D7 instruments; you cannot debug what you cannot observe.

**Series arc:** [W1D7 — LLM-as-a-Judge Evals](../../week-01/W1D7-llm-as-a-judge/README.md) introduced offline evaluation using an LLM to score outputs. [W2D7 — Deterministic Guardrails (NeMo)](../../week-02/W2D7_deterministic-guardrails-nemo/README.md) added rule-based safety enforcement at the LLM boundary. W3D7 closes the Production Evals & Guardrails vertical for Week 3: once you can evaluate quality offline and enforce safety rules online, the final capability is observability — seeing what every step of every live run actually received and returned, so regressions are root-caused in minutes, not days.

**Week 3 complete.** All 7 advanced techniques are now covered. [Return to Week 3 overview →](../README.md)

---

## Key References

- [LangSmith documentation](https://docs.smith.langchain.com/tracing)
- [LangSmith `@traceable` decorator reference](https://docs.smith.langchain.com/tracing/integrations/python)
- [OpenAI token usage and pricing](https://openai.com/api/pricing/)

---

## Continue Learning

**Week 3 complete.** Return to the [Week 3 overview](../README.md) or go to the [full series roadmap](../../README.md) to see what's coming in Week 4 — Production & Scale.
