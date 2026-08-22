# W3D7 — Distributed Tracing (LangSmith)

**Series:** AI Engineering Production Playbook
**Vertical:** Production Evals & Guardrails
**Week 3 / Day 7**

---

## What This Demonstrates

A 5-step RAG pipeline (retrieve → rerank → assemble → generate → validate) where every step emits a structured span to LangSmith. The full run tree — with inputs, outputs, token counts, and latency at each node — is visible in the LangSmith trace explorer. An automated validator attaches a grounding score to the root span after generation.

In demo mode the pipeline runs entirely offline using pre-computed mock data. No API key is needed to explore the span structure and output schema.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode available without it)
- LangSmith API key (optional — tracing is a no-op without it)

---

## Quickstart

```bash
# 1. Navigate to this folder
cd week-03/W3D7_distributed-tracing-langsmith/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run in demo mode (no API keys required)
DEMO_MODE=true python src/main.py
```

Expected output:

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

## Live Mode (Full Tracing)

To run with real API calls and see the trace tree in LangSmith:

```bash
# Set environment variables
export OPENAI_API_KEY=your-openai-api-key
export LANGCHAIN_API_KEY=your-langsmith-api-key
export LANGCHAIN_PROJECT=w3d7-distributed-tracing
export LANGCHAIN_TRACING_V2=true

# Run
python src/main.py
```

The trace URL will be printed after the run. Click it to see all 5 spans in the LangSmith UI.

---

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — external API calls are mocked via `unittest.mock`.

```
tests/test_tracing.py::TestDemoMode::test_demo_pipeline_returns_valid_schema   PASSED
tests/test_tracing.py::TestDemoMode::test_demo_pipeline_answer_is_not_empty    PASSED
tests/test_tracing.py::TestDemoMode::test_demo_pipeline_model_is_demo          PASSED
tests/test_tracing.py::TestDemoMode::test_demo_pipeline_captures_five_spans    PASSED
tests/test_tracing.py::TestDemoMode::test_demo_pipeline_run_id_is_uuid_format  PASSED
tests/test_tracing.py::TestCoreConcept::...                                     PASSED (8 tests)
tests/test_tracing.py::TestLiveMode::...                                        PASSED (2 tests)
tests/test_tracing.py::TestSampleFiles::...                                     PASSED (3 tests)
```

---

## File Structure

```
poc/
├── src/
│   ├── main.py            Entry point — run this file
│   ├── tracing_core.py    Core pipeline functions and data structures
│   └── config.py          Config dataclass + environment variable loader
├── tests/
│   └── test_tracing.py    18 pytest unit tests across 4 test classes
├── README.md              This file
├── requirements.txt       Pinned Python dependencies
├── sample_input.json      Example query input
└── sample_output.json     Expected output structure with span tree
```

---

## Key Concepts in the Code

| File | What it shows |
|---|---|
| `tracing_core.py` | `RetrievedDocument` and `PipelineResult` dataclasses; the 5 pipeline step functions; `run_demo_pipeline()` that chains them |
| `main.py` | `@traceable` decorator applied to each step in live mode; `run_demo()` for offline use; configuration via `load_config()` |
| `config.py` | `tracing_enabled` flag — True only when both `OPENAI_API_KEY` and `LANGCHAIN_API_KEY` are present |
| `test_tracing.py` | `TestDemoMode`, `TestCoreConcept`, `TestLiveMode`, `TestSampleFiles` — all four required test classes |

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Live mode only | `""` | OpenAI API authentication |
| `LANGCHAIN_API_KEY` | Tracing only | `""` | LangSmith API authentication |
| `LANGCHAIN_PROJECT` | No | `w3d7-distributed-tracing` | LangSmith project name |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangChain automatic callback tracing |
| `MODEL` | No | `gpt-4o-mini` | OpenAI model to use in live mode |
| `DEMO_MODE` | No | `false` | Force demo mode even if API key is present |

Copy the variable names above into a `.env` file (not committed to git) and load with `python-dotenv` or `export` them in your shell.

---

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Layman Scenarios](../docs/distributed-tracing-layman-scenarios.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [LinkedIn Post](../README.md)
- [LangSmith Documentation](https://docs.smith.langchain.com/tracing)
