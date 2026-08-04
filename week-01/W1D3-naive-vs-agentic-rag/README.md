# W1D3 — Naive vs. Agentic RAG

> Week 1, Day 3 | Vertical: Advanced RAG  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 1](../README.md)

---

## Overview

Naive RAG treats retrieval as a fixed preprocessing step: embed the query, fetch the top-k chunks by cosine similarity, inject them into the prompt, and call the LLM once. This works for simple factual lookups but breaks on multi-hop questions where the answer is distributed across multiple documents — the single retrieval pass fetches the most similar chunks by surface form, not by the sub-questions that actually need answering.

**Agentic RAG** replaces the fixed pipeline with a planning loop: decompose the query into atomic sub-questions, retrieve targeted evidence per sub-question, validate each result against a similarity threshold, reformulate and retry on failure, then synthesise a final answer with citations only from verified evidence.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Explain why naive RAG fails on multi-hop and ambiguous queries, with concrete failure examples
2. Distinguish the architectural difference between retrieval-as-preprocessing and retrieval-as-tool-call
3. Implement the decompose → retrieve → validate → reformulate → synthesise loop
4. Evaluate the latency and cost trade-offs of adding reasoning steps to the retrieval pipeline
5. Design an evidence-validation step that prevents hallucination when retrieved chunks are insufficient
6. Build a fallback path that degrades gracefully when retrieval returns low-confidence chunks

---

## Problem Statement

Naive RAG has a structural flaw: it retrieves by surface similarity and has no awareness of whether it found the right information.

**What breaks:** A customer support bot receives: *"My order was marked delivered but I never received it — what do I do and does my Gold membership change my options?"* Naive RAG issues a single retrieval call, gets the standard refund policy chunk at position 0, and generates an answer that completely ignores the membership-specific context. The Gold member gets a generic response.

**Production failure modes:**
- **Multi-hop blindness** — single-pass retrieval cannot answer questions that require joining evidence from two separate documents
- **Silent incompleteness** — the model generates a confident-sounding answer from partial evidence with no indication that the other half of the query was not addressed
- **No retry path** — if the top-k chunks are below the relevance threshold, naive RAG uses them anyway; agentic RAG reformulates and retries

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode
- For live mode: an OpenAI API key set in `.env`

---

## Repository Structure

```
W1D3-naive-vs-agentic-rag/
├── README.md                      # This file
├── docs/
│   ├── technical-document.md      # Full practitioner deep-dive
│   └── rag-layman-scenarios.md    # Four business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd           # Naive vs. agentic pipeline architecture (Mermaid)
│   └── sequence.mmd               # Decompose → retrieve → validate → synthesise flow (Mermaid)
└── poc/
    ├── README.md                  # Quick-start and expected output
    ├── src/
    │   ├── main.py                # Entry point — runs both pipelines and prints comparison
    │   ├── rag_core.py            # QueryDecomposer, ChunkRetriever, NaiveRAGPipeline, AgenticRAGPipeline
    │   └── config.py              # Config loaded from environment variables
    ├── tests/
    │   └── test_rag_core.py       # 30 pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json          # Demo query
    └── sample_output.json         # Expected output for both pipelines
```

---

## Core Concept: The Two Pipelines

### Naive RAG

```
Query → single retrieval call → top-k chunks → LLM → answer
```

One retrieval call, no decomposition, no validation. Fast and cheap — until the query is multi-hop.

### Agentic RAG

```
Query → QueryDecomposer → [sub-Q1, sub-Q2, ...]
           ↓ per sub-question
     ChunkRetriever.retrieve()
           ↓ if threshold not met
     reformulate → retry (up to max_reformulation_retries)
           ↓ validated evidence
     synthesise with citations → answer
```

One retrieval call **per sub-question**, with threshold validation and retry on each.

### Side-by-side on a multi-hop query

| Metric | Naive RAG | Agentic RAG |
|---|---|---|
| Sub-questions | 1 (raw query) | 2 (decomposed) |
| Retrieval calls | 1 | 2+ |
| Chunks used | 1 | 2 |
| Membership context | Missing | Covered |
| Citations | Yes | Yes (per sub-question) |

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
# Expected: 30 passed, 0 failed (no API key needed)
```

---

## Expected Output

```
RAG Pipeline Demo — Naive vs. Agentic
==========================================
Running in demo mode (no API key). Output uses pre-computed corpus.

Query: My order was marked delivered but I never received it — what do I do
       and does my Gold membership change my options?

--- Naive RAG ---
Sub-questions : ['My order was marked delivered but I never received it — ...']
Retrieval calls: 1  |  Chunks used: 1
Answer:
  Standard members must report missing or undelivered orders within 7 days
  of the expected delivery date to qualify for a replacement or refund. [1]

--- Agentic RAG ---
Sub-questions : ['My order was marked delivered but I never received it — what do I do?',
                 'Does my Gold membership change my options?']
Retrieval calls: 2  |  Chunks used: 2
Answer:
  Standard members must report missing or undelivered orders within 7 days
  of the expected delivery date to qualify for a replacement or refund. [1]
  Gold members receive an extended 30-day dispute window for lost or undelivered
  orders and qualify for expedited replacement shipping at no additional charge. [2]

==========================================
Concept demonstrated: Agentic RAG decomposes multi-hop queries and
retrieves targeted evidence per sub-question; naive RAG retrieves
by surface similarity alone and misses membership-specific context.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads query from `sample_input.json`, runs both pipelines, prints comparison, writes `sample_output.json` |
| `src/rag_core.py` | `QueryDecomposer` (heuristic/LLM decomposition), `ChunkRetriever` (similarity search + threshold filter), `AgenticRAGPipeline` (full loop), `NaiveRAGPipeline` (single-pass baseline) |
| `src/config.py` | `load_config()` — reads all env vars including `TOP_K`, `SIMILARITY_THRESHOLD`, `MAX_SUB_QUESTIONS`, `MAX_REFORMULATION_RETRIES` |
| `tests/test_rag_core.py` | 30 tests across 6 classes: `TestDemoSimilarity`, `TestQueryDecomposer`, `TestChunkRetriever`, `TestNaiveRAGPipeline`, `TestAgenticRAGPipeline`, `TestNaiveVsAgenticComparison` |
| `sample_input.json` | Multi-hop demo query: missing order + Gold membership options |
| `sample_output.json` | Both pipeline outputs with sub-questions, retrieval call counts, and answers |

---

## Configuration

Copy `.env.example` to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL=gpt-4o-mini                  # Synthesis model
DECOMPOSITION_MODEL=gpt-4o-mini   # Smaller/cheaper model for decomposition step
TEMPERATURE=0.0
MAX_TOKENS=1000
TOP_K=5                            # Chunks fetched per retrieval call
SIMILARITY_THRESHOLD=0.70          # Minimum score for a chunk to be used as evidence
MAX_SUB_QUESTIONS=4                # Cap on decomposed sub-questions
MAX_REFORMULATION_RETRIES=2        # Retry attempts when retrieval returns empty
DEMO_MODE=false                    # Set to true to run without an API key
```

Demo mode activates automatically when no `OPENAI_API_KEY` is present.

---

## Extending to a Real Vector Store

Live mode retrieval is a stub in `ChunkRetriever._live_retrieve()`. To connect a real vector store:

1. Set `OPENAI_API_KEY` and `DEMO_MODE=false` in `.env`
2. Uncomment the relevant client in `requirements.txt` (`faiss-cpu`, `pinecone-client`, or `weaviate-client`)
3. Replace the `NotImplementedError` stub in `_live_retrieve()` with your vector store query

The decomposition and synthesis steps call OpenAI automatically once demo mode is off.

---

## Technical Documentation

The full practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- Why cosine similarity alone is insufficient for multi-hop retrieval
- Query decomposition strategies and when to use LLM-based vs. rule-based decomposition
- Evidence validation patterns and confidence scoring
- Latency and cost trade-offs: naive RAG vs. agentic RAG at 1k / 10k / 100k requests per day
- LangGraph and LlamaIndex Sub Question Query Engine integration
- 8 learning objectives, security considerations, and a production checklist

For a jargon-free walkthrough, see [`docs/rag-layman-scenarios.md`](docs/rag-layman-scenarios.md).

---

## Architecture Diagrams

Mermaid source files are in [`diagrams/`](diagrams/):

- [`architecture.mmd`](diagrams/architecture.mmd) — Naive pipeline (3 steps) vs. agentic pipeline (5-step loop) side by side
- [`sequence.mmd`](diagrams/sequence.mmd) — Decompose → retrieve → validate → reformulate → synthesise sequence with retry path

---

## Connection to the Series

- **W1D1 — DSPy & Programmatic Prompts:** Typed, compiled prompt programs replace hand-crafted strings.
- **W1D2 — Lost in the Middle:** Position-aware context assembly ensures retrieved documents are placed where attention is highest.
- **Today — W1D3 Naive vs. Agentic RAG:** Even perfectly ordered context is insufficient for multi-hop queries. Agentic retrieval replaces static context assembly with an iterative planning loop.
- **Tomorrow — W1D4 Model Context Protocol (MCP):** With agentic retrieval in place, MCP formalises how the agent discovers and invokes external tools beyond document retrieval.

---

## Key References

- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv:2005.11401
- Asai, A. et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." arXiv:2310.11511
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LlamaIndex Sub Question Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/)

---

## Continue Learning

**Next:** W1D4 — Model Context Protocol (MCP) — How MCP formalises the tool contract between an LLM agent and external systems with a typed, versioned interface.

**Series index:** [Week 1 Overview](../README.md) | [Full Roadmap](../../README.md)
