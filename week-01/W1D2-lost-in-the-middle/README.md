# W1D2 — "Lost in the Middle" Context Position Decay

> Week 1, Day 2 | Vertical: Context Engineering & Tokens  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 1](../README.md)

---

## Overview

LLMs do not read a context window uniformly. Transformer attention follows a **U-shaped distribution** — the model pays the most attention to tokens at the start and end of the context, and significantly less to tokens in the middle. When a RAG pipeline places the most relevant documents at middle positions (as naive retrieval-rank ordering often does), the model silently ignores them and generates answers from less-relevant content at the edges.

**Research result (Liu et al., 2023):** Multi-document QA accuracy with GPT-3.5-Turbo drops from **71% when the answer document is at position 1** to **45% when buried at position 10 of 20** — a **26-point gap caused entirely by document position**, not content quality.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Explain the U-shaped attention distribution pattern and why it emerges in transformer architectures
2. Identify production failure modes caused by position-insensitive context assembly
3. Implement three document ordering strategies: naive, relevance-sorted, and LiTM-aware interleaving
4. Evaluate the effectiveness of each ordering strategy using simulated attention-weighted scoring
5. Distinguish between primacy bias, recency bias, and the composite Lost-in-the-Middle effect
6. Design a context budget strategy that combines token counting with position-aware ordering

---

## Problem Statement

Modern RAG pipelines retrieve the top-K documents from a vector store, concatenate them, and feed the assembled context to an LLM. The naive implementation assumes the model reads context uniformly — a reasonable assumption for humans, but empirically false for transformers.

**What breaks:** A customer support bot retrieves 8 documents for a query about a payment failure. The most relevant document (a known bug description) is at retrieval rank 4, placing it in the middle of the assembled context. The LLM generates a generic response because it effectively skips the most diagnostic document.

**Production failure modes:**

- **Silent accuracy degradation** — The model does not error; it answers from less-relevant context at the edges, producing plausible but incorrect responses
- **Wasted token spend** — Token budget is consumed by low-relevance documents in the primacy zone while high-relevance documents land in the dead zone
- **Non-reproducible failures** — The same query produces different quality answers depending on retrieval order, with document position as an invisible confound

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode
- For live mode: an OpenAI API key set in `.env`

---

## Repository Structure

```
W1D2-lost-in-the-middle/
├── README.md                        # This file
├── docs/
│   ├── technical-document.md        # Full practitioner deep-dive
│   └── litm-layman-scenarios.md     # Four business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd             # Context assembly pipeline (Mermaid)
│   └── sequence.mmd                 # Naive vs. LiTM-aware ordering flow (Mermaid)
└── poc/
    ├── README.md                    # Quick-start and expected output
    ├── src/
    │   ├── main.py                  # Entry point — runs all three strategies and compares them
    │   ├── lost_in_middle_core.py   # U-shaped attention model, three ordering strategies, scoring
    │   └── config.py                # Config loaded from environment variables
    ├── tests/
    │   └── test_lost_in_middle.py   # 8 pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json            # 6 retrieved documents with relevance scores
    └── sample_output.json           # Expected orderings and effective scores per strategy
```

---

## Core Concept: The Three Ordering Strategies

The PoC compares three approaches to assembling a context window from retrieved documents:

| Strategy | What It Does | Key Weakness |
|---|---|---|
| **Naive** | Keeps retrieval order unchanged | High-relevance docs end up in the middle dead zone |
| **Relevance-sorted** | Most relevant doc at position 0 | 2nd-best at position 1, rest still decay toward the middle |
| **LiTM-aware** | Alternates top docs to edges (1st→pos 0, 2nd→pos N-1, 3rd→pos 1…) | None — maximises use of the U-shaped attention curve |

**Attention weight model** (`u_shaped_attention_weight`):

```
weight = 0.4 + 0.6 × |cos(π × i / (N-1))|
```

This gives `weight = 1.0` at positions 0 and N-1, and approximately `0.4` at the midpoint — matching the empirical pattern from Liu et al. (2023).

**Effective score** = `relevance_score × attention_weight`

This is the per-document proxy metric: how much of a document's relevance the LLM will actually utilise given its position.

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
# Expected: 8 passed, 0 failed (no API key needed)
```

---

## Expected Output

```
⚠️  Running in demo mode (no API key required).
🚀 Lost in the Middle Demo
==============================================

Input: 6 docs | Query: 'Why does checkout fail on mobile Safari?'

  Strategy 1: Naive (retrieval order)
  Pos  ID       Relevance  Attention  Effective
  ---  -------  ---------  ---------  ---------
    0  doc_1         0.10     1.0000     0.1000
    1  doc_2         0.15     0.8854     0.1328
    2  doc_3         0.92     0.5854     0.5386  ← dead zone
    3  doc_4         0.88     0.5854     0.5152  ← dead zone
    4  doc_5         0.20     0.8854     0.1771
    5  doc_6         0.75     1.0000     0.7500
  → mean=0.3689  min=0.1000

  Strategy 2: Relevance-sorted (best first)
  ...
  → mean=0.4147  min=0.0585

  Strategy 3: LiTM-aware (best at edges)
  ...
  → mean=0.4646  min=0.0585

📊 Naive=0.3689 | Sorted=0.4147 | LiTM=0.4646

✅ Concept demonstrated: LiTM-aware ordering improves mean effective score by 26.0%.
   High-relevance docs now occupy positions 0 and N-1 where attention peaks.
```

The key observation: `doc_3` (relevance 0.92) and `doc_4` (relevance 0.88) land at positions 2 and 3 in naive order — the dead zone where attention drops to ~0.58. LiTM-aware ordering moves them to positions 0 and 5 where attention is 1.0, recovering 26% of mean effective score.

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — runs all three ordering strategies and prints a comparison table |
| `src/lost_in_middle_core.py` | `Document` dataclass, `u_shaped_attention_weight`, three orderings, `compute_effective_scores`, `summarise_effectiveness` |
| `src/config.py` | `load_config()` — reads `OPENAI_API_KEY`, `MODEL`, `DEMO_MODE`, `CONTEXT_BUDGET_TOKENS` from environment |
| `tests/test_lost_in_middle.py` | 8 pytest tests: attention weight shape, ordering invariants, LiTM improvement guarantee, effective score bounds |
| `sample_input.json` | 6 documents with pre-assigned relevance scores for the mobile Safari checkout query |
| `sample_output.json` | Ground-truth orderings and effective scores for all three strategies, including improvement percentages |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL=gpt-4o-mini
TEMPERATURE=0.0
MAX_TOKENS=1000
CONTEXT_BUDGET_TOKENS=4096   # Max tokens allocated for retrieved documents
DEMO_MODE=false               # Set to true to run without an API key
```

Demo mode is automatically enabled when no `OPENAI_API_KEY` is present.

---

## Technical Documentation

The full practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- Transformer attention mechanics and why the U-shaped curve emerges
- Three ordering strategies with worked examples
- LangChain `ContextualCompressionRetriever` and LlamaIndex `SentenceTransformerRerank` integration
- Context budget management combining token counting with position-aware ordering
- Security considerations (OWASP LLM06, LLM09)
- Cost analysis and production checklist
- Interview questions from Conceptual through Architecture and Production

For a jargon-free walkthrough using four business scenarios, see [`docs/litm-layman-scenarios.md`](docs/litm-layman-scenarios.md).

---

## Architecture Diagrams

Mermaid source files are in [`diagrams/`](diagrams/):

- [`architecture.mmd`](diagrams/architecture.mmd) — Context assembly pipeline: retrieval → ordering → token budget → LLM
- [`sequence.mmd`](diagrams/sequence.mmd) — Naive vs. LiTM-aware ordering flow with position assignment

---

## Connection to the Series

- **Yesterday — W1D1 DSPy & Programmatic Prompts:** DSPy showed how to compile structured, type-safe prompts. But a well-structured prompt still fails if the context fed to it is assembled naively.
- **Today — W1D2 Lost in the Middle:** Position-aware context assembly ensures the documents most relevant to the query occupy the positions where the LLM pays the most attention.
- **Tomorrow — W1D3 Naive vs. Agentic RAG:** Even optimal context ordering is not enough for complex multi-step queries. Tomorrow we explore how agent-driven retrieval replaces static context assembly entirely.

---

## Key Reference

Liu, N. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."  
*Transactions of the Association for Computational Linguistics*, 12.  
arXiv:2307.03172 — https://arxiv.org/abs/2307.03172

---

## Continue Learning

**Next:** W1D3 — Naive vs. Agentic RAG — Why static top-K retrieval breaks on multi-step queries, and how agent-driven retrieval loops fix it.

**Series index:** [Week 1 Overview](../README.md) | [Full Roadmap](../../README.md)
