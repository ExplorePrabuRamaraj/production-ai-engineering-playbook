# W1D2 — "Lost in the Middle" Context Position Decay

**Series:** AI Engineering Production Playbook
**Vertical:** Context Engineering & Tokens
**Week 1 / Day 2**

## What This Demonstrates

How transformer attention follows a U-shaped distribution across context positions — causing high-relevance documents placed in the middle of a long context to be effectively ignored — and how position-aware document ordering mitigates this silent accuracy tax.

## Quick Start

```bash
# 1. Navigate to the PoC directory
cd artifacts/W1D2_lost-in-the-middle/03_poc-code

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo (no API key required)
python src/main.py

# 4. Run tests
pytest tests/ -v
```

## Expected Demo Output

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

  Strategy 3: LiTM-aware (best at edges)
  ...
  → mean=0.4646  min=0.0585

📊 Naive=0.3689 | Sorted=0.4147 | LiTM=0.4646

✅ Concept demonstrated: LiTM-aware ordering improves mean effective score by 26.0%.
   High-relevance docs now occupy positions 0 and N-1 where attention peaks.
```

## Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — runs all three ordering strategies and compares them |
| `src/lost_in_middle_core.py` | Core logic: U-shaped attention model, three ordering strategies, scoring |
| `src/config.py` | Configuration via environment variables with `load_config()` |
| `tests/test_lost_in_middle.py` | pytest unit tests (8 tests, all run offline) |
| `sample_input.json` | 6 retrieved documents with relevance scores |
| `sample_output.json` | Expected ordering and effective scores for each strategy |

## Running Tests

```bash
pytest tests/ -v
# Expected: 8 passed, 0 failed
```

## Connection to the Series

- **Yesterday — W1D1 DSPy & Programmatic Prompts:** DSPy showed us how to compile structured, type-safe prompts. But a well-structured prompt still fails if the context fed to it is assembled naively.
- **Today — W1D2 Lost in the Middle:** Position-aware context assembly ensures the documents most relevant to the query occupy the positions where the LLM pays the most attention.
- **Tomorrow — W1D3 Naive vs. Agentic RAG:** Even optimal context ordering is not enough for complex multi-step queries. Tomorrow we explore how agent-driven retrieval replaces static context assembly entirely.

## Key Reference

Liu, N. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts."
*Transactions of the Association for Computational Linguistics*, 12.
arXiv:2307.03172 — https://arxiv.org/abs/2307.03172
