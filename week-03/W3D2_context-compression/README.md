# W3D2 — Context Compression

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

As conversations grow, context windows fill with low-signal turns — early pleasantries, superseded instructions, and verbose tool outputs — leaving less room for the current query and recent context. Naive sliding-window eviction discards the oldest turns, but the oldest turns often contain the most important setup context. Query-aware context compression solves both problems: it reduces input tokens by 40–80% using extractive (TF-IDF sentence scoring) or abstractive (LLM summarisation) strategies, conditioned on the current query, so what is retained is exactly what the model needs to answer.

---

## Learning Objectives

1. Understand why naive sliding-window eviction loses critical setup context and why query-aware compression is the correct alternative
2. Implement `extractive_compress()` — TF-IDF cosine similarity scoring that retains the highest-signal sentences within a token budget
3. Implement `abstractive_compress()` — LLM-based summarisation conditioned on the query, preserving named entities, dates, and decisions
4. Implement `compress_context()` — the unified interface that dispatches to `extractive`, `abstractive`, or `hybrid` strategy based on configuration
5. Configure `min_segment_tokens` to bypass compression for segments already within budget or too small to benefit
6. Measure compression ratio per segment and aggregate across multi-segment inputs (conversation history, retrieved docs, tool outputs)
7. Know when to use each strategy: extractive for documents and tool outputs (no API call), abstractive for conversation history, hybrid for mixed segment types

---

## Problem Statement

At 200,000 calls/month with a typical 3-segment context (history + docs + tool output), uncompressed inputs of ~450 tokens per call represent avoidable cost — and the problem compounds as conversations grow. After 20 turns, the context window may be 80% full of early pleasantries and overridden instructions that have no bearing on the current query. Naive sliding-window eviction removes the oldest turns, but the system prompt and early setup turns are often the most load-bearing. Without a principled way to decide which tokens to keep, engineers either over-truncate (losing important context, causing regressions) or don't truncate at all (hitting the context limit and failing). Query-aware context compression makes the trade-off explicit: TF-IDF sentence scoring ranks each sentence by relevance to the current query, and only the highest-signal sentences within the token budget are retained.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — all demos run offline with `DEMO_MODE=true`)
- Familiarity with [W1D2 — "Lost in the Middle" Decay](../../week-01/W1D2-lost-in-the-middle/README.md) provides useful context on why position in context matters
- Familiarity with [W3D1 — Prompt Distillation](../W3D1_prompt-distillation/README.md) useful — both tackle token reduction, but distillation targets the system prompt, compression targets runtime context

---

## Repository Structure

```
W3D2_context-compression/
├── README.md                                    # This file
├── docs/
│   ├── technical-document.md                    # 21-section practitioner deep-dive
│   └── context-compression-layman-scenarios.md  # Business scenarios without ML background
├── diagrams/
│   ├── architecture.mmd                         # Multi-strategy compression pipeline
│   └── sequence.mmd                             # Per-segment compress_context() flow
└── poc/
    ├── README.md                                # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                              # Entry point — demo + live mode
    │   ├── context_compression_core.py          # All compression logic (pure functions)
    │   └── config.py                            # Config dataclass + env loader
    ├── tests/
    │   └── test_context_compression.py          # pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                        # 3-segment input: history, docs, tool_output
    └── sample_output.json                       # Pre-computed demo result: 55% token reduction
```

---

## Core Concepts

### TF-IDF extractive compression

`extractive_compress()` scores each sentence by cosine similarity to the query using term-frequency vectors, then greedily retains the highest-scoring sentences until the token budget is filled. Retained sentences are reassembled in original document order to preserve narrative flow:

```python
# context_compression_core.py
def extractive_compress(text: str, query: str, token_budget: int) -> CompressionResult:
    sentences = split_sentences(text)
    query_tf = _term_frequencies(query)

    scored = [(cosine_similarity(_term_frequencies(s), query_tf), idx, s)
              for idx, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected, used = set(), 0
    for score, idx, sentence in scored:
        tok = estimate_tokens(sentence)
        if used + tok <= token_budget:
            selected.add(idx); used += tok

    retained = [sentences[i] for i in sorted(selected)]
    return CompressionResult(compressed_text=" ".join(retained), ...)
```

### Abstractive compression

`abstractive_compress()` instructs an LLM to produce a condensed, query-conditioned summary. A system prompt directs the model to preserve named entities, numerical values, dates, and decisions while omitting pleasantries and off-topic content. Falls back to extractive in demo mode when no API client is provided:

```python
# context_compression_core.py
def abstractive_compress(text, query, token_budget, openai_client=None, model="gpt-4o-mini"):
    if openai_client is None:                          # demo/offline fallback
        result = extractive_compress(text, query, token_budget)
        result.strategy_used = "abstractive-demo-fallback"
        return result

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Preserve named entities, numbers, dates, decisions. Omit pleasantries and repetitions."},
            {"role": "user", "content": f"Summarise for this query: {query}\n\nText:\n{text}\n\nMax {token_budget} tokens."},
        ],
        temperature=0.0, max_tokens=token_budget,
    )
    compressed_text = response.choices[0].message.content.strip()
    return CompressionResult(compressed_text=compressed_text, strategy_used="abstractive", ...)
```

### Unified `compress_context()` interface

The unified interface handles bypass logic (segment already within budget or below the minimum size threshold) and dispatches to the correct strategy. Hybrid runs extractive first and only calls the LLM if the result is still over budget:

```python
# context_compression_core.py
def compress_context(text, query, token_budget, strategy="extractive",
                     openai_client=None, model="gpt-4o-mini",
                     min_segment_tokens=50) -> CompressionResult:
    original_tokens = estimate_tokens(text)

    # Bypass: already within budget or too small to compress
    if original_tokens <= token_budget or original_tokens < min_segment_tokens:
        return CompressionResult(..., strategy_used="bypass")

    if strategy == "abstractive":
        return abstractive_compress(text, query, token_budget, openai_client, model)

    if strategy == "hybrid":
        result = extractive_compress(text, query, token_budget)    # free pass first
        if result.compressed_tokens > token_budget and openai_client:
            result = abstractive_compress(result.compressed_text, query, token_budget, openai_client, model)
            result.strategy_used = "hybrid"
        return result

    return extractive_compress(text, query, token_budget)           # default
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-03/W3D2_context-compression/poc
pip install -r requirements.txt
cp .env.example .env
DEMO_MODE=true python src/main.py
```

### Live mode

```bash
# Edit .env and set OPENAI_API_KEY=your-key
python src/main.py
```

### Change compression strategy

```bash
COMPRESSION_STRATEGY=abstractive python src/main.py
COMPRESSION_STRATEGY=hybrid python src/main.py
```

### Tests

```bash
pytest tests/ -v
# All tests pass offline — external API calls are mocked via unittest.mock
```

---

## Expected Output

```
[W3D2] Context Compression Demo
==================================================
Query: What is the refund policy for annual subscriptions?

Segments to compress: ['history', 'docs', 'tool_output']

[DEMO MODE] No API call made — output is pre-computed.

Results:
{
  "query": "What is the refund policy for annual subscriptions?",
  "segments": {
    "history":     { "original_tokens": 163, "compressed_tokens": 47,  "compression_ratio": 0.288, "strategy_used": "extractive" },
    "docs":        { "original_tokens": 228, "compressed_tokens": 92,  "compression_ratio": 0.404, "strategy_used": "extractive" },
    "tool_output": { "original_tokens":  63, "compressed_tokens": 63,  "compression_ratio": 1.0,   "strategy_used": "bypass"     }
  },
  "total_original_tokens": 454,
  "total_compressed_tokens": 202,
  "overall_compression_ratio": 0.445,
  "model": "demo"
}

[DONE] Concept demonstrated: 55% token reduction via query-aware context compression.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads config, runs demo or live mode, prints per-segment and aggregate compression metrics |
| `src/context_compression_core.py` | `estimate_tokens()`, `split_sentences()`, `extractive_compress()`, `abstractive_compress()`, `compress_context()`, `CompressionResult` |
| `src/config.py` | `Config` dataclass + `load_config()` with token budget, strategy, and segment-size thresholds |
| `tests/test_context_compression.py` | Unit tests for all strategies: extractive, abstractive (mocked), hybrid, bypass |
| `sample_input.json` | 3-segment input: conversation history, policy document, tool output JSON |
| `sample_output.json` | Pre-computed demo result: 454 → 202 tokens, 55% reduction across 3 segments |
| `.env.example` | All environment variable defaults |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `MODEL` | `gpt-4o-mini` | Main LLM model |
| `COMPRESSION_MODEL` | `gpt-4o-mini` | Model for abstractive summarisation (can be cheaper) |
| `TOKEN_BUDGET` | `1000` | Total token budget distributed across all segments |
| `COMPRESSION_STRATEGY` | `extractive` | `extractive`, `abstractive`, or `hybrid` |
| `MIN_SEGMENT_TOKENS` | `50` | Segments below this token count bypass compression entirely |
| `DEMO_MODE` | `false` | Set `true` to run without an API key (extractive only) |
| `TEMPERATURE` | `0.0` | LLM temperature — keep at 0 for deterministic summaries |
| `MAX_TOKENS` | `1000` | Max tokens per LLM response |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/context-compression-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — Multi-strategy compression pipeline
- [Sequence Diagram](diagrams/sequence.mmd) — Per-segment `compress_context()` flow

---

## Connection to the Series

**Previous:** [W3D1 — Prompt Distillation](../W3D1_prompt-distillation/README.md) — reduces tokens in the static system prompt using a greedy pruning loop; W3D2 extends the same token-reduction discipline to runtime context.

**Next:** [W3D3 — Hybrid Search & Reranking](../W3D3_hybrid-search-reranking/README.md) — improves retrieval precision for entity-rich queries by fusing dense and sparse search, then reranking the merged list.

**Series arc:** [W1D2 — "Lost in the Middle" Decay](../../week-01/W1D2-lost-in-the-middle/README.md) established that position in context affects model accuracy. [W2D2 — KV Caching & Token Trimming](../../week-02/W2D2_kv-caching-token-trimming/README.md) covered token trimming and cache reuse. W3D2 closes the context engineering loop: rather than trimming blindly or caching the raw context, compress each segment down to only what the current query needs — preserving the signal while cutting the token cost.

---

## Key References

- Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172
- Jiang et al. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." arXiv:2310.05736
- [LangChain ContextualCompressionRetriever](https://python.langchain.com/docs/how_to/contextual_compression/)

---

## Continue Learning

**Next:** [W3D3 — Hybrid Search & Reranking](../W3D3_hybrid-search-reranking/README.md)

Return to [Week 3 overview](../README.md) to explore all advanced techniques.
