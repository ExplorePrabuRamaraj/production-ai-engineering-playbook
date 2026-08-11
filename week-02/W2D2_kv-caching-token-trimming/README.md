# W2D2 — KV Caching & Token Trimming

> Week 2, Day 2 | Vertical: Context Engineering & Tokens  
> Part of the [Production AI Engineering Playbook](../../README.md) — [Week 2](../README.md)

---

## Overview

Every LLM call recomputes attention over the full context window. A 50-turn conversation that naively appends all history sees latency spike 40–60% per request and costs grow proportionally — you are paying to re-read the same system prompt and early conversation turns on every single call.

The fix is two-layered. **Server-side KV caching** (Anthropic `cache_control`, OpenAI automatic prefix caching) reuses previously computed attention weights for static prefixes — the system prompt and few-shot examples that never change between requests. **Client-side token trimming** enforces a hard token budget on the dynamic conversation suffix: the oldest turns are evicted at message boundaries, tool call / tool result pairs are evicted atomically, and when eviction exceeds 50% of history, a summary is compressed and injected in place of the dropped turns.

The PoC implements the full client-side pipeline as a pure Python library (`kv_caching_core.py`) and shows the exact API headers needed to activate server-side caching on both Anthropic and OpenAI — all runnable offline without an API key.

---

## Learning Objectives

By the end of this lesson, you will be able to:

1. Explain the difference between server-side KV cache (attention weight reuse) and client-side token trimming (budget enforcement before the API call)
2. Count tokens exactly using tiktoken — not estimated — before every API call
3. Implement sliding window eviction that trims at message boundaries and preserves system messages
4. Evict tool call / tool result pairs atomically to prevent orphaned messages that break the conversation schema
5. Apply summary compression for large eviction events and inject it as a synthetic system message
6. Add `cache_control` headers to static system prompt blocks for Anthropic KV cache activation
7. Measure cache hit rate using `cache_read_input_tokens` (Anthropic) and `cached_tokens` (OpenAI)

---

## Problem Statement

Naive multi-turn context management has three compounding failure modes:

- **Cost blow-up** — repeating the same 2,000-token system prompt on every turn costs $0.30/day at 100 calls. At 10,000 calls it is $30/day — entirely avoidable with prefix caching
- **Latency creep** — each added turn increases the prompt length the model must attend to; time-to-first-token grows linearly with context size for uncached calls
- **Silent context overflow** — at 128k token limits, a long conversation eventually exceeds the window; the model silently truncates from the end (losing the most recent user message) unless the application trims first

The root cause is treating the context window as an append-only log. The fix is a budget-enforced sliding window where the application controls what stays in context — system messages always, recent turns by default, older turns when budget allows, with summaries bridging the gap when turns are dropped.

---

## Prerequisites

- Python 3.10+
- No API key required for demo mode
- For live mode: an OpenAI API key set in `.env`
- `tiktoken` installed for exact token counts (falls back to ±15% approximation without it)

---

## Repository Structure

```
W2D2_kv-caching-token-trimming/
├── README.md                                  # This file
├── docs/
│   ├── technical-document.md                  # Full practitioner deep-dive (21 sections)
│   └── kv-caching-token-trimming-layman-scenarios.md  # Business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd                       # Two-layer caching architecture (Mermaid)
│   └── sequence.mmd                           # Token budget enforcement lifecycle (Mermaid)
└── poc/
    ├── README.md                              # Quick-start and expected output
    ├── src/
    │   ├── main.py                            # Entry point — demo + live mode
    │   ├── kv_caching_core.py                 # Token counting, trim_to_budget, prepare_context
    │   └── config.py                          # Config dataclass + env loader
    ├── tests/
    │   └── test_kv_caching.py                 # 25+ pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                      # 9-turn bank support conversation
    └── sample_output.json                     # Expected output at 4,000-token budget (no eviction)
```

---

## Core Concept: Two-Layer Caching

### Layer 1 — Server-Side KV Cache (static prefix)

The LLM server caches attention key/value matrices for prompt prefixes it has seen before. When the next request shares the same prefix, those matrices are reused — no recomputation.

**Anthropic** — opt-in via `cache_control`:
```python
system=[{
    "type": "text",
    "text": STATIC_SYSTEM_PROMPT,
    "cache_control": {"type": "ephemeral"}   # mark this block as cacheable
}]
# Check hit: response.usage.cache_read_input_tokens > 0
```

**OpenAI** — automatic for any shared prefix of 1,024+ tokens:
```python
# No code change required — check the hit rate:
# response.usage.prompt_tokens_details.cached_tokens
```

### Layer 2 — Client-Side Token Trimming (dynamic suffix)

`prepare_context()` enforces a hard token budget on the conversation suffix before every call:

```
messages (full history)
    │
    ▼
count_messages_tokens()          # exact count via tiktoken
    │
    ├─ within budget? → return as-is
    │
    └─ over budget?
        │
        ▼
    trim_to_budget()              # evict oldest turns at message boundaries
        │  rules:
        │  • system messages: NEVER evicted
        │  • tool_call + tool_result: evicted as an atomic pair
        │
        ▼
    eviction_ratio >= 0.5?
        │
        ├─ no  → return trimmed messages
        │
        └─ yes → build_compression_summary(evicted_turns)
                 inject_summary(trimmed, summary)
                 return messages with summary injected
```

### Key Functions

| Function | What it does |
|---|---|
| `count_tokens(text, model)` | Exact count via tiktoken; falls back to `len(text) // 4` if not installed |
| `count_messages_tokens(messages, model)` | Sums token counts across all messages + 4-token framing overhead per message |
| `trim_to_budget(messages, budget, model)` | Sliding window eviction — returns `(trimmed, original_count, final_count)` |
| `compute_eviction_ratio(original, final)` | Fraction evicted: `0.0` = none, `1.0` = all |
| `build_compression_summary(evicted)` | Builds plain-text summary of evicted turns for LLM compression in live mode |
| `inject_summary(messages, summary)` | Inserts summary as synthetic system message after last real system message |
| `prepare_context(messages, budget, ...)` | Orchestrates the full pipeline; returns dict with `messages`, `eviction_ratio`, `summary_injected` |

---

## Run the PoC

### Demo Mode (No API Key Required)

```bash
cd poc
pip install -r requirements.txt
python src/main.py
```

### Force Trimming (lower the budget to observe eviction)

```bash
MAX_CONTEXT_TOKENS=100 DEMO_MODE=true python src/main.py
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
# 25+ tests, all pass offline — no API key needed
```

---

## Expected Output

```
KV Caching & Token Trimming Demo
==================================================
Input conversation turns : 9
Token budget             : 4000 tokens
Mode                     : DEMO

[DEMO MODE] Output is pre-computed (no API call made)

Tokens before trimming   : 387
Tokens after trimming    : 387
Eviction ratio           : 0.0%
Summary injected         : False
Messages before / after  : 10 / 10

Cache note: In live mode, the system message would carry cache_control headers
to enable server-side KV cache reuse on Anthropic/OpenAI.

Latency                  : 0 ms

[OK] Concept demonstrated: token budget enforcement keeps context within limits
```

*The demo conversation (387 tokens) fits within the 4,000-token budget — no eviction occurs. Use `MAX_CONTEXT_TOKENS=100` to observe trimming and summary injection.*

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads `sample_input.json`, runs `prepare_context`, prints budget metrics |
| `src/kv_caching_core.py` | `count_tokens`, `count_messages_tokens`, `trim_to_budget`, `compute_eviction_ratio`, `build_compression_summary`, `inject_summary`, `prepare_context` |
| `src/config.py` | `Config` + `load_config()` — reads `MAX_CONTEXT_TOKENS`, `COMPRESSION_THRESHOLD`, `MODEL` |
| `tests/test_kv_caching.py` | 4 test classes, 25+ tests: token counting, boundary eviction, atomic tool pair eviction, compression threshold, summary injection, full pipeline |
| `sample_input.json` | 9-turn retail bank support conversation with system prompt (387 tokens total) |
| `sample_output.json` | Expected output: 387 tokens, `eviction_ratio=0.0`, no summary injected |

---

## Configuration

Copy `.env.example` to `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
MODEL=gpt-4o-mini
TEMPERATURE=0.0
MAX_TOKENS=500
MAX_CONTEXT_TOKENS=4000        # Hard token budget for the dynamic conversation suffix
COMPRESSION_THRESHOLD=0.5      # Trigger summary compression when eviction ratio > 50%
DEMO_MODE=false                # Set to true to run without an API key
```

Demo mode activates automatically when no `OPENAI_API_KEY` is present.

---

## Technical Documentation

The full practitioner guide is in [`docs/technical-document.md`](docs/technical-document.md). It covers:

- How transformer KV caches work and what "prefix reuse" means at the attention layer
- Anthropic `cache_control` configuration, cache TTL (5 minutes ephemeral), and cost structure
- OpenAI automatic prefix caching — minimum prefix length, hit rate measurement
- Client-side trimming strategies: sliding window vs. importance scoring vs. hierarchical summarisation
- Tool call pair integrity: why orphaned `tool_result` messages break the conversation schema
- Cost analysis: prefix caching savings at 1k / 10k / 100k calls per day
- Production checklist, anti-patterns, and 21 interview questions

For a jargon-free walkthrough, see [`docs/kv-caching-token-trimming-layman-scenarios.md`](docs/kv-caching-token-trimming-layman-scenarios.md).

---

## Architecture Diagrams

Mermaid source files are in [`diagrams/`](diagrams/):

- [`architecture.mmd`](diagrams/architecture.mmd) — Two-layer caching architecture: application → client-side trim → API call → server-side KV cache → model
- [`sequence.mmd`](diagrams/sequence.mmd) — Token budget enforcement lifecycle: message list → count → trim decision → optional compression → trimmed context

---

## Connection to the Series

- **W1D2 — Lost in the Middle:** Position-aware ordering ensures high-value content survives trimming and lands at context boundaries where attention peaks.
- **W1D5 — Episodic vs. Semantic Memory:** The memory router assembles working context before each LLM call — `prepare_context()` is what enforces the token budget on that assembled context.
- **W1D7 — LLM-as-a-Judge:** The judge prompt is a perfect prefix-caching candidate — the rubric and system instructions are static across every evaluation call.
- **W2D1 — Type-Safe Schemas:** Pydantic validation at the output boundary; token trimming at the input boundary — both enforce contracts at the LLM interface.
- **Today — W2D2 KV Caching & Token Trimming:** Budget-enforced sliding window + server-side prefix reuse keeps multi-turn agents fast and cost-efficient at scale.
- **Next — W2D3 GraphRAG & Knowledge Graphs:** With context costs under control, the next challenge is retrieval accuracy on entity-rich multi-hop queries.

---

## Key References

- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- OpenAI Prompt Caching: https://platform.openai.com/docs/guides/prompt-caching
- tiktoken: https://github.com/openai/tiktoken
- PagedAttention (vLLM KV cache management): https://arxiv.org/abs/2309.06180

---

## Continue Learning

**Next:** W2D3 — GraphRAG & Knowledge Graphs — How graph-structured knowledge enables multi-hop retrieval that flat vector search cannot support.

**Series index:** [Week 2 Overview](../README.md) | [Full Roadmap](../../README.md)
