# W2D2 — KV Caching & Token Trimming

**Series:** AI Engineering Production Playbook
**Vertical:** Context Engineering & Tokens
**Week 2 / Day 2**

## What This Demonstrates

Client-side token budget enforcement (sliding window eviction with optional summary compression) as the application-layer complement to server-side KV caching — keeping multi-turn LLM conversations fast, affordable, and within context limits.

## The Problem It Solves

Every LLM call recomputes attention over the full context window. A 50-turn conversation that naively appends all history sees latency spike 40–60% per request and costs grow proportionally. This PoC demonstrates:

- Exact token counting with tiktoken before every API call
- Sliding window eviction that always trims at message boundaries (never mid-turn)
- Atomic tool_call / tool_result pair eviction to prevent orphaned messages
- Summary compression for large eviction events (>50% of history)
- The structural pattern for applying server-side KV cache headers (Anthropic cache_control, OpenAI prompt caching)

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-02/W2D2_kv-caching-token-trimming/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API key (or leave OPENAI_API_KEY blank for demo mode)

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
KV Caching & Token Trimming Demo
==================================================
Input conversation turns : 9
Token budget             : 4000 tokens
Mode                     : DEMO

Tokens before trimming   : 387
Tokens after trimming    : 387
Eviction ratio           : 0.0%
Summary injected         : False
Messages before / after  : 10 / 10

Cache note: In live mode, the system message would carry cache_control headers
to enable server-side KV cache reuse on Anthropic/OpenAI.

Latency                  : 0 ms

Concept demonstrated: token budget enforcement keeps context within limits
```

To observe trimming in action, lower the token budget:

```bash
MAX_CONTEXT_TOKENS=100 DEMO_MODE=true python src/main.py
```

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — no API key required.

## Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — demonstrates end-to-end context preparation |
| `src/kv_caching_core.py` | Core logic: token counting, trim_to_budget, prepare_context |
| `src/config.py` | Configuration dataclass loaded from environment variables |
| `tests/test_kv_caching.py` | Unit tests — 4 test classes, 25+ tests, all offline |
| `sample_input.json` | Example 9-turn customer support conversation |
| `sample_output.json` | Expected output schema for the demo run |

## Core API

```python
from kv_caching_core import prepare_context

result = prepare_context(
    messages=conversation_history,   # OpenAI-format message list
    budget=4000,                     # Max tokens for the dynamic suffix
    compression_threshold=0.5,       # Trigger summary at >50% eviction
    model="gpt-4o-mini",
)

# result keys:
#   messages         — trimmed (and optionally summary-injected) message list
#   original_tokens  — token count before trimming
#   final_tokens     — token count after trimming
#   eviction_ratio   — fraction evicted (0.0 = none, 1.0 = all)
#   summary_injected — whether a compression summary was inserted
```

## Applying Server-Side KV Caching (Anthropic)

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": STATIC_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"}   # Cache this prefix
    }],
    messages=trimmed_conversation
)
# Check cache usage:
# response.usage.cache_read_input_tokens   — tokens served from cache
# response.usage.cache_creation_input_tokens — tokens written to cache
```

## Applying Server-Side KV Caching (OpenAI)

OpenAI prompt caching is automatic for prompts with a shared prefix of 1,024+ tokens. No code change required — check `response.usage.prompt_tokens_details.cached_tokens` to measure hit rate.

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Layman Scenarios](../docs/kv-caching-token-trimming-layman-scenarios.md)
- [Day README](../README.md)

## References

- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- OpenAI Prompt Caching: https://platform.openai.com/docs/guides/prompt-caching
- tiktoken: https://github.com/openai/tiktoken
- PagedAttention (vLLM): https://arxiv.org/abs/2309.06180
