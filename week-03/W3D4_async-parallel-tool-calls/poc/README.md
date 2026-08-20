# W3D4 — Async & Parallel Tool Calls

**Series:** AI Engineering Production Playbook
**Vertical:** MCP & Tool Integration
**Week 3 / Day 4 of 28**

## What This Demonstrates

How to fan out independent LLM tool calls concurrently using `asyncio.gather()` with per-tool timeout guards (`asyncio.wait_for()`) and a concurrency limiter (`asyncio.Semaphore`), reducing multi-tool agent latency from the sum of individual call times to the maximum.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs entirely offline with mock tools)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-03/W3D4_async-parallel-tool-calls/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API key, or leave blank for demo mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode uses mock async tool coroutines with realistic simulated latencies (200–350ms per tool). One tool has a 30% chance of exceeding the timeout threshold, demonstrating graceful fallback handling. No network calls are made.

## Expected Output

```
Async Parallel Tool Calls Demo
==================================================
Config: max_concurrent=5, timeout=2.0s
Input: product_id=PROD-001, user_id=USER-42

Dispatching 4 tools concurrently...

Tool Results:
  get_product_price       : SUCCESS   — $149.99               (312ms)
  get_stock_status        : SUCCESS   — In Stock (42 units)   (287ms)
  get_shipping_eta        : TIMEOUT   — fallback used         (2000ms)
  get_user_preferences    : SUCCESS   — Express shipping...   (198ms)

Sequential baseline would have taken: ~2797ms
Parallel actual time (wall clock):    ~2003ms
Speedup:                               1.40x

Dispatch stats: 3 success / 1 timeout / 0 error out of 4 tools

✅ Concept demonstrated: independent tool calls run concurrently; timeouts handled as first-class results, not exceptions.
```

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline. No API key required.

## Tune Concurrency and Timeouts

Edit `.env` to adjust dispatcher behaviour:

```bash
MAX_CONCURRENT_TOOLS=5    # Lower if hitting downstream rate limits
TOOL_TIMEOUT_S=2.0        # Raise for slow APIs; lower for strict SLA budgets
```

## Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — demo and live mode orchestration |
| `src/parallel_tools_core.py` | Reusable async dispatcher, aggregator, speedup calculator |
| `src/config.py` | Configuration dataclass loaded from environment |
| `tests/test_parallel_tools.py` | 20+ unit tests across 4 test classes |
| `sample_input.json` | Example input for the dispatcher |
| `sample_output.json` | Expected output structure with partial timeout result |

## Core Pattern

```python
import asyncio
from parallel_tools_core import dispatch_tools_parallel

semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

tool_coroutines = [
    ("get_price",    pricing_api(product_id)),
    ("get_stock",    inventory_api(product_id)),
    ("get_shipping", shipping_api(product_id, user_id)),
]

results = await dispatch_tools_parallel(
    tool_coroutines,
    semaphore,
    timeout_s=2.0,
)
# Each result is a ToolResult — success=True/False, timed_out, error
```

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Layman Scenarios](../docs/async-parallel-tools-layman-scenarios.md)
- [LinkedIn Post](../README.md)

## Series Navigation

**Previous:** W3D3 — Hybrid Search & Reranking
**Next:** W3D5 — Dynamic Skill Selection
