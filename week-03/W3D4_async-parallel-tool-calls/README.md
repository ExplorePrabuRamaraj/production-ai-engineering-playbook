# W3D4 — Async & Parallel Tool Calls

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

Sequential tool dispatch in agent pipelines accumulates latency: if four independent data-fetching tools each take 300ms, the agent waits 1,200ms before it can synthesise results. Parallel fan-out reduces that to the latency of the slowest single call. This PoC implements a production-ready async dispatcher that fans out independent tool coroutines with `asyncio.gather()`, bounds each call with a per-tool `asyncio.wait_for()` timeout, limits concurrency with `asyncio.Semaphore`, and converts every result — success, timeout, or error — into a typed `ToolResult` so the LLM always receives an explicit signal rather than a silent null.

---

## Learning Objectives

1. Understand why sequential tool dispatch accumulates latency and how parallel fan-out reduces wall-clock time to the slowest single call
2. Implement `dispatch_tools_parallel()` — fan out `(name, coroutine)` pairs with `asyncio.gather(return_exceptions=True)` and an `asyncio.Semaphore` concurrency cap
3. Apply `asyncio.wait_for()` per-tool timeout guards so a slow downstream API never blocks the entire dispatch batch
4. Understand why `return_exceptions=True` is critical — a single tool failure must never cancel the remaining concurrent calls
5. Implement `aggregate_results()` — convert `ToolResult` objects into an LLM-ready context dict with explicit fallback strings for timed-out or errored tools, preventing hallucination over silent nulls
6. Implement `compute_speedup()` — compare sequential baseline (sum of all latencies) against parallel wall clock (maximum latency) to quantify and log the fan-out benefit
7. Configure `MAX_CONCURRENT_TOOLS` (semaphore limit) to prevent thundering-herd effects against downstream APIs under production call volume

---

## Problem Statement

At 200,000 agent calls/month, a tool-calling agent that fetches price, stock, shipping, and user preferences sequentially (4 × ~300ms = ~1,200ms) adds over 240 hours of aggregate user-facing latency per month before the LLM generates a single response token. Sequential execution is the default because it requires no async code, but it is always wrong for independent data fetches. Parallel fan-out reduces four 300ms calls to ~300ms wall clock — a 4× improvement on the happy path. The harder production problem is the partial failure: when one of four tools times out, the agent must not block indefinitely waiting, and the LLM must receive `"data unavailable"` explicitly rather than a `None` it might hallucinate over. This PoC demonstrates both: the speedup and the graceful degradation.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs entirely offline with mock tool coroutines)
- Familiarity with [W1D4 — Model Context Protocol Intro](../../week-01/W1D4-model-context-protocol/README.md) and [W2D4 — Custom MCP Server Build](../../week-02/W2D4_custom-mcp-server-build/README.md) provides context on the tool-calling layer this dispatcher sits above

---

## Repository Structure

```
W3D4_async-parallel-tool-calls/
├── README.md                                    # This file
├── docs/
│   ├── technical-document.md                    # 21-section practitioner deep-dive
│   └── async-parallel-tools-layman-scenarios.md # Business scenarios without ML background
├── diagrams/
│   ├── architecture.mmd                         # Fan-out dispatcher architecture
│   └── sequence.mmd                             # Per-tool async dispatch + timeout sequence
└── poc/
    ├── README.md                                # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                              # Entry point — demo + live mode
    │   ├── parallel_tools_core.py               # Dispatcher, aggregator, speedup calculator
    │   └── config.py                            # Config dataclass + env loader
    ├── tests/
    │   └── test_parallel_tools.py               # 20+ unit tests across 4 test classes (all offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                        # product_id + user_id inputs
    └── sample_output.json                       # Expected output: 3 success / 1 timeout, 1.40x speedup
```

---

## Core Concepts

### `dispatch_tools_parallel()` — the fan-out engine

Fan out a list of `(name, coroutine)` pairs concurrently. The semaphore caps simultaneous outbound calls; `asyncio.wait_for()` bounds each call independently; `return_exceptions=True` ensures one bad call never cancels others:

```python
# parallel_tools_core.py
async def dispatch_tools_parallel(
    tool_coroutines: list[tuple[str, Awaitable]],
    semaphore: asyncio.Semaphore,
    timeout_s: float,
) -> list[ToolResult]:
    async def run_one(name: str, coro: Awaitable) -> ToolResult:
        start = time.monotonic()
        async with semaphore:           # cap concurrent outbound calls
            try:
                data = await asyncio.wait_for(coro, timeout=timeout_s)
                return ToolResult(tool_name=name, success=True, data=data, ...)
            except asyncio.TimeoutError:
                # timeout is a first-class result — never an unhandled exception
                return ToolResult(tool_name=name, success=False, timed_out=True, ...)
            except Exception as exc:
                return ToolResult(tool_name=name, success=False, error=str(exc), ...)

    tasks = [run_one(name, coro) for name, coro in tool_coroutines]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return safe_results   # top-level exceptions also wrapped
```

### `aggregate_results()` — LLM-safe context builder

Converts every `ToolResult` into a named field in a context dict. Timed-out or errored tools get an explicit `"data unavailable"` string — never a silent `None` that the LLM might hallucinate over:

```python
# parallel_tools_core.py
def aggregate_results(results: list[ToolResult], fallback_message="data unavailable") -> dict:
    context = {}
    for result in results:
        if result.success:
            context[result.tool_name] = result.data
        elif result.timed_out:
            context[result.tool_name] = fallback_message          # explicit — not None
        else:
            context[result.tool_name] = f"{fallback_message} (error: {result.error})"
    context["_dispatch_stats"] = {"success": ..., "timeout": ..., "error": ...}
    return context
```

### `compute_speedup()` — quantify the fan-out benefit

Sequential baseline is the sum of all individual latencies. Parallel wall clock is the maximum. Their ratio is the speedup — log this in production to verify that parallel execution is actually faster than sequential:

```python
# parallel_tools_core.py
def compute_speedup(results: list[ToolResult]) -> dict:
    sequential_ms = sum(r.latency_ms for r in results)    # what sequential would cost
    wall_ms       = max(r.latency_ms for r in results)    # what parallel actually costs
    return {
        "sequential_baseline_ms": round(sequential_ms, 1),
        "parallel_wall_ms":       round(wall_ms, 1),
        "speedup_ratio":          round(sequential_ms / wall_ms, 2),
    }
```

### Wiring it together

```python
# main.py (demo mode)
semaphore = asyncio.Semaphore(cfg.max_concurrent_tools)

tool_coroutines = [
    ("get_product_price",    mock_get_product_price(product_id)),
    ("get_stock_status",     mock_get_stock_status(product_id)),
    ("get_shipping_eta",     mock_get_shipping_eta(product_id, user_id)),
    ("get_user_preferences", mock_get_user_preferences(user_id)),
]

results = await dispatch_tools_parallel(tool_coroutines, semaphore, timeout_s=2.0)
context = aggregate_results(results)    # pass context dict to the LLM
speedup = compute_speedup(results)      # log to observability platform
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-03/W3D4_async-parallel-tool-calls/poc
pip install -r requirements.txt
cp .env.example .env
DEMO_MODE=true python src/main.py
```

### Live mode

```bash
# Edit .env and set OPENAI_API_KEY=your-key
python src/main.py
```

### Tune concurrency and timeout

```bash
MAX_CONCURRENT_TOOLS=3 TOOL_TIMEOUT_S=1.5 python src/main.py
```

### Tests

```bash
pytest tests/ -v
# 20+ tests across 4 test classes — all pass offline
```

---

## Expected Output

```
Async Parallel Tool Calls Demo
==================================================
Config: max_concurrent=5, timeout=2.0s
Input: product_id=PROD-001, user_id=USER-42

Dispatching 4 tools concurrently...

⚠️  Running in DEMO MODE — mock tools simulate realistic latencies

Tool Results:
  get_product_price       : SUCCESS   — $149.99               (312ms)
  get_stock_status        : SUCCESS   — In Stock (42 units)   (287ms)
  get_shipping_eta        : TIMEOUT   — fallback used         (2000ms)
  get_user_preferences    : SUCCESS   — Express shipping...   (198ms)

Sequential baseline would have taken: ~2797ms
Parallel actual time (wall clock):    ~2003ms
Speedup:                               1.40x

Dispatch stats: 3 success / 1 timeout / 0 error out of 4 tools

✅ Concept demonstrated: independent tool calls run concurrently;
   timeouts handled as first-class results, not exceptions.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — builds tool coroutine list, calls dispatcher, prints results and speedup |
| `src/parallel_tools_core.py` | `ToolResult`, `dispatch_tools_parallel()`, `aggregate_results()`, `compute_speedup()`, mock tool coroutines |
| `src/config.py` | `Config` dataclass + `load_config()` with semaphore limit, timeout, and tool name list |
| `tests/test_parallel_tools.py` | 20+ unit tests: dispatch happy path, timeout handling, error isolation, aggregator, speedup calculator |
| `sample_input.json` | `product_id` + `user_id` inputs for the 4 mock tools |
| `sample_output.json` | Pre-computed result: 3 success / 1 timeout, sequential ~2797ms, parallel ~2003ms, speedup 1.40× |
| `.env.example` | All environment variable defaults |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `MODEL` | `gpt-4o-mini` | LLM model for live mode synthesis |
| `MAX_CONCURRENT_TOOLS` | `5` | `asyncio.Semaphore` limit — lower if hitting downstream rate limits |
| `TOOL_TIMEOUT_S` | `2.0` | Per-tool deadline in seconds — raise for slow APIs, lower for strict SLA budgets |
| `DEMO_MODE` | `false` | Set `true` to run with mock tools (no API key needed) |
| `TEMPERATURE` | `0.0` | LLM temperature for deterministic synthesis |
| `MAX_TOKENS` | `500` | Max tokens per LLM response |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/async-parallel-tools-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — Fan-out dispatcher architecture
- [Sequence Diagram](diagrams/sequence.mmd) — Per-tool async dispatch and timeout sequence

---

## Connection to the Series

**Previous:** [W3D3 — Hybrid Search & Reranking](../W3D3_hybrid-search-reranking/README.md) — reduced retrieval latency by fusing BM25 and dense search; W3D4 tackles execution latency in the tool-calling layer above retrieval.

**Next:** [W3D5 — Dynamic Skill Selection](../README.md) — moves from how to execute tools fast to how an agent chooses the right tool at runtime using vector similarity over capability descriptions.

**Series arc:** [W1D4 — Model Context Protocol Intro](../../week-01/W1D4-model-context-protocol/README.md) introduced the tool-calling contract. [W2D4 — Custom MCP Server Build](../../week-02/W2D4_custom-mcp-server-build/README.md) showed how to build and expose tools. W3D4 closes the MCP vertical for Week 3: once you can build tools and expose them, the next problem is executing multiple tools efficiently — which means parallel fan-out, not sequential iteration.

---

## Key References

- Python `asyncio` documentation — [`asyncio.gather()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather), [`asyncio.wait_for()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for), [`asyncio.Semaphore`](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore)
- OpenAI parallel function calling — [Parallel function calling guide](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling)

---

## Continue Learning

**Next:** [W3D5 — Dynamic Skill Selection](../README.md)

Return to [Week 3 overview](../README.md) to explore all advanced techniques.
