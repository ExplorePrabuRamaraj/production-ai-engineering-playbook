#!/usr/bin/env python3
"""
W3D4 — Async & Parallel Tool Calls
====================================
Demonstrates: Fan-out independent LLM tool calls concurrently using
asyncio.gather() with per-tool timeout guards and a concurrency semaphore.

Run (live mode):  python src/main.py
Run (demo mode):  DEMO_MODE=true python src/main.py
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env && edit .env
"""

import asyncio
import json
import os
import time
from pathlib import Path

from config import load_config
from parallel_tools_core import (
    ToolResult,
    aggregate_results,
    compute_speedup,
    dispatch_tools_parallel,
    mock_get_product_price,
    mock_get_shipping_eta,
    mock_get_stock_status,
    mock_get_user_preferences,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {"product_id": "PROD-001", "user_id": "USER-42"}


# ---------------------------------------------------------------------------
# Demo mode — pre-computed output, no API key required
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    """Run the async dispatcher against mock tool coroutines (no API key needed)."""
    print("\n⚠️  Running in DEMO MODE — mock tools simulate realistic latencies\n")

    async def _run() -> list[ToolResult]:
        semaphore = asyncio.Semaphore(cfg.max_concurrent_tools)
        product_id = input_data.get("product_id", "PROD-001")
        user_id = input_data.get("user_id", "USER-42")

        tool_coroutines = [
            ("get_product_price",    mock_get_product_price(product_id)),
            ("get_stock_status",     mock_get_stock_status(product_id)),
            ("get_shipping_eta",     mock_get_shipping_eta(product_id, user_id)),
            ("get_user_preferences", mock_get_user_preferences(user_id)),
        ]
        return await dispatch_tools_parallel(tool_coroutines, semaphore, cfg.tool_timeout_s)

    wall_start = time.monotonic()
    results = asyncio.run(_run())
    wall_ms = (time.monotonic() - wall_start) * 1000

    context = aggregate_results(results)
    speedup = compute_speedup(results)
    return {"results": results, "context": context, "speedup": speedup, "wall_ms": wall_ms}


# ---------------------------------------------------------------------------
# Live mode — real OpenAI tool-calling (requires API key)
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    """
    Demonstrate parallel tool dispatch with a real OpenAI function-calling
    request. The LLM returns multiple tool_calls in one response; we fan
    them out concurrently rather than executing sequentially.
    """
    try:
        from openai import AsyncOpenAI

        async def _run() -> dict:
            client = AsyncOpenAI(api_key=cfg.openai_api_key)
            semaphore = asyncio.Semaphore(cfg.max_concurrent_tools)
            product_id = input_data.get("product_id", "PROD-001")
            user_id = input_data.get("user_id", "USER-42")

            # In production: parse tool_calls from LLM response and dispatch
            # Here we directly fan out the mock tools to show the pattern
            tool_coroutines = [
                ("get_product_price",    mock_get_product_price(product_id)),
                ("get_stock_status",     mock_get_stock_status(product_id)),
                ("get_shipping_eta",     mock_get_shipping_eta(product_id, user_id)),
                ("get_user_preferences", mock_get_user_preferences(user_id)),
            ]

            wall_start = time.monotonic()
            results = await dispatch_tools_parallel(tool_coroutines, semaphore, cfg.tool_timeout_s)
            wall_ms = (time.monotonic() - wall_start) * 1000

            context = aggregate_results(results)
            speedup = compute_speedup(results)
            return {"results": results, "context": context, "speedup": speedup, "wall_ms": wall_ms}

        return asyncio.run(_run())

    except ImportError:
        print("❌ openai package not installed. Run: pip install -r requirements.txt")
        raise
    except Exception as exc:
        print(f"❌ Live mode failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_results(output: dict) -> None:
    results: list[ToolResult] = output["results"]
    speedup: dict = output["speedup"]
    wall_ms: float = output["wall_ms"]

    print("Tool Results:")
    for r in results:
        print(f"  {r}")

    print(f"\nSequential baseline would have taken: ~{speedup['sequential_baseline_ms']:.0f}ms")
    print(f"Parallel actual time (wall clock):    ~{wall_ms:.0f}ms")
    print(f"Speedup:                               {speedup['speedup_ratio']:.2f}x")

    stats = output["context"].get("_dispatch_stats", {})
    print(f"\nDispatch stats: {stats['success']} success / "
          f"{stats['timeout']} timeout / {stats['error']} error "
          f"out of {stats['total']} tools")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nAsync Parallel Tool Calls Demo")
    print("=" * 50)
    print(f"Config: max_concurrent={cfg.max_concurrent_tools}, timeout={cfg.tool_timeout_s}s")

    input_data = load_sample_input()
    print(f"Input: product_id={input_data.get('product_id')}, user_id={input_data.get('user_id')}\n")
    print(f"Dispatching {len(cfg.tool_names)} tools concurrently...\n")

    output = run_demo(input_data) if cfg.demo_mode else run_live(input_data)
    print_results(output)

    print("\n✅ Concept demonstrated: independent tool calls run concurrently; "
          "timeouts handled as first-class results, not exceptions.")
    print("\n📚 See 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
