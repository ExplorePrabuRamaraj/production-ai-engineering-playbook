"""
W3D4 — Async & Parallel Tool Calls — Core Logic
=================================================
Reusable async dispatcher: fan-out independent tool calls with per-tool
timeout guards and a concurrency semaphore. Import this module into any
agent orchestrator that needs parallel tool execution.
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


# ---------------------------------------------------------------------------
# Typed result objects — never pass raw dicts or None out of the dispatcher
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Successful tool call result."""
    tool_name: str
    success: bool
    data: Any
    latency_ms: float
    error: str | None = None
    timed_out: bool = False

    def __str__(self) -> str:
        if self.timed_out:
            return f"{self.tool_name:<24}: TIMEOUT   — fallback used       ({self.latency_ms:.0f}ms)"
        if not self.success:
            return f"{self.tool_name:<24}: ERROR     — {self.error}  ({self.latency_ms:.0f}ms)"
        return f"{self.tool_name:<24}: SUCCESS   — {self.data!s:<20} ({self.latency_ms:.0f}ms)"


# ---------------------------------------------------------------------------
# Mock tool implementations — simulate realistic latencies for demo mode
# ---------------------------------------------------------------------------

async def mock_get_product_price(product_id: str) -> str:
    """Simulate a pricing API call (~300ms average)."""
    await asyncio.sleep(random.uniform(0.25, 0.35))
    return "$149.99"


async def mock_get_stock_status(product_id: str) -> str:
    """Simulate an inventory API call (~280ms average)."""
    await asyncio.sleep(random.uniform(0.22, 0.32))
    return "In Stock (42 units)"


async def mock_get_shipping_eta(product_id: str, user_id: str) -> str:
    """Simulate a shipping API call — occasionally slow to demonstrate timeout."""
    # 30% chance of being slow enough to trigger a 2s timeout in demo
    delay = random.uniform(2.1, 2.5) if random.random() < 0.3 else random.uniform(0.18, 0.28)
    await asyncio.sleep(delay)
    return "2-3 business days"


async def mock_get_user_preferences(user_id: str) -> str:
    """Simulate a CRM preferences lookup (~200ms average)."""
    await asyncio.sleep(random.uniform(0.15, 0.25))
    return "Express shipping preferred"


# ---------------------------------------------------------------------------
# Core dispatcher — the production-ready parallel execution engine
# ---------------------------------------------------------------------------

async def dispatch_tools_parallel(
    tool_coroutines: list[tuple[str, Awaitable]],
    semaphore: asyncio.Semaphore,
    timeout_s: float,
) -> list[ToolResult]:
    """
    Fan out a list of (name, coroutine) pairs concurrently, with per-tool
    timeout guards and exception isolation.

    Why return_exceptions=True: we never want a single tool failure to cancel
    the remaining calls. Each result is inspected individually by the caller.

    Args:
        tool_coroutines: List of (tool_name, awaitable) pairs to execute
        semaphore:        Limits max concurrent outbound calls (prevents thundering herd)
        timeout_s:        Per-tool deadline in seconds

    Returns:
        List of ToolResult objects in same order as input — never raises
    """

    async def run_one(name: str, coro: Awaitable) -> ToolResult:
        start = time.monotonic()
        # Semaphore ensures we never exceed MAX_CONCURRENT_TOOLS at once
        async with semaphore:
            try:
                data = await asyncio.wait_for(coro, timeout=timeout_s)
                return ToolResult(
                    tool_name=name,
                    success=True,
                    data=data,
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            except asyncio.TimeoutError:
                # Timeout is a first-class result, not an unhandled exception
                return ToolResult(
                    tool_name=name,
                    success=False,
                    data=None,
                    latency_ms=timeout_s * 1000,
                    timed_out=True,
                )
            except Exception as exc:
                return ToolResult(
                    tool_name=name,
                    success=False,
                    data=None,
                    latency_ms=(time.monotonic() - start) * 1000,
                    error=str(exc),
                )

    tasks = [run_one(name, coro) for name, coro in tool_coroutines]
    # gather with return_exceptions=True so one bad task never cancels others
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Wrap any unexpected top-level exceptions (should not happen, but be safe)
    safe_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            name = tool_coroutines[i][0] if i < len(tool_coroutines) else f"tool_{i}"
            safe_results.append(ToolResult(
                tool_name=name, success=False, data=None,
                latency_ms=0.0, error=f"Unexpected: {result}"
            ))
        else:
            safe_results.append(result)
    return safe_results


# ---------------------------------------------------------------------------
# Result aggregator — converts raw results into an LLM-ready context dict
# ---------------------------------------------------------------------------

def aggregate_results(
    results: list[ToolResult],
    fallback_message: str = "data unavailable — please check manually",
) -> dict[str, Any]:
    """
    Convert tool results into a structured context dict for LLM consumption.

    Timed-out or errored tools receive an explicit fallback string.
    The LLM sees every field — it never encounters a silent None.

    Why explicit fallbacks: LLMs may hallucinate values for missing fields.
    Explicit "data unavailable" markers prevent this failure mode.
    """
    context: dict[str, Any] = {}
    stats = {"total": len(results), "success": 0, "timeout": 0, "error": 0}

    for result in results:
        if result.success:
            context[result.tool_name] = result.data
            stats["success"] += 1
        elif result.timed_out:
            context[result.tool_name] = fallback_message
            stats["timeout"] += 1
        else:
            context[result.tool_name] = f"{fallback_message} (error: {result.error})"
            stats["error"] += 1

    context["_dispatch_stats"] = stats
    return context


# ---------------------------------------------------------------------------
# Latency helpers — compute sequential baseline vs parallel actual
# ---------------------------------------------------------------------------

def compute_speedup(results: list[ToolResult]) -> dict[str, float]:
    """
    Compute the theoretical sequential baseline vs observed parallel wall time.
    Demonstrates the concrete benefit of fan-out execution.
    """
    total_sequential_ms = sum(r.latency_ms for r in results)
    parallel_wall_ms = max(r.latency_ms for r in results) if results else 0.0
    speedup = total_sequential_ms / parallel_wall_ms if parallel_wall_ms > 0 else 1.0
    return {
        "sequential_baseline_ms": round(total_sequential_ms, 1),
        "parallel_wall_ms": round(parallel_wall_ms, 1),
        "speedup_ratio": round(speedup, 2),
    }
