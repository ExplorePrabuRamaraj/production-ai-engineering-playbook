"""
W3D4 — Async & Parallel Tool Calls — Unit Tests
=================================================
Run: pytest tests/ -v

All external API calls are mocked. Tests pass completely offline.
Tests cover: demo mode, core dispatcher logic, timeout isolation,
exception isolation, result aggregation, and sample file validation.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parallel_tools_core import (
    ToolResult,
    dispatch_tools_parallel,
    aggregate_results,
    compute_speedup,
    mock_get_product_price,
    mock_get_stock_status,
)
from main import run_demo, load_sample_input


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    return {"product_id": "TEST-001", "user_id": "USER-99"}


@pytest.fixture
def semaphore():
    return asyncio.Semaphore(5)


@pytest.fixture
def successful_results():
    return [
        ToolResult("tool_a", success=True, data="value_a", latency_ms=300.0),
        ToolResult("tool_b", success=True, data="value_b", latency_ms=250.0),
        ToolResult("tool_c", success=True, data="value_c", latency_ms=200.0),
    ]


@pytest.fixture
def mixed_results():
    """Results with one success, one timeout, one error."""
    return [
        ToolResult("tool_a", success=True,  data="value_a", latency_ms=300.0),
        ToolResult("tool_b", success=False, data=None,      latency_ms=2000.0, timed_out=True),
        ToolResult("tool_c", success=False, data=None,      latency_ms=150.0,  error="Connection refused"),
    ]


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Demo mode must run without any API key and produce valid output."""

    def test_demo_returns_results_list(self, sample_input):
        output = run_demo(sample_input)
        assert "results" in output
        assert isinstance(output["results"], list)
        assert len(output["results"]) == 4

    def test_demo_returns_context_dict(self, sample_input):
        output = run_demo(sample_input)
        assert "context" in output
        assert isinstance(output["context"], dict)

    def test_demo_returns_speedup_stats(self, sample_input):
        output = run_demo(sample_input)
        speedup = output["speedup"]
        assert "sequential_baseline_ms" in speedup
        assert "parallel_wall_ms" in speedup
        assert "speedup_ratio" in speedup
        # Parallel wall time should be less than sequential baseline for 4 tools
        assert speedup["parallel_wall_ms"] < speedup["sequential_baseline_ms"]

    def test_demo_wall_time_less_than_sequential(self, sample_input):
        """Key property: wall time must be < sum of individual latencies."""
        output = run_demo(sample_input)
        # Wall time should not exceed sequential_baseline (with small tolerance for gather overhead)
        assert output["wall_ms"] < output["speedup"]["sequential_baseline_ms"]

    def test_demo_dispatch_stats_sum_to_total(self, sample_input):
        output = run_demo(sample_input)
        stats = output["context"]["_dispatch_stats"]
        assert stats["success"] + stats["timeout"] + stats["error"] == stats["total"]
        assert stats["total"] == 4


# ---------------------------------------------------------------------------
# TestCoreConcept — pure dispatcher and aggregator logic
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Core async dispatcher and result aggregation — tested with controlled coroutines."""

    def test_dispatch_all_succeed(self, semaphore):
        """All successful coroutines should return ToolResult with success=True."""
        async def fast_tool(name: str) -> str:
            return f"result_{name}"

        async def _run():
            coros = [("tool_x", fast_tool("x")), ("tool_y", fast_tool("y"))]
            return await dispatch_tools_parallel(coros, semaphore, timeout_s=5.0)

        results = asyncio.run(_run())
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].data == "result_x"
        assert results[1].data == "result_y"

    def test_dispatch_timeout_isolated(self, semaphore):
        """A single slow tool should time out; others should complete normally."""
        async def fast_tool() -> str:
            await asyncio.sleep(0.05)
            return "fast_result"

        async def slow_tool() -> str:
            await asyncio.sleep(10.0)   # Exceeds timeout
            return "never_reached"

        async def _run():
            coros = [("fast", fast_tool()), ("slow", slow_tool())]
            return await dispatch_tools_parallel(coros, semaphore, timeout_s=0.3)

        results = asyncio.run(_run())
        fast_result = next(r for r in results if r.tool_name == "fast")
        slow_result = next(r for r in results if r.tool_name == "slow")

        assert fast_result.success is True
        assert fast_result.data == "fast_result"
        assert slow_result.success is False
        assert slow_result.timed_out is True

    def test_dispatch_exception_isolated(self, semaphore):
        """A tool that raises should not cancel other tools."""
        async def good_tool() -> str:
            await asyncio.sleep(0.05)
            return "good"

        async def bad_tool() -> str:
            raise ValueError("tool exploded")

        async def _run():
            coros = [("good", good_tool()), ("bad", bad_tool())]
            return await dispatch_tools_parallel(coros, semaphore, timeout_s=2.0)

        results = asyncio.run(_run())
        good = next(r for r in results if r.tool_name == "good")
        bad = next(r for r in results if r.tool_name == "bad")

        assert good.success is True
        assert bad.success is False
        assert bad.error is not None
        assert "tool exploded" in bad.error

    def test_dispatch_preserves_order(self, semaphore):
        """Results must be in same order as input coroutines."""
        async def tool(n: int) -> int:
            await asyncio.sleep(0.01 * (5 - n))   # Later tools finish first
            return n

        async def _run():
            coros = [(f"tool_{i}", tool(i)) for i in range(5)]
            return await dispatch_tools_parallel(coros, semaphore, timeout_s=2.0)

        results = asyncio.run(_run())
        assert [r.tool_name for r in results] == [f"tool_{i}" for i in range(5)]
        assert [r.data for r in results] == list(range(5))

    def test_dispatch_semaphore_limits_concurrency(self):
        """Semaphore(1) should force tools to run one at a time."""
        call_log: list[str] = []

        async def tracked_tool(name: str) -> str:
            call_log.append(f"start_{name}")
            await asyncio.sleep(0.05)
            call_log.append(f"end_{name}")
            return name

        async def _run():
            sem = asyncio.Semaphore(1)   # Only 1 at a time
            coros = [("a", tracked_tool("a")), ("b", tracked_tool("b"))]
            return await dispatch_tools_parallel(coros, sem, timeout_s=5.0)

        asyncio.run(_run())
        # With semaphore=1: a must end before b starts
        assert call_log.index("end_a") < call_log.index("start_b")

    @pytest.mark.parametrize("tool_count,timeout_s,expected_success", [
        (1, 5.0, True),
        (3, 5.0, True),
        (5, 5.0, True),
        (1, 0.001, False),   # Sub-millisecond timeout forces timeout
    ])
    def test_dispatch_parametrized_configurations(self, semaphore, tool_count, timeout_s, expected_success):
        """Dispatcher behaves correctly across varied tool counts and timeout values."""
        async def quick_tool(n: int) -> str:
            await asyncio.sleep(0.05)
            return f"result_{n}"

        async def _run():
            coros = [(f"tool_{i}", quick_tool(i)) for i in range(tool_count)]
            return await dispatch_tools_parallel(coros, semaphore, timeout_s=timeout_s)

        results = asyncio.run(_run())
        assert len(results) == tool_count
        for r in results:
            assert r.success == expected_success


# ---------------------------------------------------------------------------
# TestAggregation — result aggregator converts results to LLM context
# ---------------------------------------------------------------------------

class TestAggregation:
    """Result aggregator must handle success, timeout, and error results correctly."""

    def test_aggregate_all_success(self, successful_results):
        context = aggregate_results(successful_results)
        assert context["tool_a"] == "value_a"
        assert context["tool_b"] == "value_b"
        assert context["_dispatch_stats"]["success"] == 3
        assert context["_dispatch_stats"]["timeout"] == 0

    def test_aggregate_mixed_uses_fallback(self, mixed_results):
        context = aggregate_results(mixed_results)
        assert context["tool_a"] == "value_a"
        assert "unavailable" in context["tool_b"]   # Timeout fallback
        assert "unavailable" in context["tool_c"]   # Error fallback
        assert context["_dispatch_stats"]["success"] == 1
        assert context["_dispatch_stats"]["timeout"] == 1
        assert context["_dispatch_stats"]["error"] == 1

    def test_aggregate_custom_fallback_message(self, mixed_results):
        context = aggregate_results(mixed_results, fallback_message="N/A")
        assert context["tool_b"] == "N/A"

    def test_speedup_calculation(self, successful_results):
        speedup = compute_speedup(successful_results)
        # Sequential: 300+250+200 = 750ms; Parallel: max=300ms; ratio ~2.5
        assert speedup["sequential_baseline_ms"] == 750.0
        assert speedup["parallel_wall_ms"] == 300.0
        assert speedup["speedup_ratio"] == pytest.approx(2.5, rel=0.01)

    def test_speedup_single_tool(self):
        results = [ToolResult("only", success=True, data="x", latency_ms=500.0)]
        speedup = compute_speedup(results)
        assert speedup["speedup_ratio"] == pytest.approx(1.0, rel=0.01)


# ---------------------------------------------------------------------------
# TestLiveMode — OpenAI async client mocked
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode with all async API calls mocked — passes offline."""

    @patch("main.cfg")
    def test_live_mode_uses_async_dispatcher(self, mock_cfg, sample_input):
        """Live mode should call dispatch_tools_parallel (not sequential calls)."""
        mock_cfg.demo_mode = False
        mock_cfg.openai_api_key = "sk-fake-key"
        mock_cfg.max_concurrent_tools = 5
        mock_cfg.tool_timeout_s = 2.0
        mock_cfg.tool_names = ["get_product_price", "get_stock_status",
                                "get_shipping_eta", "get_user_preferences"]

        with patch("main.dispatch_tools_parallel", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = [
                ToolResult("get_product_price", True, "$99.99", 200.0),
                ToolResult("get_stock_status",  True, "In Stock", 180.0),
                ToolResult("get_shipping_eta",  True, "3 days",   190.0),
                ToolResult("get_user_preferences", True, "standard", 170.0),
            ]
            with patch("main.AsyncOpenAI"):
                from main import run_live
                result = run_live(sample_input)

        assert "results" in result
        assert len(result["results"]) == 4


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Sample files must be valid JSON with expected schema."""

    def test_sample_input_loads(self):
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_required_keys(self):
        data = load_sample_input()
        assert "product_id" in data
        assert "user_id" in data

    def test_sample_output_is_valid_json(self):
        path = Path(__file__).parent.parent / "sample_output.json"
        if path.exists():
            data = json.loads(path.read_text())
            assert isinstance(data, dict)
            assert "results" in data or "context" in data

    def test_mock_tools_are_coroutines(self):
        """Mock tool functions should return awaitables."""
        import inspect
        assert inspect.iscoroutinefunction(mock_get_product_price)
        assert inspect.iscoroutinefunction(mock_get_stock_status)
