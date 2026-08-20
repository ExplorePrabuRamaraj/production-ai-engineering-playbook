# W3D4 — Async & Parallel Tool Calls
## AI Engineering Production Playbook — Week 3, Day 4

**Vertical:** MCP & Tool Integration  
**Series Day:** 18 of 28  
**Prerequisite Reading:** W1D4 (MCP Intro), W2D4 (Custom MCP Server Build)

---

## 1. Overview

Async and parallel tool calls are the technique of dispatching multiple independent LLM tool invocations concurrently rather than sequentially, collecting results as they arrive. In production agentic systems, an agent frequently needs to query several external services — databases, APIs, search indexes — before it can compose a final response. When those calls are independent of each other, executing them one at a time wastes wall-clock time proportional to the sum of all call latencies rather than the maximum. This technique applies to any agentic framework that exposes tool-calling interfaces: OpenAI function calling, Anthropic tool use, LangChain tools, and MCP-connected tool servers. It is production-relevant now because agent pipelines are shifting from single-tool chains to multi-tool workflows where latency directly correlates with user-perceived quality.

---

## 2. Learning Objectives

By the end of this document, you will be able to:

1. **Explain** why sequential tool execution degrades agent response time at scale
2. **Distinguish** between independent tool calls (fan-out candidates) and dependent tool calls (must stay sequential)
3. **Implement** async parallel tool dispatch using Python's `asyncio.gather()` with proper exception isolation
4. **Apply** `asyncio.Semaphore` to control concurrency against rate-limited external APIs
5. **Design** per-tool timeout strategies using `asyncio.wait_for()` to prevent cascade failures
6. **Evaluate** the latency profile of a multi-tool agent before and after async refactoring
7. **Build** a production-ready async tool dispatcher with dependency graph resolution
8. **Benchmark** fan-out concurrency versus sequential baseline using realistic workloads

---

## 3. Problem Statement

### The Sequential Execution Trap

A production LLM agent processing a user query such as "What is the current price, stock status, and shipping estimate for product XYZ for customer ABC?" must make at least three distinct tool calls:

1. Fetch product pricing from a pricing service (~300ms)
2. Check inventory availability from a warehouse API (~250ms)
3. Retrieve customer shipping tier from a CRM (~200ms)

In a naive sequential implementation, the agent awaits each call before dispatching the next. Total wall-clock time: 750ms minimum, before any LLM processing. At P95 — factoring in network jitter, cold starts, and occasional slow responses — this regularly exceeds 2–3 seconds.

**The failure mode in production:** When this agent serves 500 concurrent users, each sequential 3-second workflow consumes an event loop slot for the full duration. Throughput collapses. SLA targets for sub-1-second responses become impossible. Operations teams observe high CPU-idle time alongside high latency — the system is waiting, not working.

**Why naive async alone does not fix it:** Wrapping synchronous blocking calls in `asyncio.run_in_executor()` without a properly sized thread pool creates the illusion of concurrency while still saturating OS threads. Developers assume they have fixed the problem; monitoring shows otherwise at scale.

**The deeper issue — missing dependency analysis:** Many teams reach for async without first auditing which tool calls are actually independent. A tool that requires an order ID returned by a previous tool cannot be parallelised with it. Skipping this analysis leads to race conditions and incorrect results silently produced at runtime.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: E-Commerce Order Summary Agent

A mid-size e-commerce company builds an AI assistant to answer order status questions. The agent calls four tools per query: `get_order_details`, `get_shipment_tracking`, `get_return_eligibility`, and `get_loyalty_points`. All four are independent given the order ID from the user's message.

The initial implementation executes them sequentially. Each tool call averages 200–400ms. Median end-to-end response time: 1.4 seconds. P95: 3.8 seconds. During peak traffic (Black Friday), response times spike to 8+ seconds as the event loop queues up. Customer satisfaction scores for the AI assistant drop 22% versus the human support baseline. The team increases server capacity to compensate — costs rise 40% — without solving the root cause.

### Scenario B — The Solution: Parallelised with Semaphore and Timeout Guards

The same agent is refactored to use `asyncio.gather()` with `return_exceptions=True`. The four independent tool calls are fanned out concurrently. A `Semaphore(10)` limits concurrent outbound connections per agent instance to prevent thundering herd against the downstream APIs. Each tool call is wrapped in `asyncio.wait_for()` with a 1.5-second deadline; timeout results are handled gracefully (partial response with a "data unavailable" fallback).

Results after refactoring: median end-to-end response time drops from 1.4 seconds to 480ms (a 66% reduction). P95 drops from 3.8 seconds to 1.1 seconds. Server capacity requirement decreases by 30% because each agent instance processes more requests per second. The Black Friday spike is handled at the same infrastructure cost as a normal weekday.

---

## 5. Solution Architecture

The async parallel tool dispatch pattern has four structural components:

**1. Dependency Graph Resolver:** Before execution, the agent analyses which tool calls depend on outputs from other tools and which are independent. Independent calls form a fan-out group. Dependent calls form a sequential chain that begins only after their dependencies resolve.

**2. Async Fan-Out Dispatcher:** Independent calls are wrapped as coroutines and dispatched via `asyncio.gather(return_exceptions=True)`. Each coroutine is individually wrapped in `asyncio.wait_for()` with a per-tool timeout budget. Results arrive as a list maintaining input order.

**3. Concurrency Limiter:** An `asyncio.Semaphore` governs the maximum number of concurrent outbound calls. This prevents rate-limiting from downstream services and controls resource consumption on the agent host.

**4. Result Aggregator:** After gather() completes, the aggregator inspects each result. Exceptions (including `TimeoutError`) are logged and replaced with structured fallback values. The LLM then receives a complete context object with available data and explicit markers for unavailable fields.

```
User Query
    │
    ▼
Dependency Graph Resolver
    │
    ├─── Sequential Chain (dependent tools)
    │         tool_A ──► tool_B ──► tool_C
    │
    └─── Fan-Out Group (independent tools)
              asyncio.gather()
              ├── tool_X  (with timeout)
              ├── tool_Y  (with timeout)
              └── tool_Z  (with timeout)
                      │
              Result Aggregator
              (exception isolation + fallback)
                      │
              LLM Context Assembly
                      │
              Final Response
```

---

## 6. Internal Working Mechanics

### asyncio.gather() Deep Dive

`asyncio.gather(*coroutines, return_exceptions=False)` schedules all passed coroutines on the current event loop and waits for all of them to complete. It returns a list of results in the same order as the input coroutines — not in completion order.

**Critical parameter — `return_exceptions`:**
- `False` (default): The first exception cancels remaining coroutines and re-raises immediately. Remaining tool calls are abandoned. This is almost never correct in agentic workflows.
- `True`: All coroutines run to completion. Exceptions are returned as values in the result list. The caller inspects each result with `isinstance(result, Exception)`.

### Timeout Architecture with asyncio.wait_for()

```python
async def call_with_timeout(coro, tool_name: str, timeout_s: float):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        return ToolTimeoutResult(tool_name=tool_name, timeout_s=timeout_s)
    except Exception as e:
        return ToolErrorResult(tool_name=tool_name, error=str(e))
```

The wrapper converts both timeouts and exceptions into typed fallback objects. The aggregator downstream never sees raw exceptions — it sees structured data it knows how to handle. This prevents a single slow external API from propagating errors into the LLM context.

### Semaphore Mechanics

```python
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

async def bounded_call(coro):
    async with semaphore:
        return await coro
```

The semaphore is acquired before each tool call and released when the coroutine completes (or raises). When all `MAX_CONCURRENT_TOOLS` slots are occupied, additional coroutines wait in the event loop queue without blocking OS threads. This is purely cooperative — zero thread overhead.

### Dependency Graph Resolution

For a set of tool calls `{T1, T2, T3, T4}` with dependencies `T3 depends on T1`:

1. Build an adjacency list: `{T1: [], T2: [], T3: [T1], T4: []}`
2. Topological sort to identify execution waves:
   - Wave 1 (no dependencies): `{T1, T2, T4}` → fan-out together
   - Wave 2 (depends on Wave 1 completion): `{T3}` → executes after T1 result is available
3. Execute Wave 1 with `gather()`, resolve results, then execute Wave 2 sequentially or as a new fan-out

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd` for the full Mermaid source.

```
Async Tool Dispatch — Architecture Overview

[Agent Orchestrator]
    │
    ▼
[Dependency Resolver] ──► [Tool Call Plan]
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      [Fan-Out Wave 1]   [Fan-Out Wave 1]  [Sequential Chain]
      Tool A (async)     Tool B (async)    Tool C → Tool D
         │                   │
         └─────── asyncio.gather() ──────┘
                       │
              [Semaphore Guard (n=10)]
                       │
              [Per-Tool Timeout Wrapper]
                       │
              [Result Aggregator]
              (exception isolation + fallback)
                       │
              [LLM Context Builder]
                       │
              [Final Response]
```

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd` for the full Mermaid source.

The sequence captures a 3-tool fan-out with one timeout:

1. Agent receives user query
2. Dependency resolver classifies tools as independent
3. asyncio.gather() dispatches all three concurrently
4. Tool A and Tool B complete within deadline
5. Tool C exceeds timeout → TimeoutResult returned
6. Aggregator builds context with fallback for Tool C
7. LLM generates response acknowledging partial data
8. Agent returns response to user

---

## 9. Implementation Guide

### Step 1: Install Dependencies

```bash
pip install openai>=1.30.0 pydantic>=2.0.0 pytest pytest-asyncio
```

### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, MAX_CONCURRENT_TOOLS, TOOL_TIMEOUT_S
```

### Step 3: Define Typed Tool Results

```python
# In parallel_tools_core.py
from dataclasses import dataclass
from typing import Any

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Any
    latency_ms: float
    error: str | None = None
    timed_out: bool = False
```

### Step 4: Implement the Bounded Async Dispatcher

```python
import asyncio
import time
from typing import Callable, Awaitable

async def dispatch_tools_parallel(
    tool_coroutines: list[tuple[str, Awaitable]],
    semaphore: asyncio.Semaphore,
    timeout_s: float,
) -> list[ToolResult]:
    async def run_one(name: str, coro: Awaitable) -> ToolResult:
        start = time.monotonic()
        async with semaphore:
            try:
                data = await asyncio.wait_for(coro, timeout=timeout_s)
                return ToolResult(
                    tool_name=name, success=True,
                    data=data, latency_ms=(time.monotonic() - start) * 1000
                )
            except asyncio.TimeoutError:
                return ToolResult(
                    tool_name=name, success=False, data=None,
                    latency_ms=timeout_s * 1000, timed_out=True
                )
            except Exception as e:
                return ToolResult(
                    tool_name=name, success=False, data=None,
                    latency_ms=(time.monotonic() - start) * 1000, error=str(e)
                )

    tasks = [run_one(name, coro) for name, coro in tool_coroutines]
    return await asyncio.gather(*tasks)
```

### Step 5: Run the PoC

```bash
# Demo mode (no API key required)
DEMO_MODE=true python src/main.py

# Live mode
python src/main.py

# Tests
pytest tests/ -v
```

### Step 6: Verify Output

Expected demo output:
```
Async Parallel Tool Calls Demo
================================
Dispatching 4 tools concurrently (max_concurrent=3, timeout=2.0s)...

Tool Results:
  get_product_price   : SUCCESS  — $149.99      (312ms)
  get_stock_status    : SUCCESS  — In Stock (42) (287ms)
  get_shipping_eta    : SUCCESS  — 2-3 days      (198ms)
  get_user_prefs      : TIMEOUT  — fallback used (2000ms)

Sequential baseline would have taken: ~797ms
Parallel actual time:                  ~312ms (wall clock)
Speedup:                               2.55x

✅ Concept demonstrated: independent tool calls run concurrently, timeouts handled gracefully
```

---

## 10. Benefits & Trade-offs

| Benefit | Trade-off |
|---|---|
| Wall-clock latency reduced to max(individual latencies) | Requires upfront dependency analysis — incorrect graph causes race conditions |
| Higher agent throughput per server instance | Semaphore sizing requires load testing — too low wastes parallelism, too high triggers rate limits |
| Single slow tool cannot block all results | Partial results require fallback logic — increases code complexity |
| Timeout isolation prevents cascade failures | `asyncio.wait_for()` cancels coroutines — underlying HTTP connections may not close immediately without explicit cleanup |
| Zero additional thread overhead (event loop only) | Does not help with CPU-bound tool logic — only I/O-bound calls benefit |

---

## 11. Performance Characteristics

### Latency Model

For `n` independent tools with individual latencies `L1, L2, ..., Ln`:
- **Sequential:** Total latency = `sum(L1..Ln)`
- **Parallel:** Total latency = `max(L1..Ln)` + gather overhead (~1–5ms)

For 4 tools averaging 250ms each:
- Sequential P50: ~1,000ms
- Parallel P50: ~260ms (3.8× speedup)
- Sequential P95: ~2,500ms (including jitter)
- Parallel P95: ~650ms (limited by slowest tool at P95)

### Memory Footprint

Each in-flight coroutine consumes approximately 2–4KB of stack space. For `Semaphore(20)` with 20 concurrent tool calls, overhead is ~80KB per agent instance — negligible.

### Throughput Scaling

Without parallelism, agent throughput is bounded by `1 / sum(tool_latencies)`. With parallelism and a semaphore of `n`, throughput scales approximately linearly with `n` until the downstream services become the bottleneck.

### Benchmark Reference

In LangChain's batch execution benchmarks (LangChain blog, 2024), moving from sequential to parallel tool invocation on a 5-tool agent reduced median latency by 68% and increased requests-per-second by 3.1× on identical infrastructure.

---

## 12. Security Considerations

**OWASP LLM Top 10 — LLM07: Insecure Plugin/Tool Design**  
Parallel tool execution increases the attack surface. When multiple tools are dispatched concurrently, an adversarially crafted input to one tool can attempt prompt injection while another tool's response is being processed. Mitigations:

- **Input sanitisation before dispatch:** Validate and sanitise all tool parameters before the fan-out — not inside each tool coroutine, where parallel execution may allow injection to reach multiple tools simultaneously.
- **Tool result isolation:** Each tool result should be treated as untrusted data until aggregated. Do not interpolate raw tool output directly into prompts; use structured schemas with Pydantic validation.
- **Semaphore as a rate-limit defence:** The concurrency semaphore also limits blast radius if one tool is compromised and begins making outbound requests — it cannot exceed `Semaphore(n)` concurrent connections.
- **Timeout as DoS mitigation:** Without per-tool timeouts, a single slow or unresponsive external endpoint can hold agent resources indefinitely. `asyncio.wait_for()` bounds resource commitment.
- **Credential isolation per tool:** Each tool should use its own scoped credentials (not a shared service account). If one tool's credential is leaked via a tool output, it cannot be used to call other tools.

---

## 13. Cost Analysis

### Token Cost

Parallel execution does not change the number of tokens consumed — the same tool calls are made regardless of execution order. However, it reduces wall-clock time, which affects cost indirectly:

- **Reduced LLM context window usage:** Faster responses mean lower probability of users re-submitting queries that seem stuck, avoiding duplicate token spend.
- **Infrastructure cost reduction:** 30–40% fewer compute resources required for the same throughput, based on the e-commerce scenario in Section 4.

### Compute Cost

| Configuration | Requests/sec (single instance) | Monthly infra cost (est.) |
|---|---|---|
| Sequential, 4 tools × 250ms | ~1.0 req/s | Baseline |
| Parallel, Semaphore(10) | ~3.5 req/s | ~0.29× baseline |
| Parallel, Semaphore(20) | ~4.2 req/s | ~0.24× baseline |

Estimates based on 4 independent tools, 250ms average each, single-threaded event loop.

### Cost vs. Accuracy Trade-off

Using fallback values for timed-out tools introduces a small accuracy cost (partial responses). This is an explicit engineering trade-off: a 2-second timeout with graceful fallback produces a useful partial response, while no timeout produces a 100%-complete response that arrives 8 seconds late (or never, in failure scenarios). For most production applications, partial-but-timely outperforms complete-but-late.

---

## 14. Best Practices

1. **Audit your dependency graph before writing async code.** Draw the data flow between tool calls. A tool that consumes another tool's output cannot run in parallel with it. This analysis takes 30 minutes and prevents hours of debugging race conditions.

2. **Always use `return_exceptions=True` with `asyncio.gather()`.** The default behaviour — raising on the first exception and abandoning remaining coroutines — is almost never correct in agent workflows. Inspect every result explicitly.

3. **Set per-tool timeout budgets based on SLA decomposition.** If your end-to-end SLA is 2 seconds and LLM processing takes ~800ms, your tool fan-out budget is ~1,200ms. Set individual tool timeouts at 80% of that budget (960ms) to leave headroom for gather overhead.

4. **Use `asyncio.Semaphore` even for small fan-outs.** Start with `Semaphore(5)` and increase only after load testing confirms downstream services can handle higher concurrency. It is much easier to relax a semaphore than to debug cascading rate-limit failures.

5. **Never use `asyncio.run_in_executor()` as a substitute for truly async tool clients.** If your tool client is synchronous (e.g., a synchronous `requests` call), use `httpx.AsyncClient` or an async SDK instead. Thread pool executors hide latency under apparent concurrency without the efficiency gains.

6. **Log per-tool latency at the result aggregation step.** Emit structured logs with `tool_name`, `latency_ms`, `success`, and `timed_out` fields. This data drives future timeout budget adjustments and identifies which tools are degrading.

7. **Handle partial results explicitly in your LLM prompt.** When one tool times out, tell the LLM explicitly: "stock_status: data unavailable — advise user to check manually." Do not silently omit the field — the LLM may hallucinate a value to fill the gap.

8. **Test timeout behaviour under load, not just in isolation.** A tool that responds in 200ms normally may consistently hit your timeout under load. Use load testing with realistic concurrency levels to validate timeout budgets before production deployment.

9. **Structure tool results as typed dataclasses or Pydantic models.** Typed results make downstream aggregation and LLM context assembly safer and easier to test. Raw dict results from async calls are difficult to distinguish from error dicts.

10. **Consider wave-based execution for mixed dependency graphs.** Rather than sequential or all-parallel, group tools into execution waves: independent tools in Wave 1 (fan-out), tools dependent on Wave 1 results in Wave 2 (fan-out among themselves), etc.

---

## 15. Anti-Patterns

### 1. The Threadpool Illusion
**What it looks like:** `loop.run_in_executor(None, requests.get, url)` for each tool call, with default executor settings.  
**Why it fails:** The default `ThreadPoolExecutor` has `min(32, os.cpu_count() + 4)` threads. With 50 concurrent tool calls, requests queue up waiting for thread slots — providing no real parallelism benefit over sequential execution.  
**Fix:** Use `httpx.AsyncClient` or other async-native HTTP clients. Reserve `run_in_executor` for CPU-bound work only.

### 2. Bare gather() Without Timeout
**What it looks like:** `await asyncio.gather(*tool_coros)` with no timeout wrappers.  
**Why it fails:** One slow or hung tool holds the entire gather() open indefinitely. In production, external APIs have intermittent hangs — this converts them from isolated failures into full agent freezes.  
**Fix:** Wrap every coroutine in `asyncio.wait_for(coro, timeout=TOOL_TIMEOUT_S)`.

### 3. Parallelising Dependent Calls
**What it looks like:** Fan-out includes `get_order_status(order_id)` and `get_order_details(order_id)` where order_id comes from a prior `search_orders(query)` call — all three dispatched together.  
**Why it fails:** `get_order_status` and `get_order_details` receive an unresolved coroutine as their argument instead of a real order_id. Causes silent None values or type errors at runtime.  
**Fix:** Run dependency-resolving tools in Wave 1, then pass their results as arguments to dependent tools in Wave 2.

### 4. Semaphore Scope Too Broad
**What it looks like:** A single global `Semaphore(5)` shared across all agent instances in the same process.  
**Why it fails:** Under load, multiple agent instances compete for the same 5 slots, serialising tool calls across instances even though each instance should have its own concurrency budget.  
**Fix:** Create the semaphore per-agent-invocation or per-request, not at module level.

### 5. Silent Exception Swallowing
**What it looks like:** `except Exception: pass` inside a tool coroutine, returning `None` on failure.  
**Why it fails:** `None` results look like valid tool responses to the aggregator and the LLM. The model generates confident answers based on missing data. Errors are invisible in logs.  
**Fix:** Return a typed `ToolErrorResult` object with the error message. Log at WARNING level. Let the aggregator handle fallback logic explicitly.

### 6. Unbounded Retry Inside Async Tool
**What it looks like:** Each tool coroutine retries up to 5 times with exponential backoff before returning.  
**Why it fails:** Retries inside the tool coroutine multiply the effective timeout. A tool with a 500ms timeout and 5 retries can run for 2.5 seconds before the outer `wait_for()` fires — breaking the latency budget.  
**Fix:** Move retry logic to the orchestration layer, outside the tool coroutine. One retry attempt is acceptable inside; exponential backoff belongs at the dispatcher level.

---

## 16. Common Mistakes

### Mistake 1: Forgetting `await` in gather() Tasks
**Symptom:** All tools appear to complete instantly; results are coroutine objects, not data.  
**Root cause:** `asyncio.gather(tool_a(), tool_b())` is correct; `asyncio.gather(tool_a, tool_b)` passes function references, not coroutines.  
**Fix:** Always call tool functions to create coroutine objects: `asyncio.gather(tool_a(), tool_b())`.

### Mistake 2: Using asyncio.gather() Outside an Event Loop
**Symptom:** `RuntimeError: no running event loop` when calling gather() from synchronous code.  
**Root cause:** `asyncio.gather()` must be called from within a coroutine. Wrapping it in `asyncio.run()` from synchronous entry points resolves this.  
**Fix:** Use `asyncio.run(dispatch_tools_parallel(...))` from synchronous orchestrators, or ensure the caller is already a coroutine (the common case in async agent frameworks).

### Mistake 3: Misinterpreting TimeoutError as a Tool Failure
**Symptom:** Monitoring shows high "tool error rate" even though tools are functioning correctly.  
**Root cause:** `asyncio.TimeoutError` is logged as a generic exception rather than a distinct event.  
**Fix:** Catch `asyncio.TimeoutError` separately from other exceptions. Emit a `tool_timeout` metric distinct from `tool_error`. Adjust timeout budgets if timeout rate exceeds 1% of calls.

---

## 17. Production Checklist

- [ ] Dependency graph documented and reviewed — all fan-out groups verified as truly independent
- [ ] All tool coroutines wrapped with `asyncio.wait_for()` and an explicit timeout value
- [ ] `asyncio.gather()` called with `return_exceptions=True` everywhere
- [ ] Semaphore created per-request (not global) with a default limit backed by load test data
- [ ] Tool result schema uses typed dataclasses or Pydantic models — no raw dicts
- [ ] Partial result handling tested: LLM prompt includes explicit "data unavailable" markers
- [ ] Per-tool latency, success, and timeout rate logged as structured metrics
- [ ] Timeout values documented with the SLA decomposition that produced them
- [ ] Load test run at 2× expected peak concurrency — timeout rates and error rates measured
- [ ] `asyncio.TimeoutError` caught and emitted as a separate metric from other exceptions
- [ ] No `asyncio.run_in_executor()` used for I/O-bound tools — replaced with async HTTP clients
- [ ] Retry logic placed at the dispatcher level, not inside individual tool coroutines
- [ ] Unit tests cover: happy path, single tool timeout, single tool error, all tools timeout
- [ ] Demo mode (`DEMO_MODE=true`) tested — no API keys required for CI pipeline

---

## 18. References

[1] Python Software Foundation (2024). "asyncio — Asynchronous I/O". Python 3.12 Documentation. https://docs.python.org/3/library/asyncio.html

[2] Python Software Foundation (2024). "asyncio.gather()". Python 3.12 Documentation. https://docs.python.org/3/library/asyncio-task.html#asyncio.gather

[3] Python Software Foundation (2024). "asyncio.wait_for()". Python 3.12 Documentation. https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for

[4] LangChain (2024). "How to invoke runnables in parallel". LangChain Documentation. https://python.langchain.com/docs/how_to/parallel/

[5] LangChain (2024). "Batch and Async". LangChain Expression Language (LCEL) Documentation. https://python.langchain.com/docs/expression_language/interface/

[6] OpenAI (2024). "Function calling". OpenAI Platform Documentation. https://platform.openai.com/docs/guides/function-calling

[7] OWASP (2023). "OWASP Top 10 for Large Language Model Applications — LLM07: Insecure Plugin Design". https://owasp.org/www-project-top-10-for-large-language-model-applications/

[8] encode/httpx (2024). "Async Support". HTTPX Documentation. https://www.python-httpx.org/async/

---

## 19. Summary

Sequential tool execution is the default in most LLM agent implementations — and it is wrong for any workflow involving multiple independent external calls. The core insight is simple: if two tool calls do not share data dependencies, they should run at the same time. The implementation requires `asyncio.gather()` with proper exception isolation (`return_exceptions=True`), per-tool timeout guards (`asyncio.wait_for()`), and a concurrency limiter (`asyncio.Semaphore`) to avoid thundering herd. The challenge is not writing the async code — it is auditing the dependency graph correctly before reaching for concurrency. Done right, this pattern reduces multi-tool agent latency from the sum of individual call times to the maximum of individual call times: a 2–4× improvement on typical 3–5 tool workflows, with the same infrastructure cost.

---

## 20. Exercises

**Beginner:** Run the PoC in demo mode (`DEMO_MODE=true python src/main.py`). Observe the output. Change the `TOOL_TIMEOUT_S` in `.env.example` to 0.1 seconds and note how the output changes when all tools time out.

**Intermediate:** Modify `parallel_tools_core.py` to track the wall-clock time of each wave separately (Wave 1 fan-out time vs. total time). Add a print statement to `main.py` that reports the observed speedup ratio compared to a sequential baseline computed from the sum of individual tool latencies.

**Advanced:** Extend the PoC to implement a two-wave execution plan: Wave 1 calls `get_order_id(customer_email)`, then Wave 2 fans out `get_order_details(order_id)`, `get_shipment_status(order_id)`, and `get_invoice(order_id)` in parallel. Verify that Wave 2 only starts after Wave 1 completes.

**Expert:** Benchmark the PoC with 1, 5, 10, and 20 concurrent agent invocations using `asyncio.gather()` at the harness level (not just within a single agent). Plot throughput (requests/sec) and P95 latency at each concurrency level. Identify the point at which the semaphore becomes the bottleneck.

**Research:** Read the LangChain LCEL documentation on batch and parallel execution (Reference [5]). Identify one limitation of LCEL's parallel execution model that is not present in a hand-rolled `asyncio.gather()` implementation. Propose a workaround.

---

## 21. Interview Questions

**Conceptual**
1. Explain async parallel tool dispatch to a non-engineer using an analogy. What is the equivalent of a "semaphore" in that analogy?
2. Why does `return_exceptions=False` in `asyncio.gather()` make it almost unusable in production agentic workflows?

**Technical**
3. What is the difference between `asyncio.gather()` and `asyncio.wait()`? When would you choose `wait()` over `gather()` in a tool dispatch scenario?
4. A tool coroutine uses `requests.get()` internally and you wrap it in `asyncio.run_in_executor()`. Under what conditions does this still perform worse than a sequential implementation?

**Design**
5. Design a tool dispatcher for an agent that must call 8 tools, where tools T3 and T6 depend on the output of T1, and T7 depends on the output of T3. Draw the execution waves and explain how you would structure the `asyncio.gather()` calls.
6. How would you implement circuit breaker logic on top of an async tool dispatcher so that a tool with a >20% failure rate in the last 60 seconds is automatically bypassed?

**Trade-off**
7. When would you intentionally keep tool calls sequential despite them being independent? Give two concrete production scenarios.
8. Compare the trade-offs of setting a tight tool timeout (500ms) versus a loose one (5,000ms) for a downstream API that averages 300ms but occasionally spikes to 3 seconds.

**Debugging**
9. An agent in production has a P95 latency of 4 seconds despite all tools averaging 300ms. Async parallelism is already implemented. What are your top three hypotheses, and how would you diagnose each?
10. Your monitoring shows that `tool_timeout` events are spiking every day between 2–3 PM. The same tools succeed at other times. What is the most likely root cause, and what mitigation would you implement?
