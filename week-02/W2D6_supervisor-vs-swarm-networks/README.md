# W2D6 — Supervisor vs. Swarm Networks

> [Week 2](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

Multi-agent systems can be wired in fundamentally different topologies. The **Supervisor Network** (hub-and-spoke) centralises control: a coordinator decomposes the task, routes each subtask to the best-matching specialist via keyword scoring, and aggregates results. The **Swarm Network** (peer-to-peer mesh) decentralises routing: each agent decides whether to handle a message or pass it on, with cycle prevention via routing history and a dead-letter queue for unroutable messages. This PoC runs both topologies against the same customer-support workflow and prints routing traces side-by-side, making delegation decisions visible and auditable.

---

## Learning Objectives

1. Distinguish hub-and-spoke (Supervisor) from peer-to-peer (Swarm) multi-agent topologies and their coordination trade-offs
2. Implement `SupervisorNetwork.decompose()` to break a task into self-contained subtask instructions
3. Implement `_find_best_agent()` using `match_score()` keyword counting to select the most specialised agent
4. Implement `SwarmNetwork._route_message()` with cycle prevention (`routing_history`) and `dead_letter_queue` for unroutable messages
5. Understand when to choose Supervisor (sequential tasks, compliance audit trail, shared world model) vs. Swarm (independent parallel subtasks, horizontal scale, low-latency fan-out)
6. Extend the agent pool by subclassing `Agent`, declaring a `capability` string, and implementing `handle()`
7. Interpret `WorkflowResult.routing_trace` to debug orchestration decisions and identify routing bottlenecks

---

## Problem Statement

Multi-agent systems that hardcode routing logic break when task structure changes. A sequential pipeline assumes a fixed order — it misses opportunities for parallelism and adds unnecessary latency when subtasks are independent. Choosing the wrong topology at design time creates bottlenecks (a supervisor that serialises parallelisable work), coordination failures (a swarm cycling between agents without finding a capable handler), or silent task loss (subtasks that fall through with no audit trail). In production, the cost shows up as latency spikes, orphaned subtasks, and compliance tasks silently dropped with nothing to debug from.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — all demos run offline with `DEMO_MODE=true`)
- Familiarity with [W1D6 — State Graphs (LangGraph)](../../week-01/W1D6-langgraph-state-graphs/README.md) helpful but not required

---

## Repository Structure

```
W2D6_supervisor-vs-swarm-networks/
├── README.md                              # This file
├── docs/
│   ├── technical-document.md              # 21-section practitioner deep-dive
│   └── supervisor-vs-swarm-layman-scenarios.md
├── diagrams/
│   ├── architecture.mmd                   # Hub-and-spoke vs. mesh topology diagram
│   └── sequence.mmd                       # Step-by-step routing sequence
└── poc/
    ├── README.md                          # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                        # Entry point — demo + live mode
    │   ├── swarm_core.py                  # SupervisorNetwork, SwarmNetwork, Agent base class
    │   └── config.py                      # Config loaded from environment variables
    ├── tests/
    │   └── test_swarm.py                  # pytest unit tests (4 test classes, 18+ tests)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json
    └── sample_output.json
```

---

## Core Concepts

### Supervisor Network (hub-and-spoke)

The Supervisor decomposes the high-level task into named subtask instructions, scores all agents against each subtask, and dispatches to the best match:

```python
# swarm_core.py
class SupervisorNetwork:
    def decompose(self, task: str) -> list[str]:
        subtasks = []
        if any(w in task.lower() for w in ["find", "search", "retrieve"]):
            subtasks.append("retrieve relevant documents and data")
        if any(w in task.lower() for w in ["analyse", "sentiment"]):
            subtasks.append("analyse content and classify sentiment")
        if any(w in task.lower() for w in ["write", "generate", "respond"]):
            subtasks.append("generate a written response")
        if any(w in task.lower() for w in ["check", "validate", "compliance"]):
            subtasks.append("validate output for compliance")
        return subtasks or ["analyse and generate a response"]

    def _find_best_agent(self, subtask: str) -> Agent:
        scored = [(agent.match_score(subtask), agent) for agent in self.agents]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_agent = scored[0]
        return best_agent if best_score > 0 else self.agents[0]
```

### Swarm Network (peer-to-peer)

The Swarm routes each part of a semicolon-delimited task independently. `routing_history` blocks revisiting the same agent; messages that exceed `max_hops` or find no capable handler land in `dead_letter_queue`:

```python
# swarm_core.py
class SwarmNetwork:
    def _route_message(self, subtask, routing_history, trace) -> AgentResult | None:
        if len(routing_history) >= self.max_hops:
            return None  # → dead_letter_queue

        for agent in self.agents:
            if agent.name in routing_history:
                continue  # cycle prevention
            if agent.can_handle(subtask):
                routing_history.append(agent.name)
                return agent.handle(subtask, demo_mode=True)
        return None  # → dead_letter_queue
```

### Adding a specialist agent

```python
from swarm_core import Agent, AgentResult, SupervisorNetwork, SwarmNetwork, DEFAULT_AGENTS
import time

class MySpecialist(Agent):
    name = "my-specialist"
    capability = "translate localise language conversion"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        start = time.monotonic()
        output = f"[MySpecialist] Translated: '{subtask}'"
        return AgentResult(
            agent_name=self.name, subtask=subtask, output=output,
            success=True, latency_ms=round((time.monotonic() - start) * 1000, 2),
            tokens_used=30,
        )

agents = DEFAULT_AGENTS + [MySpecialist()]
supervisor = SupervisorNetwork(agents=agents)
swarm = SwarmNetwork(agents=agents)
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-02/W2D6_supervisor-vs-swarm-networks/poc
pip install -r requirements.txt
cp .env.example .env
DEMO_MODE=true python src/main.py
```

### Live mode

```bash
# Edit .env and set OPENAI_API_KEY=your-key
python src/main.py
```

### Tests

```bash
pytest tests/ -v
```

All 18+ tests pass offline. No API key required.

---

## Expected Output

```
W2D6 -- Supervisor vs. Swarm Networks Demo
==================================================
Task: retrieve customer purchase history; analyse sentiment...

[DEMO MODE] Running in demo mode -- no API key required.

--- SUPERVISOR NETWORK ---
Subtasks handled : 4
Total latency    : 1.2 ms
Total tokens     : 223
Routing trace:
  [Supervisor] Decomposed into 4 subtask(s): ['retrieve relevant documents and data', ...]
  [Supervisor] Dispatching 'retrieve relevant documents and data' -> retrieval-agent
  [Supervisor] Received result from retrieval-agent (success=True, latency=0.1ms)
  [Supervisor] Dispatching 'analyse content and classify sentiment' -> analysis-agent
  [Supervisor] Received result from analysis-agent (success=True, latency=0.1ms)
  [Supervisor] Dispatching 'generate a written response' -> generation-agent
  [Supervisor] Received result from generation-agent (success=True, latency=0.1ms)
  [Supervisor] Dispatching 'validate output for compliance' -> validation-agent
  [Supervisor] Received result from validation-agent (success=True, latency=0.1ms)
  [Supervisor] Aggregation complete. Total latency: 1.2ms, tokens: 223

--- SWARM NETWORK ---
Subtasks handled : 4
Total latency    : 0.9 ms
Total tokens     : 223
Routing trace:
  [Swarm] Received 4 message(s) for routing
  [Swarm] 'retrieve customer purchase history' -> retrieval-agent (hop 1)
  [Swarm] 'analyse sentiment of recent feedback' -> analysis-agent (hop 1)
  [Swarm] 'generate a personalised response' -> generation-agent (hop 1)
  [Swarm] 'validate response for compliance' -> validation-agent (hop 1)
  [Swarm] Complete. Handled: 4, DLQ: 0, Total latency: 0.9ms, tokens: 223

--- COMPARISON ---
Supervisor latency : 1.2 ms
Swarm latency      : 0.9 ms
Faster topology    : Swarm (on this workload)

[OK] Concept demonstrated: Supervisor routes via central decomposition;
     Swarm routes via peer-to-peer capability matching.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — runs demo or live mode, prints routing traces and comparison |
| `src/swarm_core.py` | `SupervisorNetwork`, `SwarmNetwork`, `Agent` base class, four specialist agents, `AgentResult`, `WorkflowResult` |
| `src/config.py` | `Config` dataclass + `load_config()` reading all settings from env vars |
| `tests/test_swarm.py` | 4 test classes (demo mode, core concept, live mode, sample files), 18+ tests |
| `sample_input.json` | Customer-support workflow task with 4 semicolon-delimited subtasks |
| `sample_output.json` | Expected JSON output containing routing traces for both topologies |
| `.env.example` | All environment variable defaults — copy to `.env` before running live mode |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully offline in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `MODEL` | `gpt-4o-mini` | LLM model used in live mode |
| `DEMO_MODE` | `false` | Set `true` to skip all API calls |
| `SUPERVISOR_TIMEOUT` | `10.0` | Supervisor subtask timeout in seconds |
| `MAX_SUBTASK_RETRIES` | `2` | Max retries per failed subtask |
| `SWARM_MAX_HOPS` | `5` | Max routing hops before a message enters the dead-letter queue |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/supervisor-vs-swarm-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — Hub-and-spoke vs. mesh topology
- [Sequence Diagram](diagrams/sequence.mmd) — Step-by-step routing sequence for both topologies

---

## Connection to the Series

**Previous:** [W2D5 — Reflection & Self-Correction Loops](../W2D5_reflection-self-correction-loops/README.md) — agents that critique and revise their own outputs through a Generate → Critique → Revise loop.

**Next:** [W2D7 — Deterministic Guardrails (NeMo)](../W2D7_deterministic-guardrails-nemo/README.md) — add deterministic, rule-based safety enforcement that fires on every request in any multi-agent topology.

**Series arc:** [W1D6 — State Graphs (LangGraph)](../../week-01/W1D6-langgraph-state-graphs/README.md) introduced LangGraph state graphs for single-agent orchestration. W2D6 scales up to multi-agent network topologies — supervisor and swarm — that compose those state graphs into coordinated networks. [W3D6 — Hierarchical Subagent Teams](../../week-03/README.md) will extend this to nested hierarchies.

---

## Key References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [CrewAI Documentation](https://docs.crewai.com/)

---

## Continue Learning

**Next:** [W2D7 — Deterministic Guardrails (NeMo)](../W2D7_deterministic-guardrails-nemo/README.md)

Return to [Week 2 overview](../README.md) to explore all intermediate patterns.
