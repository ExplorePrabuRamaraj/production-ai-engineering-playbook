# W2D6 -- Supervisor vs. Swarm Networks

**Series:** AI Engineering Production Playbook
**Vertical:** Multi-Agent Orchestration
**Week 2 / Day 6**

## What This Demonstrates

Two multi-agent orchestration topologies -- Supervisor Network (hub-and-spoke) and Swarm Network (peer-to-peer) -- running the same task side-by-side, with routing traces showing exactly how each topology makes delegation decisions.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional -- demo mode runs fully offline)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-02/W2D6_supervisor-vs-swarm-networks/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API key, or leave OPENAI_API_KEY blank for demo mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

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
  [Supervisor] Decomposed into 4 subtask(s): [...]
  [Supervisor] Dispatching '...' -> retrieval-agent
  ...

--- SWARM NETWORK ---
Subtasks handled : 4
Total latency    : 0.9 ms
Total tokens     : 223
Routing trace:
  [Swarm] Received 4 message(s) for routing
  [Swarm] 'retrieve customer purchase history' -> retrieval-agent (hop 1)
  ...

--- COMPARISON ---
Supervisor latency : 1.2 ms
Swarm latency      : 0.9 ms
Faster topology    : Swarm (on this workload)

[OK] Concept demonstrated: Supervisor routes via central decomposition;
     Swarm routes via peer-to-peer capability matching.
```

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline. No API key required.

## File Structure

```
poc/
  src/
    main.py           # Entry point -- run this file
    swarm_core.py     # SupervisorNetwork, SwarmNetwork, Agent base class
    config.py         # Config dataclass + env loader
  tests/
    test_swarm.py     # pytest unit tests (4 test classes, 18+ tests)
  requirements.txt
  .env.example
  sample_input.json
  sample_output.json
```

## Key Concepts

**SupervisorNetwork** (`swarm_core.py`)
- Decomposes a task into subtasks via `decompose()`
- Routes each subtask to the best-matching specialist via `_find_best_agent()`
- Aggregates all results and returns a `WorkflowResult`

**SwarmNetwork** (`swarm_core.py`)
- Accepts semicolon-delimited multi-part tasks
- Routes each part peer-to-peer via `_route_message()`
- Prevents routing cycles via `routing_history` tracking
- Captures unroutable messages in `dead_letter_queue`

**When to use which:**
- Supervisor: sequential tasks, shared world model, compliance audit requirements
- Swarm: independent parallel subtasks, horizontal scale, low-latency fan-out
- Hybrid: Supervisor decomposes + Swarm executes independent branches

## Extend This Demo

Add a new specialist agent:

```python
from swarm_core import Agent, AgentResult
import time

class MySpecialist(Agent):
    name = "my-specialist"
    capability = "translate localise language conversion"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        start = time.monotonic()
        output = f"[MySpecialist] Translated: '{subtask}'"
        return AgentResult(
            agent_name=self.name,
            subtask=subtask,
            output=output,
            success=True,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            tokens_used=30,
        )
```

Register it:

```python
from swarm_core import SupervisorNetwork, SwarmNetwork, DEFAULT_AGENTS

agents = DEFAULT_AGENTS + [MySpecialist()]
supervisor = SupervisorNetwork(agents=agents)
swarm = SwarmNetwork(agents=agents)
```

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [LinkedIn Post](../README.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
