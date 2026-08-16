# W2D6 — Supervisor vs. Swarm Networks
## Technical Deep Dive: Multi-Agent Orchestration Topologies

**Series:** AI Engineering Production Playbook
**Vertical:** Multi-Agent Orchestration
**Week 2 / Day 6**

---

## 1. Overview

As LLM-powered systems grow beyond a single reasoning agent, engineers must choose how multiple agents coordinate their work. Two dominant topologies have emerged: **Supervisor Networks**, where a central coordinator delegates tasks and aggregates results, and **Swarm Networks**, where agents communicate peer-to-peer and autonomously route work. The choice between them is not a matter of preference — it is determined by the dependency structure of the workflow being automated. Both patterns are production-relevant today because single-agent systems consistently fail under load, complexity, and reliability requirements that multi-agent architectures can satisfy.

---

## 2. Learning Objectives

By the end of this document you will be able to:

1. **Explain** the structural difference between Supervisor and Swarm topologies to a non-specialist.
2. **Distinguish** which topology is appropriate for a given task dependency graph.
3. **Design** a Supervisor Network with explicit state management and error recovery.
4. **Design** a Swarm Network with idempotent message handling and cycle prevention.
5. **Evaluate** the latency, reliability, and cost trade-offs of each topology.
6. **Identify** the five most common anti-patterns in multi-agent orchestration.
7. **Implement** a hybrid architecture combining both topologies.
8. **Apply** production checklist items before deploying any multi-agent system.

---

## 3. Problem Statement

Single-agent LLM systems are inherently serial: one prompt in, one response out. This works for simple tasks but fails under three production pressures:

**Complexity:** A customer support automation that must simultaneously retrieve account history, check inventory, and draft a personalised response cannot do all three with a single LLM call. Attempts to stuff all context into one prompt increase token cost, reduce accuracy, and hit context window limits.

**Parallelism:** Serial execution of independent subtasks wastes wall-clock time. An agent that runs three 2-second subtasks one after another takes 6 seconds; a multi-agent system running them in parallel takes 2 seconds — a 3x throughput improvement.

**Specialisation:** General-purpose agents produce mediocre outputs across all domains. Specialised agents — one for code generation, one for data retrieval, one for natural language summarisation — consistently outperform generalists on their target tasks by 15-30% on standard benchmarks (ReAct, HotpotQA).

The failure mode in production is subtle: engineers often start with a single God Agent that does everything, then add complexity through prompt engineering until the agent becomes unreliable, expensive, and impossible to debug. The system works in demos and fails in production.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: The Overwhelmed God Agent

A legal technology company builds an automated contract review system. A single agent receives a 40-page contract and must: identify governing law clauses, flag non-standard indemnification terms, verify party names against a CRM, check dates for consistency, and produce an executive summary. The agent is given a single system prompt with instructions for all five tasks.

In production, the system correctly identifies governing law 94% of the time when run on short contracts. On long contracts (>30 pages), accuracy drops to 61%. The root cause: the agent's attention is split across five competing objectives, and position-dependent context decay (the "lost in the middle" problem from W1D2) causes it to miss clauses in the document's middle sections. Additionally, the CRM lookup requires a tool call, which occasionally times out, causing the entire review to fail.

Mean review time: 47 seconds. Error rate: 23% on full reviews. On-call engineers spend 4 hours per week manually correcting outputs.

### Scenario B — The Solution: Supervisor + Specialist Swarm

The same company redesigns using a Supervisor Network with five specialist sub-agents. The Supervisor receives the contract, decomposes it into five independent tasks, and dispatches each to a specialist. The CRM lookup agent runs independently and returns a partial result if it times out — the Supervisor continues with a warning rather than halting the pipeline.

Mean review time: 12 seconds (three of the five tasks run in parallel via the Swarm execution mode within the Supervisor's dispatch layer). Error rate: 7% on full reviews. The Supervisor's state log provides a complete audit trail for every decision.

Improvement: 74% latency reduction, 70% error rate reduction, zero manual corrections required per week.

---

## 5. Solution Architecture

### Supervisor Network Architecture

A Supervisor Network is a **hub-and-spoke** topology. The Supervisor agent holds the master task state and is the only agent with write access to the final output. It operates in three phases:

1. **Decomposition:** The Supervisor analyses the user request and produces a task plan — an ordered or partially-ordered list of subtasks, each with an assigned specialist and dependency constraints.
2. **Dispatch:** The Supervisor sends each subtask to its specialist agent. Tasks without dependencies on each other are dispatched concurrently.
3. **Aggregation:** As specialists return results, the Supervisor merges them into a coherent response, resolves conflicts, and handles partial failures.

The Supervisor is the single point of authority. This makes the system auditable but creates a bottleneck: every result must pass through the Supervisor before the workflow can advance.

### Swarm Network Architecture

A Swarm Network is a **mesh** topology. There is no central coordinator. Each agent in the swarm:

1. **Receives** a message from a shared message bus or directly from another agent.
2. **Decides** whether the message falls within its competency using a routing function (typically an LLM call or a rule-based classifier).
3. **Handles** the message and produces a result, OR **forwards** it to another agent.

Swarms excel at fan-out tasks where subtasks are independent and can be processed by whichever available agent has capacity. They scale horizontally: adding agents to a swarm increases throughput linearly (up to message bus limits).

The weakness is coordination: without explicit state management, swarms can produce duplicate work, circular routing, or conflicting outputs on shared resources.

### Hybrid Architecture

Production systems typically use a **Supervisor for task decomposition and dependency management** combined with **Swarm execution for independent subtasks**. The Supervisor holds the DAG (Directed Acyclic Graph) of task dependencies; once it identifies a set of independent tasks (nodes with no unresolved dependencies), it dispatches them to a swarm for concurrent execution.

---

## 6. Internal Working Mechanics

### Supervisor State Machine

The Supervisor maintains an explicit task state object:

```python
@dataclass
class TaskState:
    task_id: str
    subtasks: list[Subtask]        # All planned subtasks
    completed: dict[str, Result]   # Completed subtask results
    pending: list[str]             # Subtask IDs awaiting dispatch
    in_flight: list[str]           # Subtask IDs currently executing
    failed: dict[str, str]         # Subtask ID -> error message
    final_result: str | None       # Aggregated output
```

At each step, the Supervisor:
1. Evaluates which pending tasks have all dependencies in `completed`.
2. Moves eligible tasks from `pending` to `in_flight` and dispatches them.
3. On completion, moves the result from `in_flight` to `completed`.
4. On failure, applies the retry or fallback policy and moves to `failed`.
5. When `pending` and `in_flight` are both empty, aggregates `completed` into `final_result`.

This explicit state makes the system debuggable: at any point, the Supervisor's state object fully describes where the workflow is and why.

### Swarm Routing Function

Each swarm agent implements a routing function:

```python
def route(message: Message) -> RoutingDecision:
    if can_handle(message):
        return RoutingDecision(action="handle", agent=self)
    else:
        return RoutingDecision(action="forward", agent=find_next_agent(message))
```

`can_handle` is typically implemented as:
- A rule-based classifier (fast, deterministic, no LLM cost)
- A lightweight LLM call with a binary yes/no output (flexible, adds ~100ms latency)
- A vector similarity check against the agent's capability embedding

Cycle prevention is critical. Swarms must track message routing history and reject messages that have visited the same agent twice:

```python
if self.agent_id in message.routing_history:
    raise RoutingCycleError(f"Cycle detected: {message.routing_history}")
message.routing_history.append(self.agent_id)
```

### Message Bus Requirements

Swarm networks require a message bus that guarantees:
- **At-least-once delivery:** No messages are silently dropped.
- **Idempotency keys:** Duplicate deliveries are detected and suppressed.
- **Dead letter queue:** Messages that cannot be routed after N attempts are captured for inspection.

In-process implementations (e.g., Python asyncio queues) satisfy these requirements for single-node deployments. Distributed swarms require a proper message broker (Redis Streams, Kafka, or a managed service).

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd`.

The diagram shows a Hybrid architecture: a Supervisor receiving the user request, decomposing it, dispatching independent subtasks to a swarm of specialists, and aggregating results. The message bus and dead letter queue are shown as infrastructure components.

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd`.

The sequence diagram shows the full lifecycle of a request through a Supervisor Network: task decomposition, parallel dispatch to two specialist agents, result aggregation, and the error handling path when one specialist times out.

---

## 9. Implementation Guide

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure environment

```bash
cp .env.example .env
# Set OPENAI_API_KEY or leave blank for demo mode
```

### Step 3: Run the demonstration

```bash
# Demo mode (no API key required)
DEMO_MODE=true python src/main.py

# Live mode
python src/main.py
```

### Step 4: Understand the core modules

**`src/swarm_core.py`** — contains three classes:
- `Agent`: Base class representing a single agent in either topology. Holds a name, capability description, and `handle()` method.
- `SupervisorNetwork`: Orchestrates a set of specialist agents. The `run()` method decomposes a task, dispatches to specialists, and aggregates results.
- `SwarmNetwork`: Routes messages peer-to-peer using a capability-matching function. The `route()` method finds the best-matching agent and delegates.

**`src/config.py`** — loads all environment variables into a typed `Config` dataclass.

**`src/main.py`** — demonstrates both topologies on the same task, prints side-by-side latency results, and explains the routing decision made.

### Step 5: Extend for your use case

To add a new specialist agent:

```python
from swarm_core import Agent

class MySpecialist(Agent):
    name = "my-specialist"
    capability = "Handles requests about [specific domain]"

    def handle(self, task: str) -> str:
        # Your implementation here
        return result
```

Register it with either network:
```python
supervisor = SupervisorNetwork(agents=[existing_agent, MySpecialist()])
swarm = SwarmNetwork(agents=[existing_agent, MySpecialist()])
```

---

## 10. Benefits and Trade-offs

| Benefit | Trade-off |
|---|---|
| Supervisor: clean audit trail, full state visibility | Supervisor: central bottleneck limits throughput |
| Supervisor: predictable task ordering and retry policy | Supervisor: Supervisor LLM cost added on every workflow |
| Swarm: horizontal scalability, lower median latency | Swarm: emergent routing failures are hard to reproduce |
| Swarm: fault isolation — one agent failure does not halt the pipeline | Swarm: shared state requires explicit concurrency control |
| Hybrid: combines scalability with auditability | Hybrid: higher architectural complexity, two failure modes to monitor |

---

## 11. Performance Characteristics

### Latency

**Supervisor (serial dispatch):**
- P50: sum of all subtask latencies + Supervisor overhead (~200ms per dispatch cycle)
- P95: P50 + longest-tail subtask latency + one retry cycle

**Supervisor (parallel dispatch):**
- P50: max(subtask latencies) + Supervisor overhead
- P95: max(P95 of subtask latencies) + one retry cycle

**Swarm:**
- P50: routing overhead (50-150ms per hop) + task execution latency
- P95: routing overhead * max_hops + task execution P95

**Typical production measurements** (from LangGraph benchmark reports, 2024):
- 5-agent Supervisor (parallel): 1.2s median end-to-end for a document processing workflow
- 5-agent Swarm: 0.7s median for the same workflow with independent subtasks
- Hybrid: 0.9s median, combining decomposition overhead with parallel execution

### Memory

- Supervisor state object: O(n) where n = number of subtasks. Typically 2-10KB per workflow.
- Swarm message routing history: O(h) where h = number of hops. Typically <1KB per message.

### Throughput Scaling

- Supervisor is bounded by the Supervisor agent's concurrency limit.
- Swarm scales linearly with agent count up to message bus throughput limits.
- At 100 concurrent workflows, a Supervisor cluster requires horizontal scaling of the Supervisor itself (stateful, requires session affinity or external state store).

---

## 12. Security Considerations

### OWASP LLM Top 10 Relevance

**LLM01 — Prompt Injection:** In a Supervisor Network, a malicious input to one specialist can attempt to inject instructions that redirect the Supervisor's routing decisions. Mitigate by sanitising all specialist outputs before the Supervisor processes them, and by validating that the Supervisor's next-action decisions match the allowed action set.

**LLM06 — Sensitive Information Disclosure:** Swarm agents may inadvertently forward messages containing PII or confidential data to agents that do not need it. Implement message-level access control: each message carries a data classification label, and agents reject messages above their clearance level.

**LLM08 — Excessive Agency:** In a Swarm, agents can forward messages to any peer agent, including tool-calling agents with write access to external systems. Implement an allowlist of permitted routing paths and reject any forwarding that deviates from the allowlist.

### Additional Controls

- Validate all inter-agent message schemas with Pydantic before processing.
- Log every agent-to-agent message with timestamp, source, destination, and message hash for forensic replay.
- Apply rate limiting per agent to prevent one misbehaving agent from flooding the swarm.

---

## 13. Cost Analysis

### Token Cost per Workflow (estimated, gpt-4o-mini at $0.15/1M input tokens)

| Configuration | Subtasks | Supervisor Calls | Specialist Calls | Approx. Cost |
|---|---|---|---|---|
| Single God Agent | 1 | 0 | 1 x 2000 tokens | $0.0003 |
| Supervisor (5 specialists) | 5 | 1 x 800 tokens | 5 x 400 tokens | $0.0004 |
| Swarm (5 agents, 2 hops avg) | 5 | 0 | 5 x 400 + routing | $0.0004 |
| Hybrid (5 specialists, parallel) | 5 | 1 x 800 tokens | 5 x 400 tokens | $0.0004 |

The cost difference between topologies is small (less than $0.0001 per workflow). The dominant cost driver is the number and length of specialist LLM calls, not the orchestration overhead. Choose topology based on reliability and latency requirements, not cost.

**Cost vs. accuracy trade-off:** Investing in 5 specialised agents rather than 1 general agent typically yields 15-25% accuracy improvement on complex tasks. For most production workloads the accuracy gain vastly outweighs the marginal token cost increase.

---

## 14. Best Practices

1. **Model your task as a dependency graph first.** Before choosing a topology, draw the DAG of your subtasks. Tasks with dependencies map to Supervisor edges; independent tasks map to Swarm dispatch.

2. **Keep specialist agents narrow.** A specialist that handles one well-defined task type produces better outputs and is easier to test than one handling two or three related tasks. The cognitive load analogy applies: narrower scope means fewer competing instructions in the system prompt.

3. **Make the Supervisor stateless across workflows.** Store task state in an external store (Redis, DynamoDB) rather than in the Supervisor agent's memory. This enables horizontal scaling and workflow recovery after crashes.

4. **Implement idempotency keys on all messages.** Every message in a Swarm must carry a unique ID. Receiving agents check whether the message ID has been processed before handling. This prevents duplicate execution when messages are redelivered.

5. **Set explicit timeouts on every subtask.** A subtask that never returns will block the Supervisor indefinitely. Set a per-subtask timeout and handle the timeout as a first-class error state with its own fallback policy.

6. **Use structured output schemas for inter-agent messages.** Define a Pydantic model for every message type. Agents that receive malformed messages should return a typed error, not raise an unhandled exception.

7. **Log every routing decision with full context.** For both Supervisor dispatch and Swarm routing, log: timestamp, source agent, destination agent, task summary, and decision rationale. This makes post-hoc debugging possible in 10 minutes rather than 10 hours.

8. **Test your swarm with adversarial routing inputs.** Inject messages designed to cause routing cycles, exceeded hop counts, and unroutable tasks. Verify that the dead letter queue captures them and alerts fire.

9. **Benchmark both topologies on your specific workload.** Published latency numbers are for reference; actual numbers depend on your task structure, model choice, and network topology. Measure before committing to an architecture.

10. **Start with a Supervisor, add Swarm execution incrementally.** Supervisors are easier to debug and reason about. Once the system is stable, identify independent subtasks and migrate their dispatch to parallel Swarm execution.

---

## 15. Anti-Patterns

### The God Agent
**What it looks like:** A single agent with a 3,000-token system prompt covering 8 distinct task types.
**Why it fails:** Competing instructions degrade output quality; any failure requires debugging one monolithic prompt; no parallelism is possible.
**What to do instead:** Decompose into specialised agents with narrow, focused system prompts.

### The Chatty Supervisor
**What it looks like:** The Supervisor makes an LLM call to decide the next action after every single subtask completion, even when the plan is predetermined.
**Why it fails:** Adds 200-500ms of latency per step for no reasoning benefit; burns tokens on unnecessary calls.
**What to do instead:** Pre-compute the full task plan at decomposition time. Only invoke the Supervisor's LLM for genuine re-planning decisions triggered by unexpected subtask outcomes.

### The Stateless Swarm
**What it looks like:** Swarm agents forward messages without tracking routing history. No dead letter queue. No idempotency keys.
**Why it fails:** Routing cycles cause infinite loops; duplicate deliveries cause duplicate work; unroutable messages disappear silently.
**What to do instead:** Enforce routing history tracking, max-hop limits, and a dead letter queue from day one.

### The Shared Mutable State Swarm
**What it looks like:** Multiple swarm agents write to the same document, database record, or in-memory object without locks.
**Why it fails:** Race conditions produce corrupted outputs. The corruption is often silent — no exception is raised, the output is just wrong.
**What to do instead:** Use event sourcing or a CRDT (Conflict-free Replicated Data Type) for shared state, or route all writes through a single designated writer agent.

### The Depth-First Swarm
**What it looks like:** Each agent forwards to the next agent only after completing its full task. The workflow is effectively serial despite using multiple agents.
**Why it fails:** No parallelism benefit; all the routing overhead with none of the latency improvement.
**What to do instead:** Design agents to accept partial inputs and produce partial outputs. Use a fan-out dispatch that sends to all relevant agents simultaneously.

### The Missing Fallback Supervisor
**What it looks like:** The Supervisor has no policy for what to do when a specialist returns an error or times out. The entire workflow fails on any subtask failure.
**Why it fails:** In production, external dependencies fail regularly. A system with no partial-result handling has the reliability of its least reliable dependency.
**What to do instead:** Define a fallback result for each subtask type. Allow the Supervisor to complete with partial results and indicate which components are missing.

---

## 16. Common Mistakes

### Mistake 1: Choosing topology before understanding task structure
**Symptom:** The system is slow despite using a Swarm topology.
**Root cause:** The tasks are not actually independent — each task needs the output of the previous one. A Swarm on a dependent task graph executes serially anyway, but with added routing overhead.
**Fix:** Map task dependencies before choosing a topology. If tasks form a chain, use a Supervisor or a simple pipeline. Reserve Swarms for genuinely fan-out workloads.

### Mistake 2: No circuit breaker on specialist agents
**Symptom:** One slow specialist causes cascading latency across all workflows.
**Root cause:** Without a circuit breaker, the Supervisor keeps dispatching to a degraded specialist, filling the in-flight queue and starving other workflows.
**Fix:** Implement a circuit breaker per specialist: if the failure rate exceeds a threshold (e.g., 50% in a 60-second window), the Supervisor routes to a fallback agent or returns a partial result.

### Mistake 3: Testing only the happy path
**Symptom:** System works in staging, fails in production on edge cases.
**Root cause:** Tests only cover the case where all specialists succeed within their time budget. Production has timeouts, malformed outputs, and unexpected empty results.
**Fix:** Write explicit tests for: specialist timeout, specialist returning an empty result, specialist returning a schema-invalid result, routing cycle in swarm, max hops exceeded.

---

## 17. Production Checklist

- [ ] Task dependency graph documented and reviewed before topology selection
- [ ] Supervisor state persisted to external store (not in-process memory)
- [ ] All inter-agent messages use Pydantic-validated schemas
- [ ] Idempotency keys present on all Swarm messages
- [ ] Per-subtask timeouts configured with explicit fallback policies
- [ ] Dead letter queue configured and monitored with alerting
- [ ] Routing history tracked per message; max-hop limit enforced
- [ ] Circuit breaker per specialist agent with configurable threshold
- [ ] All agent-to-agent messages logged with source, destination, and decision rationale
- [ ] Swarm tested with adversarial inputs: cycles, unroutable tasks, duplicate deliveries
- [ ] Load test at 2x expected peak concurrency before production launch
- [ ] Graceful degradation tested: system returns partial result when one specialist fails
- [ ] PII and sensitive data classification enforced at message level
- [ ] Runbook documented: how to inspect dead letter queue, restart a stalled workflow, roll back a bad deployment

---

## 18. References

[1] Wu et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv:2308.08155. https://arxiv.org/abs/2308.08155

[2] LangChain (2024). "LangGraph: Build Stateful, Multi-Actor Applications with LLMs." Official Documentation. https://langchain-ai.github.io/langgraph/

[3] Anthropic (2025). "Building Effective Agents." Anthropic Engineering Blog. https://www.anthropic.com/research/building-effective-agents

[4] OpenAI (2025). "A Practical Guide to Building Agents." OpenAI Documentation. https://platform.openai.com/docs/guides/agents

[5] Hong et al. (2023). "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." arXiv:2308.00352. https://arxiv.org/abs/2308.00352

---

## 19. Summary

Single-agent LLM systems fail on complex, parallelisable workloads because they are inherently serial and cannot specialise. Supervisor Networks solve this by introducing a central coordinator that decomposes tasks, dispatches to specialists, and aggregates results — providing full auditability at the cost of a coordination bottleneck. Swarm Networks eliminate the bottleneck with peer-to-peer routing, enabling horizontal scalability at the cost of harder debugging and explicit concurrency management. The production answer is almost always a hybrid: a Supervisor holds the task dependency graph and manages state, while a Swarm executes independent branches in parallel. The choice of topology should be derived from the task's dependency structure, not from architectural preference.

---

## 20. Exercises

**Beginner:** Run `DEMO_MODE=true python src/main.py` and observe the output. Identify which subtasks the Supervisor dispatches in parallel vs. serially.

**Intermediate:** Modify `src/swarm_core.py` to add a fourth specialist agent for a domain of your choice. Run both topologies and compare the routing decision output.

**Advanced:** Implement a circuit breaker in `SupervisorNetwork.dispatch()` that tracks the failure rate of each specialist over a 60-second rolling window and routes to a fallback agent when the failure rate exceeds 50%.

**Expert:** Add timing instrumentation to both `SupervisorNetwork` and `SwarmNetwork`. Run a benchmark with 10 concurrent workflows of varying subtask counts (2, 5, 10) and produce a latency comparison chart. Identify the crossover point where Swarm becomes faster than parallel-dispatch Supervisor.

**Research:** Read Wu et al. (2023) "AutoGen" (arXiv:2308.08155). Identify one limitation of the GroupChat pattern not addressed in this document, and propose a mitigation strategy based on the literature.

---

## 21. Interview Questions

1. **Conceptual:** Explain the difference between a Supervisor Network and a Swarm Network to a software engineer who has never worked with LLMs.

2. **Technical:** In a Supervisor Network, what happens to a workflow when the Supervisor agent's LLM call returns a malformed task decomposition? How should the system handle this?

3. **Design:** How would you architect a Supervisor Network to handle 10,000 concurrent legal document reviews per hour? Where are the bottlenecks, and how would you eliminate them?

4. **Trade-off:** When would you choose a Swarm topology over a Supervisor topology, even knowing that Swarms are harder to debug?

5. **Debugging:** A Swarm-based customer support system is producing duplicate responses for the same customer ticket. What are the three most likely root causes, and how would you diagnose each?

6. **Security:** An attacker submits a support ticket containing instructions that attempt to redirect the Swarm's routing logic to an agent with write access to the billing database. What defences would prevent this attack?

7. **Conceptual:** What is a routing cycle in a Swarm, and what are two mechanisms for preventing one?

8. **Technical:** Explain why the Supervisor's task state must be persisted to an external store rather than held in the Supervisor agent's process memory.

9. **Design:** You have a workflow where tasks A, B, and C are independent, but task D requires outputs from both A and B, and task E requires outputs from both C and D. Draw the task DAG and describe how a hybrid Supervisor-Swarm architecture would execute it.

10. **Trade-off:** A team argues that adding more specialist agents to their Swarm always improves throughput. Under what conditions is this false, and what measurement would you use to prove it?
