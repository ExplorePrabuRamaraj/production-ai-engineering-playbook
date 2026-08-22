# W3D6 — Hierarchical Subagent Teams

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

A flat pool of peer-to-peer agents (W2D6) handles homogeneous parallel tasks but degrades on work that requires decomposition, dependency tracking, and result aggregation — without a planner layer, complex multi-step goals either run sequentially or produce uncoordinated outputs. This PoC implements a 3-tier agent hierarchy: an Orchestrator (Tier 1) that decomposes a goal into bounded subtasks, Team Leads (Tier 2) that own a domain, dispatch workers, handle scoped retries, and validate typed results before returning them, and Worker Agents (Tier 3) that are stateless leaf executors receiving only the context needed for one atomic task. Typed result contracts (`WorkerResult`, `LeadResult`, `FinalResult`) are enforced at every tier boundary — no raw LLM string crosses a tier, eliminating context bleed, duplicate work, and unscoped error propagation.

---

## Learning Objectives

1. Understand why flat agent pools fail on complex tasks requiring decomposition and why typed tier boundaries prevent context bleed
2. Implement `run_worker()` — stateless Tier 3 leaf executor that receives bounded context, returns one `WorkerResult`, and writes nothing to shared state
3. Implement `run_team_lead()` — Tier 2 domain owner that dispatches workers, retries individual failures with scoped retry (not the full lead), aggregates typed `WorkerResult` objects, and returns a single validated `LeadResult`
4. Implement `run_orchestrator()` — Tier 1 planner that receives a `TaskPlan`, dispatches leads, checks `success` and `partial` flags on every `LeadResult` before synthesis, and assembles the final output
5. Understand why the orchestrator never does leaf work — planning, routing, and assembly only — so it stays context-free between dispatches
6. Understand why aggregation happens at the lead layer, not the orchestrator — the orchestrator should never see raw worker outputs, which may be inconsistently formatted
7. Use `SubtaskSpec.depends_on` and `ExecutionOrder` to express task dependencies and control whether leads run in parallel or sequentially

---

## Problem Statement

A competitive intelligence pipeline runs 6 sub-tasks: competitor identification, pricing research, feature gap analysis, market sizing, growth trend analysis, and risk profiling. In a flat agent pool, each agent can call all 6 tools and accumulate each other's context — within 3 turns the 8,000-token context window is 70% full of other agents' outputs, the model confuses which subtask it is still completing, and a single failed agent blocks the synthesiser. In a hierarchy, the Orchestrator decomposes the goal into two bounded subtasks (Competitive Research, Market Analysis), each Team Lead dispatches 2–3 atomic workers, retries only the failed worker on error (not the full lead), and returns a single validated domain summary. The Orchestrator sees two clean `LeadResult` objects — not 6 raw worker dumps — before it synthesises the final output. Total tokens: 330 for a full competitive analysis.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs entirely offline with pre-computed results)
- Familiarity with [W1D6 — State Graphs (LangGraph)](../../week-01/W1D6-langgraph-state-graphs/README.md) and [W2D6 — Supervisor vs. Swarm Networks](../../week-02/W2D6_supervisor-vs-swarm-networks/README.md) provides the multi-agent orchestration arc that W3D6 advances

---

## Repository Structure

```
W3D6_hierarchical-subagent-teams/
├── README.md                                              # This file
├── docs/
│   ├── technical-document.md                             # 21-section practitioner deep-dive
│   └── hierarchical-subagent-teams-layman-scenarios.md   # Business scenarios without ML background
├── diagrams/
│   ├── architecture.mmd                                  # 3-tier hierarchy with typed contract boundaries
│   └── sequence.mmd                                      # Orchestrator → Lead → Worker dispatch flow
└── poc/
    ├── README.md                                         # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                                       # Entry point — demo + live mode
    │   ├── hierarchical_core.py                          # All tier logic + typed result contracts
    │   └── config.py                                     # Config dataclass + env loader
    ├── tests/
    │   └── test_hierarchical.py                          # Unit tests (all offline, all mocked)
    ├── requirements.txt
    ├── sample_input.json                                 # Example CRM competitive analysis goal
    └── sample_output.json                                # Pre-computed FinalResult: 330 tokens, 1240ms
```

---

## Core Concepts

### Typed result contracts

Four dataclasses enforce strict boundaries between tiers. No raw LLM string crosses a tier — the lead validates worker output before the orchestrator ever sees it:

```python
# hierarchical_core.py
@dataclass
class WorkerResult:       # Tier 3 → Tier 2
    worker_id: str
    output: str
    tokens_used: int
    latency_ms: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class LeadResult:         # Tier 2 → Tier 1
    lead_id: str
    domain: str
    aggregated_output: str  # normalised by the lead — orchestrator never sees raw worker output
    worker_results: List[WorkerResult]
    tokens_used: int
    success: bool
    partial: bool = False   # True if some workers failed after retry budget
```

### `run_worker()` — stateless Tier 3 executor

Workers receive only their one instruction plus the lead's subtask as context. They write nothing to shared memory. This bounds the blast radius when a worker fails — the lead retries just that worker without rerunning siblings:

```python
# hierarchical_core.py
def run_worker(worker_id, instruction, context, api_key, model, max_tokens) -> WorkerResult:
    # receives: one instruction + parent lead's subtask (NOT the full goal)
    # returns:  WorkerResult — typed, never raw string
    # writes:   nothing — stateless
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": "Complete exactly the task given. Be concise."},
                  {"role": "user", "content": f"Context: {context}\n\nTask: {instruction}"}],
        temperature=0.0, max_tokens=max_tokens,
    )
    return WorkerResult(worker_id=worker_id, output=response.choices[0].message.content, ...)
```

### `run_team_lead()` — scoped retry and aggregation

The lead owns a domain subtask. It dispatches workers one at a time, retrying only the failed worker up to `max_retries` (not the full lead or its siblings). Once all workers complete, the lead makes one aggregation LLM call to normalise their outputs before returning a single `LeadResult`:

```python
# hierarchical_core.py
def run_team_lead(lead_id, domain, subtask, worker_instructions, ..., max_retries=2) -> LeadResult:
    for w_def in worker_instructions:
        attempts = 0
        while attempts <= max_retries:
            result = run_worker(worker_id=w_def["worker_id"], instruction=w_def["instruction"], ...)
            if result.success:
                break              # scoped retry — only this worker, not siblings
            attempts += 1
        worker_results.append(result)

    # One aggregation call per lead — orchestrator sees one clean summary, not N worker dumps
    aggregated = client.chat.completions.create(messages=[..., combined_worker_outputs], ...)
    return LeadResult(lead_id=lead_id, aggregated_output=aggregated, partial=len(failed) > 0, ...)
```

### `run_orchestrator()` — contract-checked synthesis

The orchestrator dispatches leads, reads `success` and `partial` flags on every `LeadResult` before synthesis, and adds explicit warnings for partial results rather than silently omitting them:

```python
# hierarchical_core.py
def run_orchestrator(goal, subtask_specs, worker_map, ...) -> FinalResult:
    for spec in subtask_specs:
        lead_result = run_team_lead(lead_id=spec.lead_id, domain=spec.domain, ...)
        lead_results.append(lead_result)

        if lead_result.partial:   # check contract flags — never proceed silently
            warnings.append(f"Lead '{spec.lead_id}' returned partial result.")
        if not lead_result.success:
            warnings.append(f"Lead '{spec.lead_id}' failed — domain missing from output.")

    # Synthesise from validated LeadResults only — never from raw worker output
    combined_leads = "\n\n".join(f"[{lr.domain}]: {lr.aggregated_output}" for lr in successful_leads)
    final_output = client.chat.completions.create(messages=[..., combined_leads], ...)
    return FinalResult(goal=goal, final_output=final_output, warnings=warnings, ...)
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-03/W3D6_hierarchical-subagent-teams/poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

### Live mode

```bash
export OPENAI_API_KEY=your-key-here
python src/main.py
```

### Tests

```bash
pytest tests/ -v
# All tests pass offline — no API key required, all LLM calls mocked
```

---

## Expected Output

```
[DEMO] Hierarchical Subagent Teams
==================================================
Goal: Analyse competitive landscape for a SaaS CRM product...

[DEMO MODE] Output is pre-computed (no API call made)

--- Tier 2: Team Lead Results ---
  Lead 'lead_research' [Competitive Research]: SUCCESS | 212 tokens
    Worker 'lead_research_worker_0': ok | 87 tokens | 412ms
    Worker 'lead_research_worker_1': ok | 74 tokens | 388ms
  Lead 'lead_analysis' [Market Analysis]: SUCCESS | 118 tokens
    Worker 'lead_analysis_worker_0': ok | 63 tokens | 401ms

--- Tier 1: Orchestrator Final Output ---
Competitive analysis complete.

The mid-market CRM space is growing at 14% YoY with an AI-native subsegment at 31%.
Primary competitors (Salesforce, HubSpot, Pipedrive) all raised prices in Q1 2024,
creating a pricing advantage window. Key differentiators to position: 40% lower TCO,
3-day onboarding (vs 14-day average), and AI-native architecture.
Critical gap to address before Q3: CPQ module — absence is cited in 38% of lost deals vs Salesforce.

--- Stats ---
  Total tokens: 330
  Total latency: 1240ms
  Overall success: True

[OK] Concept demonstrated: 3-tier hierarchy with typed contracts prevents context bleed
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — builds TaskPlan, dispatches hierarchy or runs demo, prints tier-by-tier trace |
| `src/hierarchical_core.py` | `SubtaskSpec`, `WorkerResult`, `LeadResult`, `FinalResult`, `ExecutionOrder`, `run_worker()`, `run_team_lead()`, `run_orchestrator()`, `run_demo()` |
| `src/config.py` | `Config` dataclass + `load_config()` with worker timeout, retry count, parallel lead limit |
| `tests/test_hierarchical.py` | Unit tests: worker success/failure/retry, lead aggregation, orchestrator contract checking, partial result warnings (all offline, all mocked) |
| `sample_input.json` | CRM competitive analysis goal with domain list |
| `sample_output.json` | Pre-computed `FinalResult`: 2 leads, 330 total tokens, 1240ms, no warnings |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `MODEL` | `gpt-4o-mini` | LLM model for all tiers |
| `WORKER_TIMEOUT_SECONDS` | `30` | Per-worker call deadline in seconds |
| `WORKER_MAX_RETRIES` | `2` | Max retry attempts per failed worker (scoped to that worker only) |
| `MAX_PARALLEL_LEADS` | `5` | Maximum simultaneous lead dispatches |
| `DEMO_MODE` | `false` | Set `true` to run with pre-computed output |
| `TEMPERATURE` | `0.0` | LLM temperature — keep at 0 for deterministic synthesis |
| `MAX_TOKENS` | `500` | Max tokens per LLM call at any tier |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/hierarchical-subagent-teams-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — 3-tier hierarchy with typed contract boundaries
- [Sequence Diagram](diagrams/sequence.mmd) — Orchestrator → Lead → Worker dispatch and aggregation flow

---

## Connection to the Series

**Previous:** [W3D5 — Dynamic Skill Selection](../W3D5_dynamic-skill-selection/README.md) — selects which tools to give to an agent; W3D6 organises multiple agents into a hierarchy so complex tasks can be decomposed and parallelised safely.

**Next:** [W3D7 — Distributed Tracing (LangSmith)](../W3D7_distributed-tracing-langsmith/README.md) — once a hierarchy is running in production, the next problem is observability: distributed tracing surfaces which tier introduced latency, which worker produced the token-heavy output, and where accuracy regressions originate.

**Series arc:** [W1D6 — State Graphs (LangGraph)](../../week-01/W1D6-langgraph-state-graphs/README.md) introduced state-based orchestration with explicit transitions. [W2D6 — Supervisor vs. Swarm Networks](../../week-02/W2D6_supervisor-vs-swarm-networks/README.md) contrasted centrally-supervised vs. peer-to-peer coordination. W3D6 advances the arc: a full 3-tier hierarchy with typed contracts, scoped retry, and contract-checked synthesis — the production pattern for complex multi-step goals that neither a flat swarm nor a single supervisor can handle reliably.

---

## Key References

- Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." arXiv:2304.03442 — foundational paper on multi-agent memory and coordination
- [LangGraph multi-agent architectures](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [OpenAI Assistants API — agent handoffs](https://platform.openai.com/docs/assistants/overview)

---

## Continue Learning

**Next:** [W3D7 — Distributed Tracing (LangSmith)](../W3D7_distributed-tracing-langsmith/README.md)

Return to [Week 3 overview](../README.md) to explore all advanced techniques.
