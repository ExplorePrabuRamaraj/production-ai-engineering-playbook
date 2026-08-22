# W3D6 — Hierarchical Subagent Teams

**Series:** AI Engineering Production Playbook
**Vertical:** Multi-Agent Orchestration
**Week 3 / Day 6**

## What This Demonstrates

A 3-tier agent hierarchy — Orchestrator, Team Leads, and Worker Agents — with typed result contracts (Pydantic) enforced at every tier boundary. This pattern eliminates the coordination failures (duplicate work, context bleed, unscoped error propagation) that occur in flat agent pools handling complex tasks.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-03/W3D6_hierarchical-subagent-teams/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run in demo mode (no API key needed)
DEMO_MODE=true python src/main.py

# 4. Run with a real API key
export OPENAI_API_KEY=your-key-here
python src/main.py
```

## Demo Mode (No API Key)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
Hierarchical Subagent Teams Demo
==================================================
Goal: Analyse competitive landscape for a SaaS CRM product...

Running in DEMO MODE — output is pre-computed (no API call made)

--- Tier 2: Team Lead Results ---
  Lead 'lead_research' [Competitive Research]: SUCCESS | 212 tokens
    Worker 'lead_research_worker_0': ok | 87 tokens | 412ms
    Worker 'lead_research_worker_1': ok | 74 tokens | 388ms
  Lead 'lead_analysis' [Market Analysis]: SUCCESS | 118 tokens
    Worker 'lead_analysis_worker_0': ok | 63 tokens | 401ms

--- Tier 1: Orchestrator Final Output ---
Competitive analysis complete. ...

--- Stats ---
  Total tokens: 330
  Total latency: 1240ms
  Overall success: True

Concept demonstrated: 3-tier hierarchy with typed contracts prevents context bleed
```

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — no API key required.

## Architecture

```
Orchestrator (Tier 1)
  Decomposes goal → TaskPlan
  Checks LeadResult flags before synthesis
  Assembles final output

  Team Lead A (Tier 2) — Competitive Research
    Dispatches workers, retries failures, validates contract
    Worker A1 — List competitors (Tier 3)
    Worker A2 — Identify advantages (Tier 3)

  Team Lead B (Tier 2) — Market Analysis
    Dispatches workers, retries failures, validates contract
    Worker B1 — TAM/SAM/growth (Tier 3)
```

## Key Design Decisions

**Typed contracts at tier boundaries** — `WorkerResult` and `LeadResult` are dataclasses with required fields. No raw LLM string passes between tiers. This is the mechanism that prevents context bleed.

**Scoped retry** — When a worker fails, the team lead retries only that worker. The orchestrator and sibling leads are unaffected. This is why `max_retries` is set at the lead level, not the orchestrator level.

**Orchestrator checks flags before synthesis** — The orchestrator reads `success` and `partial` flags on every `LeadResult` before calling the synthesis LLM. A partial result triggers a warning in the final output rather than silent omission.

## File Structure

```
poc/
  src/
    main.py               — Entry point: runs the full 3-tier hierarchy
    hierarchical_core.py  — Tier implementations + typed result contracts
    config.py             — Environment variable config loader
  tests/
    test_hierarchical.py  — Unit tests (all offline, all mocked)
  requirements.txt
  sample_input.json       — Example goal for main.py
  sample_output.json      — Expected output structure
```

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [LinkedIn Post](../README.md)
- [Layman Scenarios](../docs/hierarchical-subagent-teams-layman-scenarios.md)
