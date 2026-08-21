# W3D5 — Dynamic Skill Selection

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

An agent with 30 registered tools injects all 30 tool schemas into every LLM prompt by default — wasting thousands of tokens per call and degrading selection accuracy as the model struggles to distinguish between similar tools. Dynamic skill selection treats tool routing as a retrieval problem: embed skill descriptions once at registration, embed the user query at inference, and use cosine similarity to inject only the top-k relevant skills per turn. This PoC implements `SkillRegistry`, `EmbeddingRouter`, and `SkillInjector` — plus role-based permission filtering and a stale-skill eviction policy — reducing tool-definition tokens by 60–80% per prompt while measurably improving routing accuracy.

---

## Learning Objectives

1. Understand why injecting all tools into every prompt degrades LLM tool-selection accuracy and why cosine similarity over skill descriptions is the right routing mechanism
2. Implement `SkillRegistry` — register skills with descriptions, schemas, and required roles; assign embeddings at registration time
3. Implement `EmbeddingRouter.select()` — embed the query, score all registered skills by cosine similarity, apply a similarity threshold, and filter results by user roles
4. Implement `SkillInjector.build_tool_block()` — serialise selected `Skill` objects into OpenAI function-calling format ready for prompt injection
5. Understand why permission filtering happens after similarity scoring to avoid leaking information about which tools exist for which roles via timing side-channels
6. Implement `SkillRegistry.evict_stale()` — remove skills unused for `eviction_threshold` turns to bound registry growth in long-running agents
7. Know how to scale from brute-force cosine (suitable to ~500 tools) to FAISS ANN index for larger registries without changing the interface

---

## Problem Statement

A customer support agent has 30 tools: billing, network diagnostics, account management, security, IT ticketing, and more. Injecting all 30 schemas on every turn costs ~4,500 prompt tokens per call in tool definitions alone — at 200,000 calls/month, that is 900M tokens/month in tool overhead before any user content is counted. Worse, the model accuracy degrades: with 30 similar-looking tools in context, the model confuses `process_refund` with `get_invoice`, and `reset_password` with `provision_access`. The fix is to treat skill routing as retrieval: embed each tool description once, embed the query at runtime, and inject only the 3–5 most relevant tools. The network diagnostics query `"Why is my internet so slow?"` needs `run_ping_diagnostic` and `check_network_speed` — not the 28 billing and account tools that would just add noise.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs entirely offline with pre-computed 6-dimensional mock embeddings)
- Familiarity with [W1D5 — Episodic vs. Semantic Memory](../../week-01/W1D5-agent-memory/README.md) provides context on how agents manage what they know about their own capabilities
- Familiarity with [W2D5 — Reflection & Self-Correction Loops](../../week-02/W2D5_reflection-self-correction-loops/README.md) provides the Week 2 Agent Memory & Capabilities baseline

---

## Repository Structure

```
W3D5_dynamic-skill-selection/
├── README.md                                          # This file
├── docs/
│   ├── technical-document.md                          # 21-section practitioner deep-dive
│   └── dynamic-skill-selection-layman-scenarios.md    # Business scenarios without ML background
├── diagrams/
│   ├── architecture.mmd                               # SkillRegistry → EmbeddingRouter → Injector pipeline
│   └── sequence.mmd                                   # Per-query selection + permission filter flow
└── poc/
    ├── README.md                                      # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                                    # Entry point — 5 demo scenarios across 3 domains
    │   ├── skill_selection_core.py                    # SkillRegistry, EmbeddingRouter, SkillInjector
    │   └── config.py                                  # Config dataclass + env loader
    ├── tests/
    │   └── test_skill_selection.py                    # 16 unit tests across 4 test classes (all offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                              # 5 query scenarios with role context
    └── sample_output.json                             # Expected output for all 5 scenarios
```

---

## Core Concepts

### `SkillRegistry` — capability store

Registers skills with descriptions, JSON schemas, and role requirements. In demo mode, embeddings are assigned from a pre-computed mock table; in live mode, `embed_all()` calls the OpenAI Embeddings API once at startup:

```python
# skill_selection_core.py
registry = SkillRegistry(demo_mode=True)
registry.register(
    name="run_ping_diagnostic",
    description="Run ping and traceroute diagnostics to identify network latency issues",
    schema={"type": "object", "properties": {"ip_address": {"type": "string"}}, ...},
    required_roles=set(),          # empty = accessible to all roles
)
registry.register(
    name="process_refund",
    description="Process a billing refund for a customer",
    schema={...},
    required_roles={"billing", "admin"},   # restricted to billing and admin roles
)
```

### `EmbeddingRouter.select()` — cosine similarity routing

Embeds the query, scores all registered skills by cosine similarity, applies the threshold, then filters by user role. Permission filtering intentionally happens after scoring to avoid timing side-channels that could reveal which roles have access to which tools:

```python
# skill_selection_core.py
router = EmbeddingRouter(registry, top_k=5, similarity_threshold=0.35)
result = router.select(
    query="Why is my internet connection so slow today?",
    user_roles={"user"},
)
# result.selected_skills → [run_ping_diagnostic (0.992), check_network_speed (0.988), create_it_ticket (0.541)]
# provision_access (score 0.20) filtered: below threshold
# process_refund  (score 0.05) filtered: below threshold + requires billing role
```

### `SkillInjector.build_tool_block()` — prompt serialisation

Converts selected `Skill` objects into the OpenAI function-calling format for direct injection into the LLM call:

```python
# skill_selection_core.py
injector = SkillInjector()
tool_block = injector.build_tool_block(result.selected_skills)
# tool_block → list of {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
# Pass directly to: client.chat.completions.create(..., tools=tool_block)
```

### Fallback and eviction

When no skill clears the similarity threshold, the router activates the `general_response` fallback rather than injecting nothing. Stale skills (unused for `eviction_after_turns` turns) are pruned from the registry to prevent unbounded growth:

```python
# skill_selection_core.py
# Fallback: "What is the meaning of life?" → no tool clears 0.35 → general_response activated
result = router.select("What is the meaning of life?", user_roles={"user"})
# result.used_fallback == True; result.selected_skills == [general_response]

# Eviction: remove skills unused for 50+ turns
evicted = registry.evict_stale(eviction_threshold=50)
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-03/W3D5_dynamic-skill-selection/poc
pip install -r requirements.txt
cp .env.example .env
DEMO_MODE=true python src/main.py
```

### Live mode (real embeddings via OpenAI)

```bash
# Edit .env and set OPENAI_API_KEY=your-key
python src/main.py
```

### Tests

```bash
pytest tests/ -v
# 16 tests across 4 classes — all pass offline
```

---

## Expected Output

```
W3D5 Dynamic Skill Selection Demo
==================================================

--- Running in DEMO MODE (no API key — embeddings are pre-computed) ---

Query:    Why is my internet connection so slow today?
Roles:    ['user']
Selected: ['run_ping_diagnostic', 'check_network_speed', 'create_it_ticket'] (3 of 8 tools)
Scores:   {'run_ping_diagnostic': 0.992, 'check_network_speed': 0.988, 'create_it_ticket': 0.541}

Query:    I need a refund on my last invoice, I was charged twice
Roles:    ['billing']
Selected: ['process_refund', 'get_invoice'] (2 of 8 tools)
Scores:   {'process_refund': 0.996, 'get_invoice': 0.991, 'create_it_ticket': 0.312}

Query:    Grant Sarah access to the shared finance drive
Roles:    ['user']
Selected: ['reset_password', 'create_it_ticket'] (2 of 8 tools)
          [provision_access filtered out — requires admin role]

Query:    What is the meaning of life?
Roles:    ['user']
Selected: ['general_response'] (1 of 8 tools)
          [fallback activated — low similarity across all skills]

==================================================
Concept demonstrated: Routing injects only relevant skills per query,
reducing tool definitions from 8 to ~3 per turn.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — registers 8 skills, runs 5 query scenarios, prints selection results and scores |
| `src/skill_selection_core.py` | `Skill`, `SelectionResult`, `SkillRegistry`, `EmbeddingRouter`, `SkillInjector`, `_cosine_similarity()` |
| `src/config.py` | `Config` dataclass + `load_config()` with top_k, similarity_threshold, eviction_after_turns |
| `tests/test_skill_selection.py` | 16 unit tests: cosine similarity, role filtering, fallback activation, eviction, live API mock, domain routing (parametrised) |
| `sample_input.json` | 5 query scenarios with varying roles: network, billing, password, admin-restricted, unknown |
| `sample_output.json` | Pre-computed results for all 5 scenarios: skill names, scores, fallback flags |
| `.env.example` | All environment variable defaults |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for skill and query embeddings (live mode only) |
| `TOP_K` | `5` | Maximum number of skills injected per turn |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum cosine similarity score to qualify for injection |
| `EVICTION_AFTER_TURNS` | `50` | Turns of inactivity before a skill is evicted from the registry |
| `DEMO_MODE` | `false` | Set `true` to run with pre-computed mock embeddings |
| `MODEL` | `gpt-4o-mini` | LLM model for live mode |
| `TEMPERATURE` | `0.0` | LLM temperature for deterministic tool selection |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/dynamic-skill-selection-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — SkillRegistry → EmbeddingRouter → SkillInjector pipeline
- [Sequence Diagram](diagrams/sequence.mmd) — Per-query selection and permission filter flow

---

## Connection to the Series

**Previous:** [W3D4 — Async & Parallel Tool Calls](../W3D4_async-parallel-tool-calls/README.md) — executes multiple known tools in parallel; W3D5 solves the upstream problem of which tools to execute in the first place.

**Next:** [W3D6 — Hierarchical Subagent Teams](../README.md) — once agents can select skills dynamically, the next step is organising multiple agents into hierarchical teams where a planner decomposes tasks and dispatches to specialists.

**Series arc:** [W1D5 — Episodic vs. Semantic Memory](../../week-01/W1D5-agent-memory/README.md) introduced how agents store and retrieve knowledge about their world. [W2D5 — Reflection & Self-Correction Loops](../../week-02/W2D5_reflection-self-correction-loops/README.md) added self-improvement over multiple turns. W3D5 closes the Agent Memory & Capabilities vertical for Week 3: the agent's knowledge of its own capabilities is now dynamically retrieved — not statically wired — enabling adaptation as the tool set grows.

---

## Key References

- Qin, Y. et al. (2023). "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." arXiv:2307.16789
- [Semantic Router](https://github.com/aurelio-ai/semantic-router) — production embedding-based routing library
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

## Continue Learning

**Next:** [W3D6 — Hierarchical Subagent Teams](../README.md)

Return to [Week 3 overview](../README.md) to explore all advanced techniques.
