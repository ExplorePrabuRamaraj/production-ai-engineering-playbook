# W3D5 — Dynamic Skill Selection

**Series:** AI Engineering Production Playbook
**Vertical:** Agent Memory & Capabilities
**Week 3 / Day 5 of 28**

## What This Demonstrates

An agent with 30 registered tools injects all 30 into every LLM prompt by default — wasting thousands of tokens and degrading tool-selection accuracy. This PoC shows how to treat tool selection as a retrieval problem: embed skill descriptions once at registration, embed the user query at inference, and use cosine similarity to inject only the top-k relevant skills per turn.

Result: 60–80% fewer tool-definition tokens per prompt, measurably higher routing accuracy, and role-based access control enforced at the skill-visibility layer.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-03/W3D5_dynamic-skill-selection/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env to add your OpenAI API key, or leave blank for demo mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode uses pre-computed 6-dimensional mock embeddings that demonstrate the same routing behaviour as live embeddings, with zero API calls. The output format is identical to live mode.

## Run Tests

```bash
pytest tests/ -v
```

All 16 tests pass offline. No API key needed. Tests cover:
- Demo mode output schema validation
- Cosine similarity correctness
- Role-based permission filtering
- Fallback activation on low-similarity queries
- Eviction policy for stale skills
- Domain-routing accuracy across 4 query types (parametrised)
- Mocked live embedding API call

## Expected Output (Demo Mode)

```
W3D5 Dynamic Skill Selection Demo
==================================================

Query:    Why is my internet connection so slow today?
Roles:    ['user']
Selected: ['run_ping_diagnostic', 'check_network_speed', 'create_it_ticket'] (3 of 8 tools)
Scores:   {'run_ping_diagnostic': 0.992, 'check_network_speed': 0.988, 'create_it_ticket': 0.541}

Query:    I need a refund on my last invoice, I was charged twice
Roles:    ['billing']
Selected: ['process_refund', 'get_invoice'] (2 of 8 tools)
Scores:   {'process_refund': 0.996, 'get_invoice': 0.991, 'create_it_ticket': 0.312}

Query:    What is the meaning of life?
Roles:    ['user']
Selected: ['general_response'] (1 of 8 tools)
          [fallback activated — low similarity across all skills]

==================================================
Concept demonstrated: Routing injects only relevant skills per query,
reducing tool definitions from 8 to ~3 per turn.
```

## Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — end-to-end demo with 5 scenarios |
| `src/skill_selection_core.py` | `SkillRegistry`, `EmbeddingRouter`, `SkillInjector` |
| `src/config.py` | All configuration via environment variables |
| `tests/test_skill_selection.py` | 16 unit tests across 4 test classes |
| `sample_input.json` | 5 realistic query scenarios with role context |
| `sample_output.json` | Expected output for each scenario |

## Extending This PoC

**Add a new skill:**

```python
registry.register(
    name="search_knowledge_base",
    description="Search the internal knowledge base for product documentation and FAQs",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    required_roles=set(),   # Available to all roles
)
```

The embedding is computed once at registration (live mode) or assigned from the mock table (demo mode). The skill becomes selectable immediately.

**Scale to 500+ tools:**

Uncomment `faiss-cpu` in `requirements.txt` and replace the brute-force cosine loop in `EmbeddingRouter.select()` with a FAISS index query. The rest of the interface is unchanged.

## Architecture

```
User Message + Auth Context
        │
        ▼
  Intent Router (embed query → cosine similarity)
        │
        ▼
  Permission Filter (role-based gating)
        │
        ▼
  Skill Injector (top-k tool schemas → LLM prompt)
        │
        ▼
       LLM
        │
        ▼
  Tool Executor → Usage Tracker → Eviction Policy
```

See `../diagrams/architecture.mmd` for the full Mermaid diagram.

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Layman Scenarios](../docs/dynamic-skill-selection-layman-scenarios.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [LinkedIn Post](../README.md)

## References

- ToolLLM paper: arXiv:2307.16789
- Semantic Router: https://github.com/aurelio-ai/semantic-router
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
