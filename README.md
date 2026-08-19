# Production AI Engineering Playbook

> A 28-day hands-on curriculum for software engineers building production AI systems — one topic per day, with technical deep-dives and runnable PoC code.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## About

This repository is the companion resource for the **AI Engineering Production Playbook** — a 28-day LinkedIn learning series covering the full stack of production AI engineering: from prompt design and context management through advanced RAG, multi-agent orchestration, and production guardrails.

Each day delivers three coordinated artifacts:

| Artifact | What it contains |
|---|---|
| **LinkedIn post** | 1,200–1,800 character deep-dive published daily |
| **Technical document** | 21-section practitioner guide with architecture diagrams |
| **PoC code** | Runnable Python demo — works offline with `DEMO_MODE=true` |

The series follows a strict **WHY → WHAT → HOW → WHEN → IMPLEMENTATION → BEST PRACTICES → ANTI-PATTERNS → PRODUCTION → NEXT** educational flow on every topic.

---

## Why This Repository?

Hand-crafting prompts, eyeballing retrieval quality, and wiring agents together manually works in a notebook. It breaks in production. This series documents — with runnable code — the patterns, tools, and trade-offs that matter when AI features must be reliable, observable, and maintainable at scale.

Every topic is chosen because it has a real production failure mode attached to it. The PoC code for each day demonstrates that failure and its fix.

---

## Who Should Follow This Series?

- **Software engineers** transitioning into AI/ML engineering roles
- **ML practitioners** moving from research prototypes to production systems
- **Senior engineers** who need to evaluate AI tooling decisions (DSPy, MCP, LangGraph, and more)
- Anyone who has ever had a prompt "just stop working" after a model update

No ML research background required. Python proficiency and basic LLM familiarity assumed.

---

## Learning Roadmap

### Week 1 — Foundations

| Day | Topic | Vertical | Status |
|---|---|---|---|
| [W1D1](week-01/W1D1-dspy-programmatic-prompts/) | DSPy & Programmatic Prompts | Prompt Engineering & Schemas | ✅ Complete |
| [W1D2](week-01/W1D2-lost-in-the-middle/) | "Lost in the Middle" Decay | Context Engineering & Tokens | ✅ Complete |
| [W1D3](week-01/W1D3-naive-vs-agentic-rag/) | Naive vs. Agentic RAG | Advanced RAG | ✅ Complete |
| [W1D4](week-01/W1D4-model-context-protocol/) | Model Context Protocol (MCP) Intro | MCP & Tool Integration | ✅ Complete |
| [W1D5](week-01/W1D5-agent-memory/) | Episodic vs. Semantic Memory | Agent Memory & Capabilities | ✅ Complete |
| [W1D6](week-01/W1D6-langgraph-state-graphs/) | State Graphs (LangGraph) | Multi-Agent Orchestration | ✅ Complete |
| [W1D7](week-01/W1D7-llm-as-a-judge/) | LLM-as-a-Judge Evals | Production Evals & Guardrails | ✅ Complete |

### Week 2 — Intermediate Patterns

| Day | Topic | Vertical | Status |
|---|---|---|---|
| [W2D1](week-02/W2D1_type-safe-schemas-pydantic-ai/) | Type-Safe Schemas (Pydantic AI) | Prompt Engineering & Schemas | ✅ Complete |
| [W2D2](week-02/W2D2_kv-caching-token-trimming/) | KV Caching & Token Trimming | Context Engineering & Tokens | ✅ Complete |
| [W2D3](week-02/W2D3_graphrag-knowledge-graphs/) | GraphRAG & Knowledge Graphs | Advanced RAG | ✅ Complete |
| [W2D4](week-02/W2D4_custom-mcp-server-build/) | Custom MCP Server Build | MCP & Tool Integration | ✅ Complete |
| [W2D5](week-02/W2D5_reflection-self-correction-loops/) | Reflection & Self-Correction Loops | Agent Memory & Capabilities | ✅ Complete |
| [W2D6](week-02/W2D6_supervisor-vs-swarm-networks/) | Supervisor vs. Swarm Networks | Multi-Agent Orchestration | ✅ Complete |
| [W2D7](week-02/W2D7_deterministic-guardrails-nemo/) | Deterministic Guardrails (NeMo) | Production Evals & Guardrails | ✅ Complete |

### Week 3 — Advanced Techniques

| Day | Topic | Vertical | Status |
|---|---|---|---|
| [W3D1](week-03/W3D1_prompt-distillation/) | Prompt Distillation | Prompt Engineering & Schemas | ✅ Complete |
| [W3D2](week-03/W3D2_context-compression/) | Context Compression | Context Engineering & Tokens | ✅ Complete |
| [W3D3](week-03/W3D3_hybrid-search-reranking/) | Hybrid Search & Reranking | Advanced RAG | ✅ Complete |
| W3D4 | Async & Parallel Tool Calls | MCP & Tool Integration | 🔜 Coming |
| W3D5 | Dynamic Skill Selection | Agent Memory & Capabilities | 🔜 Coming |
| W3D6 | Hierarchical Subagent Teams | Multi-Agent Orchestration | 🔜 Coming |
| W3D7 | Distributed Tracing (LangSmith) | Production Evals & Guardrails | 🔜 Coming |

### Week 4 — Production & Scale

| Day | Topic | Vertical | Status |
|---|---|---|---|
| W4D1 | Multi-Modal Prompting | Prompt Engineering & Schemas | 🔜 Coming |
| W4D2 | Needle-in-a-Haystack Benchmarking | Context Engineering & Tokens | 🔜 Coming |
| W4D3 | RAG Triad & Chunking Strategies | Advanced RAG | 🔜 Coming |
| W4D4 | Agent-to-Agent (A2A) Protocols | MCP & Tool Integration | 🔜 Coming |
| W4D5 | Long-Running Memory Decay | Agent Memory & Capabilities | 🔜 Coming |
| W4D6 | Consensus-Based Agent Voting | Multi-Agent Orchestration | 🔜 Coming |
| W4D7 | Circuit Breakers & Safety Loops | Production Evals & Guardrails | 🔜 Coming |

---

## Weekly Progress

| Week | Theme | Progress |
|---|---|---|
| [Week 1](week-01/) | Foundations | 7 / 7 days ✅ |
| [Week 2](week-02/) | Intermediate Patterns | 7 / 7 days ✅ |
| [Week 3](week-03/) | Advanced Techniques | 3 / 7 days |
| Week 4 | Production & Scale | 0 / 7 days |

---

## Repository Structure

Each completed day follows a consistent layout. The per-day template is shown once below (W1D1), followed by a listing of all days with their unique core module names.

```
production-ai-engineering-playbook/
├── README.md
├── CONTRIBUTING.md
├── ROADMAP.md
│
├── week-01/                                        # Week 1 — Foundations (7 / 7 ✅)
│   ├── README.md
│   │
│   ├── W1D1-dspy-programmatic-prompts/             # ── template for all Week 1 days ──
│   │   ├── README.md                               # Day overview, learning objectives, run guide
│   │   ├── docs/
│   │   │   ├── technical-document.md               # 21-section practitioner deep-dive
│   │   │   └── dspy-layman-scenarios.md            # Business scenarios (no ML background needed)
│   │   ├── diagrams/                               # Mermaid source files
│   │   │   ├── architecture.mmd
│   │   │   └── sequence.mmd
│   │   └── poc/
│   │       ├── README.md                           # PoC quick-start and expected output
│   │       ├── src/
│   │       │   ├── main.py                         # Entry point — demo + live mode
│   │       │   ├── dspy_core.py                    # Core logic (pure, independently testable)
│   │       │   └── config.py                       # Config loaded from environment variables
│   │       ├── tests/
│   │       │   └── test_dspy.py                    # pytest unit tests (all run offline)
│   │       ├── requirements.txt
│   │       ├── .env.example
│   │       ├── sample_input.json
│   │       └── sample_output.json
│   │
│   ├── W1D2-lost-in-the-middle/                    # core: lost_in_middle_core.py | test: test_lost_in_middle.py
│   ├── W1D3-naive-vs-agentic-rag/                  # core: rag_core.py            | test: test_rag_core.py
│   ├── W1D4-model-context-protocol/                # core: mcp_core.py            | test: test_mcp_core.py
│   ├── W1D5-agent-memory/                          # core: memory_core.py         | test: test_memory_core.py
│   ├── W1D6-langgraph-state-graphs/                # core: state_graph_core.py    | test: test_state_graph.py
│   └── W1D7-llm-as-a-judge/                        # core: judge_core.py          | test: test_judge.py
│
├── week-02/                                        # Week 2 — Intermediate Patterns (7 / 7 ✅)
│   ├── README.md
│   ├── W2D1_type-safe-schemas-pydantic-ai/         # ⚠ underscore separator; diagram/ (singular)
│   │   ├── README.md
│   │   ├── docs/
│   │   │   ├── technical-document.md
│   │   │   └── type-safe-schemas-pydantic-ai-layman-scenarios.md
│   │   ├── diagram/                                 # singular — differs from Week 1 convention
│   │   │   ├── architecture.mmd
│   │   │   └── sequence.mmd
│   │   └── poc/
│   │       ├── README.md
│   │       ├── src/
│   │       │   ├── main.py
│   │       │   ├── pydantic_schemas_core.py
│   │       │   └── config.py
│   │       ├── tests/
│   │       │   └── test_pydantic_schemas.py
│   │       ├── requirements.txt
│   │       ├── .env.example
│   │       ├── sample_input.json
│   │       └── sample_output.json
│   ├── W2D2_kv-caching-token-trimming/             # core: kv_caching_core.py     | test: test_kv_caching.py
│   ├── W2D3_graphrag-knowledge-graphs/             # core: graphrag_core.py       | test: test_graphrag.py
│   ├── W2D4_custom-mcp-server-build/               # core: mcp_server_core.py     | test: test_mcp_server.py
│   ├── W2D5_reflection-self-correction-loops/      # core: reflection_core.py     | test: test_reflection.py
│   ├── W2D6_supervisor-vs-swarm-networks/          # core: swarm_core.py          | test: test_swarm.py
│   └── W2D7_deterministic-guardrails-nemo/         # core: guardrails_core.py     | test: test_guardrails.py
│
└── week-03/                                        # Week 3 — Advanced Techniques (3 / 7)
    ├── README.md
    ├── W3D1_prompt-distillation/                   # core: distillation_core.py          | test: test_distillation.py
    ├── W3D2_context-compression/                   # core: context_compression_core.py   | test: test_context_compression.py
    └── W3D3_hybrid-search-reranking/               # core: hybrid_search_core.py         | test: test_hybrid_search.py
```

---

## How to Use This Repository

**Follow the series day-by-day:**

1. Read the LinkedIn post (linked from each day's README) for the big picture
2. Read the technical document in `docs/` for depth
3. Run the PoC code in `poc/` — all demos work without an API key
4. Study the architecture diagrams in `diagrams/` for system-level understanding

**Run any PoC in demo mode:**

```bash
cd week-01/W1D1-dspy-programmatic-prompts/poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

**Run the tests:**

```bash
cd week-01/W1D1-dspy-programmatic-prompts/poc
pytest tests/ -v
```

---

## Technologies Covered

| Vertical | Tools & Libraries |
|---|---|
| Prompt Engineering & Schemas | DSPy, Pydantic AI |
| Context Engineering & Tokens | KV caching, token trimming strategies |
| Advanced RAG | LlamaIndex, LangChain, GraphRAG |
| MCP & Tool Integration | Model Context Protocol, async tool calls |
| Agent Memory & Capabilities | LangGraph, episodic/semantic memory patterns |
| Multi-Agent Orchestration | LangGraph state graphs, supervisor/swarm patterns |
| Production Evals & Guardrails | LLM-as-a-Judge, NeMo Guardrails, LangSmith |

---

## License

MIT — see [LICENSE](LICENSE) for details.
