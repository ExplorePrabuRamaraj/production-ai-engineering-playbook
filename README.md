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
| W1D6 | State Graphs (LangGraph) | Multi-Agent Orchestration | 🔜 Coming |
| W1D7 | LLM-as-a-Judge Evals | Production Evals & Guardrails | 🔜 Coming |

### Week 2 — Intermediate Patterns

| Day | Topic | Vertical | Status |
|---|---|---|---|
| W2D1 | Type-Safe Schemas (Pydantic AI) | Prompt Engineering & Schemas | 🔜 Coming |
| W2D2 | KV Caching & Token Trimming | Context Engineering & Tokens | 🔜 Coming |
| W2D3 | GraphRAG & Knowledge Graphs | Advanced RAG | 🔜 Coming |
| W2D4 | Custom MCP Server Build | MCP & Tool Integration | 🔜 Coming |
| W2D5 | Reflection & Self-Correction Loops | Agent Memory & Capabilities | 🔜 Coming |
| W2D6 | Supervisor vs. Swarm Networks | Multi-Agent Orchestration | 🔜 Coming |
| W2D7 | Deterministic Guardrails (NeMo) | Production Evals & Guardrails | 🔜 Coming |

### Week 3 — Advanced Techniques

| Day | Topic | Vertical | Status |
|---|---|---|---|
| W3D1 | Prompt Distillation | Prompt Engineering & Schemas | 🔜 Coming |
| W3D2 | Context Compression | Context Engineering & Tokens | 🔜 Coming |
| W3D3 | Hybrid Search & Reranking | Advanced RAG | 🔜 Coming |
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
| [Week 1](week-01/) | Foundations | 5 / 7 days |
| Week 2 | Intermediate Patterns | 0 / 7 days |
| Week 3 | Advanced Techniques | 0 / 7 days |
| Week 4 | Production & Scale | 0 / 7 days |

---

## Repository Structure

```
production-ai-engineering-playbook/
├── week-01/                               # Week 1 — Foundations
│   ├── README.md
│   └── W1D1-dspy-programmatic-prompts/
│       ├── README.md                      # Day overview and quick start
│       ├── docs/
│       │   └── technical-document.md      # 21-section deep-dive
│       ├── diagrams/
│       │   ├── architecture.mmd           # Mermaid architecture diagram
│       │   └── sequence.mmd               # Mermaid sequence diagram
│       └── poc/
│           ├── src/
│           │   ├── main.py                # Entry point
│           │   ├── dspy_core.py           # Signatures, predictors, teleprompter
│           │   └── config.py              # Config from environment variables
│           ├── tests/
│           │   └── test_dspy.py           # 16 pytest tests (no API key needed)
│           ├── requirements.txt
│           ├── .env.example
│           ├── sample_input.json
│           └── sample_output.json
├── CONTRIBUTING.md
├── ROADMAP.md
└── README.md
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
