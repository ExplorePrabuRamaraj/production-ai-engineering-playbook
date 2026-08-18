# Week 3 — Advanced Techniques

> Part of the [Production AI Engineering Playbook](../README.md)

---

## Objective

Build on the Week 2 intermediate patterns with advanced production techniques. Week 3 goes deeper on compression, retrieval accuracy, parallel tool execution, dynamic agent capabilities, hierarchical orchestration, and end-to-end observability — the techniques that distinguish a prototype that works from a system that scales.

---

## Topics Covered

Week 3 maps one advanced topic per day across the five production AI engineering verticals:

| Day | Topic | Vertical | Status |
|---|---|---|---|
| [W3D1](W3D1_prompt-distillation/) | Prompt Distillation | Prompt Engineering & Schemas | ✅ Complete |
| [W3D2](W3D2_context-compression/) | Context Compression | Context Engineering & Tokens | ✅ Complete |
| W3D3 | Hybrid Search & Reranking | Advanced RAG | 🔜 Coming |
| W3D4 | Async & Parallel Tool Calls | MCP & Tool Integration | 🔜 Coming |
| W3D5 | Dynamic Skill Selection | Agent Memory & Capabilities | 🔜 Coming |
| W3D6 | Hierarchical Subagent Teams | Multi-Agent Orchestration | 🔜 Coming |
| W3D7 | Distributed Tracing (LangSmith) | Production Evals & Guardrails | 🔜 Coming |

---

## Learning Outcomes

After completing Week 3, you will be able to:

- Compress verbose teacher prompts into minimal student prompts using a greedy pruning loop with a quantified accuracy floor
- Apply context compression techniques to reduce conversation history and retrieved context without losing the information the model needs
- Combine dense vector search with BM25 sparse retrieval and rerank the fused result list for higher precision on entity-rich queries
- Execute multiple tool calls in parallel using async patterns to eliminate sequential latency in MCP-backed agent pipelines
- Implement dynamic skill selection so agents choose the right tool at runtime based on task context rather than hardcoded routing
- Build hierarchical subagent teams where a planner agent decomposes tasks and specialist subagents execute in parallel under its supervision
- Instrument an AI pipeline end-to-end with distributed tracing (LangSmith) to surface latency, token cost, and accuracy regressions across runs

---

## Daily Lessons

### [W3D1 — Prompt Distillation](W3D1_prompt-distillation/)

**Vertical:** Prompt Engineering & Schemas  
**Core problem:** Production LLM prompts bloat organically — every production incident triggers an extra rule or example. A 200-token prompt becomes 1,800 tokens within months, adding ~$54/month overhead at 200k calls/month on gpt-4o-mini before any user content is counted. No one removes instructions because the accuracy impact is unknown.  
**Solution:** A greedy sentence-pruning loop (`distill_prompt()`) that iteratively removes the longest sentence from the teacher prompt, scores the candidate against a labelled eval set via `score_prompt_candidate()`, and accepts the removal only if accuracy stays above the `accuracy_floor`. `compute_token_savings()` projects the resulting monthly and annual cost reduction at a given call volume.

**Key concepts:** `build_teacher_prompt()`, `build_student_prompt()`, `score_prompt_candidate()`, `distill_prompt()`, `compute_token_savings()`, `accuracy_floor`, `max_distillation_iterations`, greedy sentence pruning, load-bearing sentences, DSPy `MIPROv2` as production alternative  
**Status:** ✅ Complete — [Start here →](W3D1_prompt-distillation/README.md)

---

### [W3D2 — Context Compression](W3D2_context-compression/)

**Vertical:** Context Engineering & Tokens  
**Core problem:** As conversation history grows, the context window fills with low-signal turns — early pleasantries, superseded instructions, verbose tool outputs — leaving less room for the current query and recent context. Naive sliding-window eviction discards the oldest turns, but the oldest turns often contain the most important setup context.  
**Solution:** Query-aware context compression — score each sentence by TF-IDF cosine similarity to the current query (`extractive_compress()`), or summarise via LLM (`abstractive_compress()`), reducing input tokens by 40–80% while retaining only what the model needs to answer.

**Key concepts:** `extractive_compress()`, `abstractive_compress()`, `compress_context()`, `CompressionResult`, TF-IDF cosine similarity, token budget, `min_segment_tokens` bypass, hybrid strategy, demo mode  
**Status:** ✅ Complete — [Start here →](W3D2_context-compression/README.md)

---

### W3D3 — Hybrid Search & Reranking

**Vertical:** Advanced RAG  
**Core problem:** Dense vector search excels at semantic similarity but misses exact keyword matches. BM25 sparse search hits exact terms but ignores paraphrase. On entity-rich queries — product codes, legal citations, named individuals — neither alone returns the best results. Precision suffers.  
**Solution:** Run dense and sparse retrieval in parallel, fuse the ranked lists with Reciprocal Rank Fusion (RRF), then rerank the top-N candidates with a cross-encoder for final precision.

**Status:** 🔜 Coming soon

---

### W3D4 — Async & Parallel Tool Calls

**Vertical:** MCP & Tool Integration  
**Core problem:** Sequential tool calls in agent pipelines accumulate latency: if three independent data-fetching tools each take 300ms, the agent waits 900ms before it can synthesise results. At production call volumes, this latency compounds across every user request.  
**Solution:** Async parallel tool dispatch — fire all independent tool calls concurrently with `asyncio.gather()` and collect results when all complete, reducing wall-clock time to the latency of the slowest single call.

**Status:** 🔜 Coming soon

---

### W3D5 — Dynamic Skill Selection

**Vertical:** Agent Memory & Capabilities  
**Core problem:** Hardcoded routing (`if task_type == "X": call_tool_Y()`) breaks when new tools are added, when task types overlap, or when the model's description of the task doesn't match the hardcoded string. Agents need to choose tools at runtime based on capability description, not static if-chains.  
**Solution:** Dynamic skill selection — embed tool capability descriptions and the current task, retrieve the closest-matching skill by vector similarity, and dispatch to it at runtime.

**Status:** 🔜 Coming soon

---

### W3D6 — Hierarchical Subagent Teams

**Vertical:** Multi-Agent Orchestration  
**Core problem:** A flat swarm of peer-to-peer agents (W2D6) works for homogeneous parallel tasks but degrades on tasks that require decomposition, dependency tracking, and result aggregation. Without a planner layer, complex multi-step tasks either run sequentially or produce uncoordinated outputs.  
**Solution:** Hierarchical teams — a planner agent decomposes the task into a dependency graph, dispatches independent subtasks to specialist subagents in parallel, and aggregates their results before returning to the user.

**Status:** 🔜 Coming soon

---

### W3D7 — Distributed Tracing (LangSmith)

**Vertical:** Production Evals & Guardrails  
**Core problem:** LLM pipelines are opaque: a latency spike or accuracy regression in production has no obvious root cause without tracing through each model call, tool invocation, and retrieval step. Without observability, debugging means adding print statements and re-running — expensive and unreliable.  
**Solution:** Instrument the pipeline end-to-end with LangSmith distributed tracing — every LLM call, tool call, and retrieval step emits a structured trace, enabling latency attribution, token cost breakdown, and accuracy tracking across runs.

**Status:** 🔜 Coming soon

---

## Prerequisites

- Python 3.10+
- Basic familiarity with LLM APIs (OpenAI, Anthropic, or equivalent)
- Week 1 and Week 2 completed (or familiarity with the core concepts)

All PoC demos run offline in `DEMO_MODE=true` — no API key required.
