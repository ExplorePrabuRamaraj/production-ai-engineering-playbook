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
| [W3D3](W3D3_hybrid-search-reranking/) | Hybrid Search & Reranking | Advanced RAG | ✅ Complete |
| [W3D4](W3D4_async-parallel-tool-calls/) | Async & Parallel Tool Calls | MCP & Tool Integration | ✅ Complete |
| [W3D5](W3D5_dynamic-skill-selection/) | Dynamic Skill Selection | Agent Memory & Capabilities | ✅ Complete |
| [W3D6](W3D6_hierarchical-subagent-teams/) | Hierarchical Subagent Teams | Multi-Agent Orchestration | ✅ Complete |
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

### [W3D3 — Hybrid Search & Reranking](W3D3_hybrid-search-reranking/)

**Vertical:** Advanced RAG  
**Core problem:** Dense vector search excels at semantic similarity but misses exact keyword matches. BM25 sparse search hits exact terms but ignores paraphrase. On entity-rich queries — error codes, product IDs, version strings — neither retriever alone returns the best results. Precision suffers.  
**Solution:** Three-stage pipeline — BM25 and dense retrieval run in parallel, their ranked lists are merged with Reciprocal Rank Fusion (`reciprocal_rank_fusion()`), and the fused top-N are precision-ranked by a cross-encoder (`rerank_with_flashrank()`). RRF avoids score-scale incompatibility between BM25 and cosine similarity entirely.

**Key concepts:** `bm25_retrieve()`, `dense_retrieve()`, `reciprocal_rank_fusion()`, `rerank_with_flashrank()`, `hybrid_search_pipeline()`, RRF k=60, BM25Okapi, bi-encoder vs. cross-encoder, graceful fallback  
**Status:** ✅ Complete — [Start here →](W3D3_hybrid-search-reranking/README.md)

---

### [W3D4 — Async & Parallel Tool Calls](W3D4_async-parallel-tool-calls/)

**Vertical:** MCP & Tool Integration  
**Core problem:** Sequential tool calls in agent pipelines accumulate latency: four independent data-fetching tools at ~300ms each = ~1,200ms before synthesis. At 200,000 calls/month that compounds to 240+ hours of aggregate user-facing delay.  
**Solution:** Async parallel fan-out — `dispatch_tools_parallel()` fires all independent coroutines with `asyncio.gather(return_exceptions=True)`, bounds each with `asyncio.wait_for()`, caps concurrency with `asyncio.Semaphore`, and converts every outcome (success, timeout, error) into a typed `ToolResult` so the LLM always receives an explicit signal.

**Key concepts:** `dispatch_tools_parallel()`, `aggregate_results()`, `compute_speedup()`, `ToolResult`, `asyncio.gather(return_exceptions=True)`, `asyncio.wait_for()`, `asyncio.Semaphore`, explicit fallback strings, sequential baseline vs parallel wall clock  
**Status:** ✅ Complete — [Start here →](W3D4_async-parallel-tool-calls/README.md)

---

### [W3D5 — Dynamic Skill Selection](W3D5_dynamic-skill-selection/)

**Vertical:** Agent Memory & Capabilities  
**Core problem:** Injecting all tool schemas into every LLM prompt wastes thousands of tokens and degrades selection accuracy as similar tools compete for attention. An agent with 30 tools spends ~4,500 tokens per call on tool definitions alone.  
**Solution:** Treat tool routing as retrieval — `SkillRegistry` stores skill descriptions with embeddings, `EmbeddingRouter.select()` scores all skills by cosine similarity to the query and injects only the top-k, and `SkillInjector.build_tool_block()` serialises them into OpenAI function-calling format. Role-based permission filtering and stale-skill eviction included.

**Key concepts:** `SkillRegistry`, `EmbeddingRouter`, `SkillInjector`, `SelectionResult`, `similarity_threshold`, `top_k`, `evict_stale()`, role-based permission filtering, fallback activation, FAISS scaling path  
**Status:** ✅ Complete — [Start here →](W3D5_dynamic-skill-selection/README.md)

---

### [W3D6 — Hierarchical Subagent Teams](W3D6_hierarchical-subagent-teams/)

**Vertical:** Multi-Agent Orchestration  
**Core problem:** A flat agent pool accumulates each other's context, retries fail across the whole pool, and the synthesiser sees raw, inconsistently-formatted worker outputs — leading to context bleed, duplicate work, and silent data loss on partial failures.  
**Solution:** A 3-tier hierarchy — `run_orchestrator()` decomposes and synthesises, `run_team_lead()` owns a domain and retries individual workers with scoped retry, `run_worker()` executes one atomic task statelessly. Typed contracts (`WorkerResult`, `LeadResult`, `FinalResult`) at every tier boundary prevent raw LLM strings from crossing tiers.

**Key concepts:** `run_orchestrator()`, `run_team_lead()`, `run_worker()`, `SubtaskSpec`, `WorkerResult`, `LeadResult`, `FinalResult`, `ExecutionOrder`, scoped retry, partial result flag, typed tier contracts, context bleed prevention  
**Status:** ✅ Complete — [Start here →](W3D6_hierarchical-subagent-teams/README.md)

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
