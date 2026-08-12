# W2D3 — GraphRAG & Knowledge Graphs

> [Week 2](../README.md) · [Playbook](../../README.md)

## Overview

GraphRAG combines a property knowledge graph with vector search to answer multi-hop relational queries that naive RAG cannot — such as "who approved this contract and under which policy?" — by preserving entity relationships that chunk-based indexing destroys. At index time, entities and typed relationships are extracted from every chunk; at query time, graph traversal and vector search run in parallel and are merged via Reciprocal Rank Fusion (RRF).

**Vertical:** Advanced RAG | **Week:** 2 | **Day:** 3

---

## Learning Objectives

1. Understand why naive vector RAG fails on multi-hop relational queries — and what relationship information is lost at chunk boundaries
2. Build a property knowledge graph from (subject, predicate, object) triples using `build_knowledge_graph()`
3. Implement community detection using connected-components clustering as a Leiden algorithm proxy
4. Execute N-hop graph traversal from seed entities with `traverse_graph()` and configurable hop depth
5. Merge graph and vector results using Reciprocal Rank Fusion (RRF) with `rrf_merge()`
6. Run the full hybrid retrieval pipeline with `hybrid_retrieve()` — graph traversal + vector search + RRF in one call
7. Know when to replace the in-memory graph with Neo4j or Amazon Neptune for corpora exceeding 500,000 nodes

---

## Problem Statement

Naive RAG splits documents into isolated text chunks and stores embeddings indexed by semantic similarity. When a question requires understanding how entities *relate* across multiple documents — "who approved Contract_42 and which policy governs it?" — the chunk containing Alice, the chunk containing Contract_42, and the chunk containing Policy_GDPR_17 may never land in the same top-k retrieval result.

On entity-dense corpora with thousands of documents, flat vector retrieval returns the semantically closest chunks but misses relationship chains that span chunk boundaries. Microsoft's GraphRAG benchmarks show 30–60% accuracy improvement on multi-hop queries compared to naive vector RAG, precisely because the graph encodes the relationships that embeddings cannot.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs fully offline)
- [W1D3 — Naive vs. Agentic RAG](../../week-01/W1D3-naive-vs-agentic-rag/README.md) recommended

---

## Repository Structure

```
W2D3_graphrag-knowledge-graphs/
├── README.md                               # This file
├── docs/
│   ├── technical-document.md              # 21-section practitioner deep-dive
│   └── graphrag-layman-scenarios.md       # Business scenarios, no ML background needed
├── diagrams/
│   ├── architecture.mmd                   # System architecture (Mermaid)
│   └── sequence.mmd                       # Query-time retrieval sequence (Mermaid)
└── poc/
    ├── README.md                          # PoC quickstart and expected output
    ├── src/
    │   ├── main.py                        # Entry point — demo and live modes
    │   ├── graphrag_core.py               # Core: build_knowledge_graph, detect_communities,
    │   │                                  #        traverse_graph, rrf_merge, hybrid_retrieve
    │   └── config.py                      # Config dataclass loaded from environment
    ├── tests/
    │   └── test_graphrag.py               # 15 unit tests, all offline-capable
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                  # 7-entity graph fixture with a multi-hop query
    └── sample_output.json                 # Expected output for the sample input
```

---

## Core Concepts

### Step 1 — Build Knowledge Graph

```python
from graphrag_core import build_knowledge_graph

triples = [
    ("Alice", "APPROVED", "Contract_42"),
    ("Contract_42", "GOVERNED_BY", "Policy_GDPR_17"),
    ("Alice", "REPORTS_TO", "Bob"),
]
graph = build_knowledge_graph(triples)
# {"nodes": {entity_id: Entity}, "edges": {(src, tgt, rel_type): Relationship}}
```

Entities are normalised to lowercase-underscore IDs. Duplicate triples increment the edge `weight` rather than creating duplicate edges — this lets `MIN_EDGE_WEIGHT` filter extraction noise before community detection.

### Step 2 — Detect Communities

```python
from graphrag_core import detect_communities

communities = detect_communities(graph, gamma=1.0)
# [Community(community_id="c000", member_ids=["alice", "contract_42", ...], summary="...")]
```

Uses connected-components BFS as a simplified Leiden proxy. Each community gets an auto-generated summary that describes its member entities. In production, replace `detect_communities()` with `leidenalg.find_partition()` for true modularity maximisation.

### Step 3 — Hybrid Retrieval (Graph + Vector via RRF)

```python
from graphrag_core import hybrid_retrieve

results = hybrid_retrieve(
    query="Who approved Contract_42 and which policy governs it?",
    graph=graph,
    communities=communities,
    seed_entities=["Alice", "Contract_42"],
    vector_results=vector_results,  # from your vector index (LanceDB, Pinecone, etc.)
    hop_depth=2,
    rrf_k=60,
    top_m=8,
)
```

Graph traversal collects the N-hop neighbourhood of seed entities; RRF merges the ranked graph results with the ranked vector results into a single list by combining reciprocal rank scores: `RRF(d) = Σ 1 / (k + rank_i)`. This is rank-based, so it handles score-scale differences between cosine similarity (0–1) and graph proximity scores automatically.

---

## Run the PoC

**Demo mode (no API key required):**

```bash
cd week-02/W2D3_graphrag-knowledge-graphs/poc
pip install -r requirements.txt
DEMO_MODE=true python src/main.py
```

**Live mode (OpenAI API key required):**

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python src/main.py
```

**Run tests:**

```bash
pytest tests/ -v
```

---

## Expected Output

```
  GraphRAG & Knowledge Graphs Demo
==================================================
  Query: Who approved Contract_42 and which policy governs it?
  Seed entities: ['Alice', 'Contract_42']

  Running in DEMO MODE — graph built from sample_input.json fixtures

  Graph built: 7 nodes, 7 edges
  Communities detected: 1

  Output:
{
  "query": "Who approved Contract_42 and which policy governs it?",
  "graph_nodes": 7,
  "graph_edges": 7,
  "communities": 1,
  "results": [
    {
      "text": "Entity: Alice (type: Person). Reached via path: alice.",
      "source": "graph",
      "rrf_score": 0.016393,
      "provenance": "alice"
    },
    {
      "text": "Alice reviewed and signed Contract_42 on 2024-03-15.",
      "source": "vector",
      "rrf_score": 0.016129,
      "provenance": ""
    },
    ...
  ],
  "model": "demo",
  "latency_ms": 0
}

  Concept demonstrated: hybrid graph + vector retrieval enables
  multi-hop reasoning over entity relationships.
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — selects demo or live mode, calls `hybrid_retrieve()` |
| `src/graphrag_core.py` | Pure functions: `build_knowledge_graph`, `detect_communities`, `traverse_graph`, `rrf_merge`, `hybrid_retrieve` |
| `src/config.py` | `Config` dataclass + `load_config()` from environment variables |
| `tests/test_graphrag.py` | 15 tests: `TestDemoMode`, `TestCoreConcept`, `TestLiveMode`, `TestSampleFiles` |
| `sample_input.json` | 7-entity, 7-edge fixture with a multi-hop query |
| `sample_output.json` | Expected hybrid retrieval output for the sample input |

---

## Configuration

All parameters are loaded from environment variables (see `.env.example`):

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for live mode. Absent → demo mode activates automatically. |
| `MODEL` | `gpt-4o-mini` | OpenAI model for live answer generation. |
| `HOP_DEPTH` | `2` | Max graph traversal hops. Cap at 3 to avoid context explosion. |
| `MAX_NODES_PER_TRAVERSAL` | `30` | Hard limit on nodes retrieved per query. Prevents context overflow. |
| `MIN_EDGE_WEIGHT` | `1` | Filter noisy single-occurrence edges before community detection. |
| `LEIDEN_GAMMA` | `1.0` | Community resolution. Higher = smaller, more specific communities. |
| `RRF_K` | `60` | RRF smoothing constant (Cormack et al., SIGIR 2009). |
| `TOP_K_VECTOR` | `5` | Number of vector search results to retrieve from the index. |
| `TOP_K_MERGED` | `8` | Number of merged results passed to the LLM as context. |
| `DEMO_MODE` | `false` | Set `true` to force demo mode regardless of API key. |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section deep-dive on GraphRAG architecture, Leiden community detection, RRF fusion, and production deployment patterns
- [Layman Scenarios](docs/graphrag-layman-scenarios.md) — Business scenarios explaining GraphRAG without ML background

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — End-to-end GraphRAG system: entity extraction → graph build → community detection → hybrid retrieval → LLM answer generation
- [Sequence Diagram](diagrams/sequence.mmd) — Query-time flow: seed entity extraction → parallel graph traversal + vector search → RRF merge → context assembly → LLM response

---

## Connection to the Series

| | Day | Topic |
|---|---|---|
| ← Previous | [W2D2 — KV Caching & Token Trimming](../W2D2_kv-caching-token-trimming/README.md) | Cost-efficient context management |
| → Next | W2D4 — Custom MCP Server Build | Typed tools for internal systems |

**Why this follows W2D2:** W2D2 managed *what goes into context* — token budget and KV reuse. W2D3 improves *what you retrieve* — ensuring the right information reaches context even when answers span multiple entity relationships across documents.

---

## Key References

- Microsoft GraphRAG paper: [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- Leiden algorithm: Traag et al. (2019), *Scientific Reports*
- RRF: Cormack et al. (SIGIR 2009)
- Microsoft GraphRAG OSS: [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)

---

## Continue Learning

**Next:** W2D4 — Custom MCP Server Build *(coming soon)*
