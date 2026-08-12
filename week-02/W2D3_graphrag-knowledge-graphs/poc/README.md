# W2D3 — GraphRAG & Knowledge Graphs

**Series:** AI Engineering Production Playbook
**Vertical:** Advanced RAG
**Week 2 / Day 3**

## What This Demonstrates

GraphRAG combines a knowledge graph with vector search to answer multi-hop relational queries that naive RAG cannot — such as "who approved this contract and under which policy?" — by preserving entity relationships that chunk-based indexing destroys.

## The Core Insight

Naive RAG splits documents into isolated text chunks. When the answer to a question spans multiple entities across multiple documents, the relationships between those entities are lost at chunk boundaries. GraphRAG solves this by:

1. Extracting entities and typed relationships from every chunk at index time
2. Building a property graph (nodes = entities, edges = relationships)
3. Clustering connected nodes into communities with LLM-generated summaries
4. At query time: running vector search AND graph traversal in parallel, then merging results via Reciprocal Rank Fusion (RRF)

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs fully offline)

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/your-org/production-ai-engineering-playbook

# 2. Navigate to this folder
cd week-02/W2D3_graphrag-knowledge-graphs/poc

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key — or leave blank for demo mode

# 5. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode builds a small knowledge graph from the fixtures in `sample_input.json`,
runs the full hybrid retrieval pipeline (graph traversal + RRF merge), and prints
the ranked results — no network call made.

Expected output:

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
  "results": [ ... ],
  "model": "demo",
  "latency_ms": 0
}

  Concept demonstrated: hybrid graph + vector retrieval enables
  multi-hop reasoning over entity relationships.
```

## Run Tests

```bash
pytest tests/ -v
```

All 15 tests pass offline — no API key needed. The test suite covers:

- `TestDemoMode` — demo pipeline runs end-to-end without credentials
- `TestCoreConcept` — graph build, community detection, traversal, RRF merge
- `TestLiveMode` — live OpenAI calls fully mocked via `unittest.mock`
- `TestSampleFiles` — `sample_input.json` and `sample_output.json` schema validation

## Project Structure

```
poc/
├── src/
│   ├── main.py             Entry point — demo and live modes
│   ├── graphrag_core.py    Core logic: build_knowledge_graph, detect_communities,
│   │                       traverse_graph, rrf_merge, hybrid_retrieve
│   └── config.py           Config dataclass loaded from environment variables
├── tests/
│   └── test_graphrag.py    15 unit tests, all offline-capable
├── README.md               This file
├── requirements.txt        Pinned Python dependencies
├── .env.example            Environment variable template (no real values)
├── sample_input.json       7-entity graph fixture with a multi-hop query
└── sample_output.json      Expected output for the sample input
```

## Key Parameters (tunable via .env)

| Parameter | Default | Effect |
|---|---|---|
| `HOP_DEPTH` | 2 | Max graph traversal hops. Increase for longer reasoning chains; cap at 3 to avoid context explosion. |
| `MAX_NODES_PER_TRAVERSAL` | 30 | Hard limit on nodes retrieved per query. Prevents context window overflow. |
| `MIN_EDGE_WEIGHT` | 1 | Filter noisy single-occurrence edges before community detection. |
| `LEIDEN_GAMMA` | 1.0 | Community resolution. Higher = smaller, more specific communities. |
| `RRF_K` | 60 | RRF smoothing constant. Standard value from Cormack et al. (SIGIR 2009). |

## Extending This PoC

**Replace the in-memory graph with Neo4j:**
```python
# In graphrag_core.py, swap the dict-based graph for a Neo4j driver session.
# See .env.example for NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
```

**Add real vector search:**
```python
# Replace the static vector_results fixture in run_demo() with a live
# LanceDB or Pinecone query using the embedded query vector.
```

**Add true Leiden community detection:**
```bash
pip install leidenalg python-igraph
# Then replace detect_communities() with leidenalg.find_partition()
```

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Layman Scenarios](../docs/graphrag-layman-scenarios.md)
- [LinkedIn Post / Day Overview](../README.md)

## References

- Microsoft GraphRAG paper: arXiv:2404.16130
- Leiden algorithm: Traag et al. (2019), Scientific Reports
- RRF: Cormack et al. (SIGIR 2009)
- Microsoft GraphRAG OSS: github.com/microsoft/graphrag
