# W3D3 — Hybrid Search & Reranking

**Series:** AI Engineering Production Playbook
**Vertical:** Advanced RAG
**Week 3 / Day 3**

## What This Demonstrates

A three-stage RAG retrieval pipeline that combines BM25 sparse retrieval and dense vector retrieval in parallel, merges their results using Reciprocal Rank Fusion (RRF), and applies a cross-encoder reranker to precision-rank the top-N candidates — covering both exact-match and semantic queries that no single retriever handles alone.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd week-03/W3D3_hybrid-search-reranking/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key (or leave blank for demo mode)

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode runs the full BM25 + RRF pipeline with pre-computed mock embeddings. No API key, no network access, no external model download required.

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline — external API calls are fully mocked.

## Skip the Reranker

To return RRF fusion results directly without cross-encoder reranking:

```bash
USE_RERANKER=false python src/main.py
```

Useful for latency-sensitive workloads or when FlashRank is not installed.

## Project Structure

```
poc/
├── src/
│   ├── main.py                 # Entry point — run this file
│   ├── hybrid_search_core.py   # BM25, dense retrieval, RRF, reranking logic
│   └── config.py               # Config dataclass + environment loader
├── tests/
│   └── test_hybrid_search.py   # pytest unit tests (all offline)
├── requirements.txt
├── .env.example
├── sample_input.json           # Example query + document corpus
└── sample_output.json          # Expected output with RRF and rerank scores
```

## Pipeline Stages

| Stage | What It Does | Key Parameter |
|---|---|---|
| BM25 Retrieval | Exact-match keyword scoring | `BM25_TOP_K=100` |
| Dense Retrieval | Embedding cosine similarity | `DENSE_TOP_K=100` |
| RRF Fusion | Merges both ranked lists | `RRF_K=60` |
| Cross-Encoder Reranking | Joint query-doc scoring | `RERANKER_TOP_K=5` |

## Expected Output

```
  Hybrid Search & Reranking Demo
==================================================
Query: CUDA out of memory error during model training
Corpus size: 6 documents

Top results after Hybrid Search + RRF Fusion:
  1. [d1] RRF=0.0317
     BM25 rank=1 | Dense rank=1
     RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB...

  2. [d3] RRF=0.0282
     BM25 rank=3 | Dense rank=2
     How to reduce memory usage during PyTorch training loops...

  Concept demonstrated: Parallel BM25 + dense retrieval fused via RRF,
  then precision-ranked by a cross-encoder reranker.
```

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Layman Scenarios](../docs/hybrid-search-layman-scenarios.md)
- [LinkedIn Post](../README.md)

## References

- Thakur et al. (2021). BEIR Benchmark. arXiv:2104.08663
- Cormack et al. (2009). Reciprocal Rank Fusion. SIGIR 2009.
- FlashRank: https://github.com/PrithivirajDamodaran/FlashRank
- rank-bm25: https://github.com/dorianbrown/rank_bm25
