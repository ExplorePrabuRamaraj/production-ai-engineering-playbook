# W3D3 — Hybrid Search & Reranking

> [Week 3](../README.md) · [Production AI Engineering Playbook](../../README.md)

---

## Overview

Dense vector search excels at semantic similarity but misses exact keyword matches. BM25 sparse search hits exact terms but ignores paraphrase. On entity-rich queries — error codes, product IDs, version strings, named individuals — neither retriever alone returns the best results, and precision suffers. This PoC implements a three-stage hybrid retrieval pipeline: BM25 and dense retrieval run in parallel, their ranked lists are merged using Reciprocal Rank Fusion (RRF), and the top-N fused candidates are precision-ranked by a cross-encoder reranker — combining the strengths of all three stages without requiring score normalisation between incompatible scoring systems.

---

## Learning Objectives

1. Understand why single-retriever RAG fails on entity-rich queries and why neither BM25 nor dense retrieval alone is sufficient
2. Implement `bm25_retrieve()` — BM25Okapi term-frequency scoring with saturation and length normalisation for exact-match queries
3. Implement `dense_retrieve()` — cosine similarity scoring over pre-computed bi-encoder embeddings for semantic/paraphrase queries
4. Implement `reciprocal_rank_fusion()` — merge two ranked lists using RRF score `1 / (k + rank)`, avoiding the incompatible score scales of BM25 and cosine similarity
5. Implement `rerank_with_flashrank()` — cross-encoder joint query-document scoring that outperforms bi-encoders on precision
6. Understand the graceful fallback chain: FlashRank unavailable → return RRF results; rank-bm25 unavailable → simple TF fallback
7. Know when each stage matters: BM25 for recall on rare terms, dense for paraphrase recall, RRF for safe fusion, cross-encoder for final precision

---

## Problem Statement

A developer support chatbot serving 500,000 queries/month operates over a corpus of error messages, API documentation, and forum threads. On exact-match queries like `"RuntimeError: CUDA out of memory"`, dense retrieval returns semantically related GPU memory docs but ranks the exact error document 4th — the embedding averages over the full passage, diluting rare technical terms. On paraphrase queries like `"how do I free GPU memory during training"`, BM25 returns nothing useful because the query shares no exact tokens with the relevant document. Switching between retrievers based on query type is brittle and requires an upstream classifier. Hybrid search with RRF requires no query-type classification — it runs both retrievers in parallel and lets RRF assign credit to documents that rank well in either list, then the cross-encoder makes the final precision call jointly over query and document.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs with pre-computed mock embeddings, no API call)
- `rank-bm25` and `flashrank` are optional — the pipeline degrades gracefully if either is absent
- Familiarity with [W1D3 — Naive vs. Agentic RAG](../../week-01/W1D3-naive-vs-agentic-rag/README.md) provides context on why retrieval quality drives generation quality
- Familiarity with [W2D3 — GraphRAG & Knowledge Graphs](../../week-02/W2D3_graphrag-knowledge-graphs/README.md) provides the Week 2 advanced RAG baseline

---

## Repository Structure

```
W3D3_hybrid-search-reranking/
├── README.md                              # This file
├── docs/
│   ├── technical-document.md              # 21-section practitioner deep-dive
│   └── hybrid-search-layman-scenarios.md  # Business scenarios without ML background
├── diagrams/
│   ├── architecture.mmd                   # Three-stage hybrid pipeline architecture
│   └── sequence.mmd                       # Per-query BM25 + dense + RRF + rerank flow
└── poc/
    ├── README.md                          # PoC quick-start and expected output
    ├── src/
    │   ├── main.py                        # Entry point — demo + live mode
    │   ├── hybrid_search_core.py          # BM25, dense, RRF, reranking logic (pure functions)
    │   └── config.py                      # Config dataclass + env loader
    ├── tests/
    │   └── test_hybrid_search.py          # pytest unit tests (all run offline)
    ├── requirements.txt
    ├── .env.example
    ├── sample_input.json                  # Example query + 6-document corpus
    └── sample_output.json                 # Expected output with RRF + rerank scores
```

---

## Core Concepts

### Stage 1a — BM25 sparse retrieval

`bm25_retrieve()` uses BM25Okapi term-frequency scoring with saturation (k1=1.5) and length normalisation (b=0.75). Documents with zero query-term overlap are filtered out. Falls back to simple TF scoring if `rank-bm25` is not installed:

```python
# hybrid_search_core.py
def bm25_retrieve(query, documents, top_k=100):
    tokenised_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(range(len(documents)), scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [RetrievalResult(..., retriever="bm25") for idx, score in ranked if score > 0]
```

### Stage 1b — Dense retrieval

`dense_retrieve()` computes cosine similarity between the query embedding and each document embedding. Embeddings are passed in — the function is pure and does not call any API itself. In demo mode, pre-computed mock embeddings are used instead:

```python
# hybrid_search_core.py
def dense_retrieve(query, documents, query_embedding, doc_embeddings, top_k=100):
    scores = [(i, _cosine_similarity(query_embedding, doc_emb))
              for i, doc_emb in enumerate(doc_embeddings)]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [RetrievalResult(..., retriever="dense") for idx, score in ranked]
```

### Stage 2 — Reciprocal Rank Fusion

`reciprocal_rank_fusion()` assigns each document a score of `1 / (k + rank)` from each list where it appears, summing across all lists. RRF is preferred over weighted linear combination because BM25 and cosine scores are incompatible scales — mixing them raw skews results. The k=60 smoothing constant (Cormack 2009) limits the reward for being ranked first:

```python
# hybrid_search_core.py
def reciprocal_rank_fusion(ranked_lists, k=60, top_n=50):
    rrf_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_0based, result in enumerate(ranked_list):
            rrf_scores[result.doc_id] = (
                rrf_scores.get(result.doc_id, 0.0) + 1.0 / (k + rank_0based + 1)
            )
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
```

### Stage 3 — Cross-encoder reranking

`rerank_with_flashrank()` scores query-document pairs jointly using a MiniLM cross-encoder. Cross-encoders outperform bi-encoders on precision because the model attends to specific query-relevant phrases within the document. Falls back gracefully to RRF ordering if FlashRank is unavailable:

```python
# hybrid_search_core.py
def rerank_with_flashrank(query, candidates, top_k=5, model_name="ms-marco-MiniLM-L-6-v2"):
    try:
        ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
        request = RerankRequest(query=query, passages=[{"id": doc_id, "text": text} ...])
        reranked = ranker.rerank(request)
        return [HybridSearchResult(...) for item in reranked[:top_k]]
    except ImportError:
        return _rrf_fallback(candidates, top_k)  # graceful degradation
```

### Pipeline orchestrator

`hybrid_search_pipeline()` chains all three stages. Use `use_reranker=False` to return RRF results directly:

```python
# hybrid_search_core.py
def hybrid_search_pipeline(query, documents, query_embedding, doc_embeddings, ...):
    bm25_results  = bm25_retrieve(query, documents, top_k=bm25_top_k)
    dense_results = dense_retrieve(query, documents, query_embedding, doc_embeddings, top_k=dense_top_k)
    fused         = reciprocal_rank_fusion([bm25_results, dense_results], k=rrf_k, top_n=fusion_top_n)
    if use_reranker:
        return rerank_with_flashrank(query, fused, top_k=reranker_top_k)
    return _rrf_fallback(fused, reranker_top_k)
```

---

## Run the PoC

### Demo mode (no API key required)

```bash
cd week-03/W3D3_hybrid-search-reranking/poc
pip install -r requirements.txt
cp .env.example .env
DEMO_MODE=true python src/main.py
```

### Live mode (real embeddings via OpenAI)

```bash
# Edit .env and set OPENAI_API_KEY=your-key
python src/main.py
```

### Skip reranker (return RRF results directly)

```bash
USE_RERANKER=false python src/main.py
```

### Tests

```bash
pytest tests/ -v
# All tests pass offline — external API calls are fully mocked
```

---

## Expected Output

```
  Hybrid Search & Reranking Demo
==================================================
Query: CUDA out of memory error during model training
Corpus size: 6 documents

  Running in DEMO MODE — output is pre-computed (no API call made)

Top results after Hybrid Search + RRF Fusion:
  1. [d1] RRF=0.0317
     BM25 rank=1 | Dense rank=1
     RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB...

  2. [d3] RRF=0.0282
     BM25 rank=3 | Dense rank=2
     How to reduce memory usage during PyTorch training loops...

  3. [d6] RRF=0.0261
     BM25 rank=2 | Dense rank=3
     Gradient checkpointing in PyTorch: trade compute for memory...

  Concept demonstrated: Parallel BM25 + dense retrieval fused via RRF,
  then precision-ranked by a cross-encoder reranker.
  Latency: 0ms | Model: demo
```

---

## PoC File Reference

| File | Purpose |
|---|---|
| `src/main.py` | Entry point — loads config, runs demo or live mode, prints per-result RRF and rerank scores |
| `src/hybrid_search_core.py` | `Document`, `RetrievalResult`, `HybridSearchResult`, `bm25_retrieve()`, `dense_retrieve()`, `reciprocal_rank_fusion()`, `rerank_with_flashrank()`, `hybrid_search_pipeline()` |
| `src/config.py` | `Config` dataclass + `load_config()` with BM25/dense top-k, RRF k, reranker top-k, use_reranker flag |
| `tests/test_hybrid_search.py` | Unit tests for each stage and the full pipeline; all offline (API and FlashRank mocked) |
| `sample_input.json` | 6-document corpus on GPU memory management + query |
| `sample_output.json` | Pre-computed result: d1 top-ranked by both BM25 and dense, rerank score 0.9821 |
| `.env.example` | All environment variable defaults |

---

## Configuration

All settings are loaded from environment variables. Defaults run fully in demo mode:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key — leave blank to auto-enable demo mode |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for dense retrieval embeddings (live mode only) |
| `BM25_TOP_K` | `100` | Candidate documents fetched from BM25 retriever |
| `DENSE_TOP_K` | `100` | Candidate documents fetched from dense retriever |
| `RRF_K` | `60` | RRF smoothing constant (Cormack 2009) |
| `FUSION_TOP_N` | `50` | Documents passed to reranker after RRF fusion |
| `RERANKER_TOP_K` | `5` | Final results returned to caller |
| `USE_RERANKER` | `true` | Set `false` to return RRF results without cross-encoder pass |
| `DEMO_MODE` | `false` | Set `true` to run without API key |

---

## Technical Documentation

- [Technical Document](docs/technical-document.md) — 21-section practitioner deep-dive
- [Layman Scenarios](docs/hybrid-search-layman-scenarios.md) — Business scenarios without ML background required

---

## Architecture Diagrams

- [Architecture Diagram](diagrams/architecture.mmd) — Three-stage hybrid pipeline architecture
- [Sequence Diagram](diagrams/sequence.mmd) — Per-query BM25 + dense + RRF + rerank flow

---

## Connection to the Series

**Previous:** [W3D2 — Context Compression](../W3D2_context-compression/README.md) — reduces the input token cost of the retrieved context before generation; W3D3 improves what gets retrieved in the first place.

**Next:** [W3D4 — Async & Parallel Tool Calls](../W3D4_async-parallel-tool-calls/README.md) — moves from retrieval latency to agent execution latency, firing independent tool calls concurrently with `asyncio.gather()`.

**Series arc:** [W1D3 — Naive vs. Agentic RAG](../../week-01/W1D3-naive-vs-agentic-rag/README.md) introduced the retrieval gap between naive single-vector search and agentic retrieval. [W2D3 — GraphRAG & Knowledge Graphs](../../week-02/W2D3_graphrag-knowledge-graphs/README.md) added structured knowledge traversal for multi-hop queries. W3D3 closes the retrieval precision loop: for queries that are neither purely semantic nor purely lexical, fuse both retrieval signals and let a cross-encoder make the final call.

---

## Key References

- Thakur et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." arXiv:2104.08663
- Cormack, G. V. et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.
- [FlashRank cross-encoder reranker](https://github.com/PrithivirajDamodaran/FlashRank)
- [rank-bm25 library](https://github.com/dorianbrown/rank_bm25)

---

## Continue Learning

**Next:** [W3D4 — Async & Parallel Tool Calls](../W3D4_async-parallel-tool-calls/README.md)

Return to [Week 3 overview](../README.md) to explore all advanced techniques.
