#!/usr/bin/env python3
"""
W3D3 — Hybrid Search & Reranking
==================================
Demonstrates: Three-stage pipeline — parallel BM25 + dense retrieval,
              Reciprocal Rank Fusion, cross-encoder reranking.

Run (live mode):  python src/main.py
Run (demo mode):  DEMO_MODE=true python src/main.py
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env && edit .env
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from hybrid_search_core import (
    Document,
    hybrid_search_pipeline,
    HybridSearchResult,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true" or not OPENAI_API_KEY
SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "query": "CUDA out of memory error during model training",
        "documents": [
            {"id": "d1", "text": "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB."},
            {"id": "d2", "text": "GPU memory management best practices for deep learning."},
            {"id": "d3", "text": "How to reduce memory usage during PyTorch training loops."},
        ],
    }


# ---------------------------------------------------------------------------
# Demo mode — pre-computed output, no API keys required
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    print("\n  Running in DEMO MODE — output is pre-computed (no API call made)\n")

    query = input_data["query"]
    raw_docs = input_data.get("documents", [])
    documents = [Document(id=d["id"], text=d["text"]) for d in raw_docs]

    # Pre-computed mock embeddings: d1 is most semantically similar to query
    mock_query_embedding = [0.9, 0.1, 0.05, 0.8, 0.3]
    mock_doc_embeddings = [
        [0.85, 0.12, 0.06, 0.78, 0.31],   # d1 — high cosine similarity
        [0.60, 0.40, 0.50, 0.55, 0.20],   # d2 — moderate
        [0.65, 0.35, 0.45, 0.60, 0.25],   # d3 — moderate
    ]

    # Run the real pipeline with mock embeddings and no live reranker
    results = hybrid_search_pipeline(
        query=query,
        documents=documents,
        query_embedding=mock_query_embedding,
        doc_embeddings=mock_doc_embeddings,
        bm25_top_k=len(documents),
        dense_top_k=len(documents),
        rrf_k=60,
        fusion_top_n=len(documents),
        reranker_top_k=min(3, len(documents)),
        use_reranker=False,  # FlashRank not required in demo mode
    )

    return {
        "query": query,
        "results": [
            {
                "rank": i + 1,
                "doc_id": r.doc_id,
                "text": r.text,
                "rrf_score": round(r.rrf_score, 4),
                "bm25_rank": r.bm25_rank,
                "dense_rank": r.dense_rank,
            }
            for i, r in enumerate(results)
        ],
        "model": "demo",
        "latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# Live mode — real embeddings via OpenAI API
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    import time

    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not installed. Run: pip install -r requirements.txt")
        raise

    cfg = load_config()
    client = OpenAI(api_key=cfg.openai_api_key)

    query = input_data["query"]
    raw_docs = input_data.get("documents", [])
    documents = [Document(id=d["id"], text=d["text"]) for d in raw_docs]

    # Generate real embeddings for query and all documents
    all_texts = [query] + [d.text for d in documents]
    start = time.perf_counter()
    response = client.embeddings.create(model=cfg.embedding_model, input=all_texts)
    query_embedding = response.data[0].embedding
    doc_embeddings = [e.embedding for e in response.data[1:]]
    latency_ms = int((time.perf_counter() - start) * 1000)

    results = hybrid_search_pipeline(
        query=query,
        documents=documents,
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        bm25_top_k=cfg.bm25_top_k,
        dense_top_k=cfg.dense_top_k,
        rrf_k=cfg.rrf_k,
        fusion_top_n=cfg.fusion_top_n,
        reranker_top_k=cfg.reranker_top_k,
        use_reranker=cfg.use_reranker,
        reranker_model=cfg.reranker_model,
    )

    return {
        "query": query,
        "results": [
            {
                "rank": i + 1,
                "doc_id": r.doc_id,
                "text": r.text,
                "rrf_score": round(r.rrf_score, 4),
                "rerank_score": r.rerank_score,
                "bm25_rank": r.bm25_rank,
                "dense_rank": r.dense_rank,
            }
            for i, r in enumerate(results)
        ],
        "model": cfg.embedding_model,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\n  Hybrid Search & Reranking Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"Query: {input_data['query']}")
    print(f"Corpus size: {len(input_data.get('documents', []))} documents\n")

    if DEMO_MODE:
        result = run_demo(input_data)
    else:
        print(f"  Using embedding model: {load_config().embedding_model}")
        result = run_live(input_data)

    print("Top results after Hybrid Search + RRF Fusion:")
    for r in result["results"]:
        bm25_str = f"BM25 rank={r['bm25_rank']}" if r["bm25_rank"] else "BM25 not retrieved"
        dense_str = f"Dense rank={r['dense_rank']}" if r["dense_rank"] else "Dense not retrieved"
        rerank_str = f"  rerank={r.get('rerank_score', 'N/A')}" if r.get("rerank_score") else ""
        print(f"  {r['rank']}. [{r['doc_id']}] RRF={r['rrf_score']}{rerank_str}")
        print(f"     {bm25_str} | {dense_str}")
        print(f"     {r['text'][:80]}...")
        print()

    print(f"  Concept demonstrated: Parallel BM25 + dense retrieval fused via RRF,")
    print(f"  then precision-ranked by a cross-encoder reranker.")
    print(f"  Latency: {result['latency_ms']}ms | Model: {result['model']}")
    print("\n  See docs/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
