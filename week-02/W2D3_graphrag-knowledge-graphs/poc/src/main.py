#!/usr/bin/env python3
"""
W2D3 — GraphRAG & Knowledge Graphs
====================================
Demonstrates: Hybrid retrieval combining a knowledge graph with vector search,
enabling multi-hop reasoning over entity relationships that naive RAG cannot answer.

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

# ---------------------------------------------------------------------------
# Configuration — all secrets come from environment variables, never hardcoded
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from graphrag_core import (
    RetrievalResult,
    build_knowledge_graph,
    detect_communities,
    hybrid_retrieve,
)

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load the sample query and fixtures from sample_input.json."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "query": "Who approved the contract and which policy governs it?",
        "seed_entities": ["Alice", "Contract_42"],
        "triples": [
            ["Alice", "APPROVED", "Contract_42"],
            ["Contract_42", "GOVERNED_BY", "Policy_GDPR_17"],
            ["Alice", "REPORTS_TO", "Bob"],
        ],
    }


# ---------------------------------------------------------------------------
# Demo mode — pre-computed graph + retrieval results for offline demonstration
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    """
    Build a small knowledge graph from the sample triples and run hybrid retrieval.
    No API key required — demonstrates the full GraphRAG pipeline with static data.
    """
    print("\n  Running in DEMO MODE — graph built from sample_input.json fixtures\n")

    config = load_config()
    triples = [(t[0], t[1], t[2]) for t in input_data.get("triples", [])]
    query = input_data.get("query", "")
    seed_entities = input_data.get("seed_entities", [])

    # Build knowledge graph from sample triples
    graph = build_knowledge_graph(triples)
    print(f"  Graph built: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    # Detect communities
    communities = detect_communities(graph, gamma=config.leiden_gamma)
    print(f"  Communities detected: {len(communities)}")

    # Simulate vector search results (in production: query a live vector index)
    vector_results = [
        RetrievalResult(
            text="Alice reviewed and signed Contract_42 on 2024-03-15.",
            source="vector",
            score=0.91,
        ),
        RetrievalResult(
            text="Policy GDPR Article 17 governs all data processing contracts.",
            source="vector",
            score=0.78,
        ),
    ]

    # Hybrid retrieval: graph traversal + vector results merged via RRF
    results = hybrid_retrieve(
        query=query,
        graph=graph,
        communities=communities,
        seed_entities=seed_entities,
        vector_results=vector_results,
        hop_depth=config.hop_depth,
        max_nodes=config.max_nodes_per_traversal,
        rrf_k=config.rrf_k,
        top_m=config.top_k_merged,
    )

    return {
        "query": query,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "communities": len(communities),
        "results": [
            {
                "text": r.text,
                "source": r.source,
                "rrf_score": r.score,
                "provenance": r.provenance,
            }
            for r in results
        ],
        "model": "demo",
        "latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# Live mode — real LLM-based entity extraction + answer generation
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    """
    Run GraphRAG with live OpenAI calls for entity extraction and answer generation.
    Only called when OPENAI_API_KEY is set and DEMO_MODE is false.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not installed. Run: pip install -r requirements.txt")
        raise

    config = load_config()
    client = OpenAI(api_key=config.openai_api_key)
    query = input_data.get("query", "")
    triples = [(t[0], t[1], t[2]) for t in input_data.get("triples", [])]
    seed_entities = input_data.get("seed_entities", [])

    graph = build_knowledge_graph(triples)
    communities = detect_communities(graph, gamma=config.leiden_gamma)

    # Run hybrid retrieval (vector results simulated for PoC)
    results = hybrid_retrieve(
        query=query,
        graph=graph,
        communities=communities,
        seed_entities=seed_entities,
        vector_results=None,
        hop_depth=config.hop_depth,
        top_m=config.top_k_merged,
    )

    # Assemble context from merged results
    context = "\n".join(f"- {r.text}" for r in results[:5])

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Answer using ONLY the provided context. "
                    "If the context is insufficient, say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return {
        "query": query,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "communities": len(communities),
        "answer": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens,
        "model": response.model,
        "latency_ms": 0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    print("\n  GraphRAG & Knowledge Graphs Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"  Query: {input_data.get('query', '')}")
    print(f"  Seed entities: {input_data.get('seed_entities', [])}\n")

    if config.demo_mode:
        result = run_demo(input_data)
    else:
        print(f"  Using model: {config.model}")
        result = run_live(input_data)

    print(f"\n  Output:\n{json.dumps(result, indent=2)}")
    print("\n  Concept demonstrated: hybrid graph + vector retrieval enables")
    print("  multi-hop reasoning over entity relationships.\n")
    print("  See 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
