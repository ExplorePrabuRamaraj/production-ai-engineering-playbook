#!/usr/bin/env python3
"""
W1D3 — Naive vs. Agentic RAG
==============================
Demonstrates: side-by-side comparison of single-pass retrieval vs.
              the decompose → retrieve → validate → synthesise loop.

Run:             python src/main.py
Run (demo mode): DEMO_MODE=true python src/main.py
"""

import json
import os
import sys

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from rag_core import AgenticRAGPipeline, NaiveRAGPipeline


def load_sample_query() -> str:
    """Load the demo query from sample_input.json, fallback to a hardcoded default."""
    input_path = os.path.join(os.path.dirname(__file__), "..", "sample_input.json")
    try:
        with open(input_path) as f:
            data = json.load(f)
            return data.get("query", _default_query())
    except (FileNotFoundError, json.JSONDecodeError):
        return _default_query()


def _default_query() -> str:
    return (
        "My order was marked delivered but I never received it — "
        "what do I do and does my Gold membership change my options?"
    )


def print_result(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"Sub-questions : {result.sub_questions}")
    print(f"Retrieval calls: {result.retrieval_calls}  |  Chunks used: {result.chunks_used}")
    print(f"Answer:\n  {result.answer}")


def main() -> None:
    config = load_config()

    print("\nRAG Pipeline Demo — Naive vs. Agentic")
    print("=" * 42)

    if config.demo_mode:
        print("Running in demo mode (no API key). Output uses pre-computed corpus.\n")
    else:
        print("Running in live mode with OpenAI API.\n")

    query = load_sample_query()
    print(f"Query: {query}\n")

    # Run naive pipeline first — single retrieval pass, no decomposition
    naive_pipeline = NaiveRAGPipeline(config=config)
    naive_result = naive_pipeline.run(query)
    print_result("Naive RAG", naive_result)

    # Run agentic pipeline — decompose, retrieve per sub-question, validate, synthesise
    agentic_pipeline = AgenticRAGPipeline(config=config)
    agentic_result = agentic_pipeline.run(query)
    print_result("Agentic RAG", agentic_result)

    print("\n" + "=" * 42)
    print("Concept demonstrated: Agentic RAG decomposes multi-hop queries and")
    print("retrieves targeted evidence per sub-question; naive RAG retrieves")
    print("by surface similarity alone and misses membership-specific context.")

    # Write output to sample_output.json for reference
    output_path = os.path.join(os.path.dirname(__file__), "..", "sample_output.json")
    output = {
        "query": query,
        "naive_rag": {
            "sub_questions": naive_result.sub_questions,
            "retrieval_calls": naive_result.retrieval_calls,
            "chunks_used": naive_result.chunks_used,
            "answer": naive_result.answer,
        },
        "agentic_rag": {
            "sub_questions": agentic_result.sub_questions,
            "retrieval_calls": agentic_result.retrieval_calls,
            "chunks_used": agentic_result.chunks_used,
            "answer": agentic_result.answer,
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to sample_output.json")


if __name__ == "__main__":
    main()
