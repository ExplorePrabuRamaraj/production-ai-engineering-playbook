#!/usr/bin/env python3
"""
W1D6 — State Graphs (LangGraph)
================================
Demonstrates: A document triage workflow with typed shared state,
conditional edges (low-risk vs high-risk path), and human-in-the-loop
interrupt simulation — all running in demo mode without an API key.

Run (demo mode):  DEMO_MODE=true python src/main.py
Run (live mode):  python src/main.py   (requires OPENAI_API_KEY)
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env  # add API key for live mode (optional)
"""

import os
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — all values come from environment variables
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true" or not OPENAI_API_KEY

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"

# ---------------------------------------------------------------------------
# Input loader
# ---------------------------------------------------------------------------

def load_sample_input() -> dict:
    """Load sample input from file, falling back to an inline example."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "document_text": (
            "This agreement shall indemnify the party against all liability. "
            "Any disputes shall be resolved through arbitration. "
            "Termination for cause requires thirty days written notice. "
            "The vendor warrants all deliverables meet specification."
        )
    }

# ---------------------------------------------------------------------------
# Demo mode — simulates graph execution without langgraph or API key
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    """
    Run the state graph simulation using pure Python node functions.
    No langgraph installation or API key required.
    Produces output that mirrors what the live graph would return.
    """
    print("\n  Running in DEMO MODE — langgraph simulation (no API call)\n")
    # Import here so an ImportError surfaces clearly
    sys.path.insert(0, str(Path(__file__).parent))
    from state_graph_core import run_demo_graph

    final_state = run_demo_graph(input_data["document_text"])
    return {
        "risk_score": final_state["risk_score"],
        "risk_label": final_state["risk_label"],
        "flags": final_state["flags"],
        "human_approved": final_state["human_approved"],
        "summary": final_state["summary"],
        "clauses_extracted": len(final_state["clauses"]),
        "model": "demo",
        "latency_ms": 0,
    }

# ---------------------------------------------------------------------------
# Live mode — runs the real LangGraph graph with OpenAI LLM nodes
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    """
    Build and invoke the real LangGraph state graph.
    Requires langgraph, langchain-openai, and OPENAI_API_KEY.
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        sys.path.insert(0, str(Path(__file__).parent))
        from state_graph_core import build_graph

        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)
        if graph is None:
            raise ImportError("langgraph not available — use demo mode")

        config = {"configurable": {"thread_id": "poc-run-001"}}
        # recursion_limit guards against infinite retry loops
        result = graph.invoke(
            {
                "document_text": input_data["document_text"],
                "clauses": [],
                "risk_score": 0.0,
                "risk_label": "low_risk",
                "flags": [],
                "human_approved": None,
                "summary": None,
                "retry_count": 0,
                "error": None,
            },
            config,
            {"recursion_limit": 10},
        )
        return {
            "risk_score": result["risk_score"],
            "risk_label": result["risk_label"],
            "flags": result["flags"],
            "human_approved": result.get("human_approved"),
            "summary": result["summary"],
            "clauses_extracted": len(result["clauses"]),
            "model": "langgraph-live",
            "latency_ms": 0,
        }
    except ImportError as e:
        print(f"  langgraph not installed ({e}). Falling back to demo mode.")
        return run_demo(input_data)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\n  State Graphs (LangGraph) Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"  Input document ({len(input_data['document_text'])} chars):")
    print(f"  \"{input_data['document_text'][:120]}...\"\n")

    if DEMO_MODE:
        result = run_demo(input_data)
    else:
        print(f"  Using model: {os.getenv('MODEL', 'gpt-4o-mini')}")
        result = run_live(input_data)

    print(f"\n  Output:")
    print(f"  risk_score       : {result['risk_score']:.2f}")
    print(f"  risk_label       : {result['risk_label']}")
    print(f"  flags            : {result['flags']}")
    print(f"  human_approved   : {result['human_approved']}")
    print(f"  clauses_extracted: {result['clauses_extracted']}")
    print(f"  summary          : {result['summary']}")
    print(f"  model            : {result['model']}")

    print(
        "\n  Concept demonstrated: typed shared state flows through a "
        "conditional state graph — low-risk documents auto-process, "
        "high-risk documents pause for human approval."
    )
    print("\n  See docs/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
