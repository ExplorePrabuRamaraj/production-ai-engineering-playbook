#!/usr/bin/env python3
"""
W2D6 -- Supervisor vs. Swarm Networks
======================================
Demonstrates: two multi-agent orchestration topologies on the same task,
showing routing decisions and latency side-by-side.

Run (demo mode):  DEMO_MODE=true python src/main.py
Run (live mode):  python src/main.py
Run tests:        pytest tests/ -v
"""

import json
import os
import sys
from pathlib import Path

# Allow running from either the repo root or the src/ directory
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from swarm_core import SupervisorNetwork, SwarmNetwork, WorkflowResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()
SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load sample input from file, falling back to an inline default."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text(encoding="utf-8"))
    return {
        "task": "retrieve customer history; analyse sentiment; generate response; validate compliance",
        "description": "Customer support workflow: retrieval, analysis, generation, validation"
    }


# ---------------------------------------------------------------------------
# Demo mode -- pre-computed output, no API key required
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    """Run both topologies against the sample input using demo agents."""
    print("\n[DEMO MODE] Running in demo mode -- no API key required.\n")

    task = input_data.get("task", "analyse and respond to customer inquiry")

    supervisor_net = SupervisorNetwork(max_hops=cfg.swarm_max_hops)
    swarm_net = SwarmNetwork(max_hops=cfg.swarm_max_hops)

    sup_result: WorkflowResult = supervisor_net.run(task, demo_mode=True)
    swarm_result: WorkflowResult = swarm_net.run(task, demo_mode=True)

    return _build_output(sup_result, swarm_result)


# ---------------------------------------------------------------------------
# Live mode -- real API calls (same orchestration, LLM-backed agents)
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    """
    Run with real LLM-backed agents.
    The demo specialist agents are used here too (they produce rule-based
    outputs). In a real system, replace Agent.handle() with an OpenAI call.
    """
    task = input_data.get("task", "analyse and respond to customer inquiry")

    supervisor_net = SupervisorNetwork(max_hops=cfg.swarm_max_hops)
    swarm_net = SwarmNetwork(max_hops=cfg.swarm_max_hops)

    sup_result: WorkflowResult = supervisor_net.run(task, demo_mode=False)
    swarm_result: WorkflowResult = swarm_net.run(task, demo_mode=False)

    return _build_output(sup_result, swarm_result)


# ---------------------------------------------------------------------------
# Shared output builder
# ---------------------------------------------------------------------------

def _build_output(sup: WorkflowResult, swarm: WorkflowResult) -> dict:
    return {
        "supervisor": {
            "topology": sup.topology,
            "subtasks_handled": len(sup.subtask_results),
            "total_latency_ms": sup.total_latency_ms,
            "total_tokens": sup.total_tokens,
            "routing_trace": sup.routing_trace,
            "final_output": sup.final_output,
        },
        "swarm": {
            "topology": swarm.topology,
            "subtasks_handled": len(swarm.subtask_results),
            "total_latency_ms": swarm.total_latency_ms,
            "total_tokens": swarm.total_tokens,
            "routing_trace": swarm.routing_trace,
            "final_output": swarm.final_output,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\nW2D6 -- Supervisor vs. Swarm Networks Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"Task: {input_data.get('task', '')}\n")

    if cfg.demo_mode:
        result = run_demo(input_data)
    else:
        print(f"[LIVE MODE] Using model: {cfg.model}")
        result = run_live(input_data)

    # Display routing traces
    for topology in ("supervisor", "swarm"):
        t = result[topology]
        print(f"\n--- {topology.upper()} NETWORK ---")
        print(f"Subtasks handled : {t['subtasks_handled']}")
        print(f"Total latency    : {t['total_latency_ms']} ms")
        print(f"Total tokens     : {t['total_tokens']}")
        print("Routing trace:")
        for line in t["routing_trace"]:
            print(f"  {line}")

    print("\n--- COMPARISON ---")
    sup_lat = result["supervisor"]["total_latency_ms"]
    swm_lat = result["swarm"]["total_latency_ms"]
    print(f"Supervisor latency : {sup_lat} ms")
    print(f"Swarm latency      : {swm_lat} ms")
    faster = "Supervisor" if sup_lat < swm_lat else "Swarm"
    print(f"Faster topology    : {faster} (on this workload)")

    print("\n[OK] Concept demonstrated: Supervisor routes via central decomposition;")
    print("     Swarm routes via peer-to-peer capability matching.")
    print("\nSee docs/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
