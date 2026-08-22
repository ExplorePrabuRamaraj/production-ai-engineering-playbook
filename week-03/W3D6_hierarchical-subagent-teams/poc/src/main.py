#!/usr/bin/env python3
"""
W3D6 — Hierarchical Subagent Teams
====================================
Demonstrates: A 3-tier agent hierarchy (Orchestrator → Team Leads → Workers)
with typed result contracts at every tier boundary.

Run (live mode):  python src/main.py
Run (demo mode):  DEMO_MODE=true python src/main.py
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  Set OPENAI_API_KEY in environment (or leave blank for demo mode)
"""

import os
import json
from pathlib import Path

from config import load_config
from hierarchical_core import (
    SubtaskSpec,
    ExecutionOrder,
    run_orchestrator,
    run_demo,
)

# ---------------------------------------------------------------------------
# Configuration — all secrets from environment variables, never hardcoded
# ---------------------------------------------------------------------------
cfg = load_config()

# ---------------------------------------------------------------------------
# Load sample input
# ---------------------------------------------------------------------------
SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load sample input from file; fall back to inline example."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "goal": "Analyse competitive landscape for a SaaS CRM product",
        "domains": ["Competitive Research", "Market Analysis"],
    }


# ---------------------------------------------------------------------------
# Pre-built TaskPlan for the live demo (in production the orchestrator
# generates this via an LLM decomposition call)
# ---------------------------------------------------------------------------

def build_task_plan(input_data: dict) -> tuple:
    """Return (subtask_specs, worker_map) for the demo goal."""
    goal = input_data.get("goal", "")

    subtask_specs = [
        SubtaskSpec(
            lead_id="lead_research",
            domain="Competitive Research",
            instruction=f"Research direct competitors for: {goal}. Identify top 3 competitors, their pricing, and key differentiators.",
            execution_order=ExecutionOrder.PARALLEL,
        ),
        SubtaskSpec(
            lead_id="lead_analysis",
            domain="Market Analysis",
            instruction=f"Analyse market size and growth trends for: {goal}. Provide TAM, SAM, and YoY growth rate.",
            execution_order=ExecutionOrder.PARALLEL,
        ),
    ]

    # Each lead receives a list of atomic worker instructions.
    # Workers only see their specific instruction + the lead's subtask as context.
    worker_map = {
        "lead_research": [
            {
                "worker_id": "lead_research_worker_0",
                "instruction": "List the top 3 direct competitors with their market share and recent pricing changes.",
            },
            {
                "worker_id": "lead_research_worker_1",
                "instruction": "Identify the top 3 feature gaps and top 2 competitive advantages vs the market leader.",
            },
        ],
        "lead_analysis": [
            {
                "worker_id": "lead_analysis_worker_0",
                "instruction": "Provide TAM, SAM, YoY growth rate, and fastest-growing subsegment with growth rate.",
            },
        ],
    }

    return subtask_specs, worker_map


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\n[DEMO] Hierarchical Subagent Teams")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"Goal: {input_data.get('goal', '')}\n")

    if cfg.demo_mode:
        print("[DEMO MODE] Output is pre-computed (no API call made)\n")
        result = run_demo(input_data)
    else:
        print(f"⚙️  Using model: {cfg.model}")
        print("Building task plan and dispatching hierarchy...\n")
        subtask_specs, worker_map = build_task_plan(input_data)
        result = run_orchestrator(
            goal=input_data["goal"],
            subtask_specs=subtask_specs,
            worker_map=worker_map,
            api_key=cfg.openai_api_key,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            max_retries=cfg.worker_max_retries,
        )

    # Display tier-by-tier trace
    print("--- Tier 2: Team Lead Results ---")
    for lr in result.lead_results:
        status = "SUCCESS" if lr.success else "FAILED"
        partial = " (partial)" if lr.partial else ""
        print(f"  Lead '{lr.lead_id}' [{lr.domain}]: {status}{partial} | {lr.tokens_used} tokens")
        for wr in lr.worker_results:
            w_status = "ok" if wr.success else f"FAILED: {wr.error_message}"
            print(f"    Worker '{wr.worker_id}': {w_status} | {wr.tokens_used} tokens | {wr.latency_ms:.0f}ms")

    print("\n--- Tier 1: Orchestrator Final Output ---")
    print(result.final_output)

    if result.warnings:
        print("\n⚠️  Warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    print(f"\n--- Stats ---")
    print(f"  Total tokens: {result.total_tokens_used}")
    print(f"  Total latency: {result.total_latency_ms:.0f}ms")
    print(f"  Overall success: {result.success}")

    print("\n[OK] Concept demonstrated: 3-tier hierarchy with typed contracts prevents context bleed")
    print("\nSee 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
