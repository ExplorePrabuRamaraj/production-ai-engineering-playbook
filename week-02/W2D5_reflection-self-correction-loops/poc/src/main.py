#!/usr/bin/env python3
"""
W2D5 - Reflection & Self-Correction Loops
==========================================
Demonstrates: A three-node Generate->Critique->Revise loop that checks
              its own output against a rubric before returning it.

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
from reflection_core import (
    CriterionResult,
    CritiqueResult,
    ReflectionState,
    DEFAULT_RUBRIC,
    run_reflection_loop,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load task input from sample_input.json, or use an inline fallback."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text(encoding="utf-8"))
    return {
        "task": (
            "Summarise the risks of deploying LLMs in production in exactly "
            "3 bullet points, each under 20 words, citing at least one source."
        )
    }


# ---------------------------------------------------------------------------
# Demo mode - pre-computed output, no API call required
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    """
    Return pre-computed output that mirrors what live mode produces.
    Allows the PoC to demonstrate the concept completely offline.
    """
    print("WARNING: Running in DEMO MODE -- output is pre-computed (no API call made)\n")

    # Simulate a two-iteration correction scenario
    demo_state = {
        "task": input_data.get("task", ""),
        "final_draft": (
            "- Hallucination risk: LLMs confidently produce false outputs; "
            "validate every factual claim before surfacing to users (Anthropic, 2024).\n"
            "- Prompt injection: adversarial inputs can hijack agent behaviour; "
            "sanitise all user-controlled content before interpolation.\n"
            "- Cost unpredictability: unbounded context growth spikes token spend; "
            "enforce input and output length limits at the gateway."
        ),
        "iterations_used": 2,
        "all_criteria_passed": True,
        "exited_at_cap": False,
        "iteration_log": [
            {
                "iteration": 1,
                "critique_summary": "Iteration 1: FAIL (2/3 criteria passed)",
                "failing_criteria": ["completeness"],
            },
            {
                "iteration": 2,
                "critique_summary": "Iteration 2: PASS (3/3 criteria passed)",
                "failing_criteria": [],
            },
        ],
        "model": "demo",
        "latency_ms": 0,
    }
    return demo_state


# ---------------------------------------------------------------------------
# Live mode - real API calls through the reflection loop
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    """
    Execute the full reflection loop using real OpenAI API calls.
    Only called when OPENAI_API_KEY is set and DEMO_MODE is false.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install -r requirements.txt")
        raise

    import time
    client = OpenAI(api_key=cfg.openai_api_key)
    task = input_data.get("task", "")

    start = time.monotonic()
    state: ReflectionState = run_reflection_loop(
        task=task,
        client=client,
        model=cfg.model,
        critic_model=cfg.critic_model,
        max_tokens=cfg.max_tokens,
        max_iterations=cfg.max_iterations,
        rubric=DEFAULT_RUBRIC,
    )
    elapsed_ms = int((time.monotonic() - start) * 1000)

    return {
        "task": task,
        "final_draft": state.draft,
        "iterations_used": state.iteration,
        "all_criteria_passed": state.critique.all_passed if state.critique else False,
        "exited_at_cap": state.exited_at_cap,
        "iteration_log": state.history,
        "model": cfg.model,
        "latency_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    """Print a structured, readable summary of the reflection loop result."""
    status = "PASS" if result["all_criteria_passed"] else "PARTIAL PASS (hit iteration cap)"
    cap_note = " [exited at max_iterations]" if result.get("exited_at_cap") else ""

    print(f"Task:       {result['task'][:80]}...")
    print(f"Model:      {result['model']}")
    print(f"Status:     {status}{cap_note}")
    print(f"Iterations: {result['iterations_used']}")
    print(f"Latency:    {result['latency_ms']} ms\n")

    print("--- Iteration Log ---")
    for entry in result.get("iteration_log", []):
        summary = entry.get("critique_summary", "")
        failing = entry.get("failing_criteria", [])
        print(f"  {summary}")
        if failing:
            print(f"    Failing criteria: {', '.join(str(f) for f in failing)}")

    print("\n--- Final Draft ---")
    print(result["final_draft"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nReflection & Self-Correction Loops Demo")
    print("=" * 50)

    input_data = load_sample_input()

    if cfg.demo_mode:
        result = run_demo(input_data)
    else:
        print(f"Using model: {cfg.model}  |  max_iterations: {cfg.max_iterations}\n")
        result = run_live(input_data)

    print_result(result)
    print("\nConcept demonstrated: Generate -> Critique -> Revise loop with hard termination cap.")
    print("See 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
