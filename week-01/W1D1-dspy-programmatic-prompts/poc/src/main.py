#!/usr/bin/env python3
"""
W1D1 — DSPy & Programmatic Prompts
====================================
Demonstrates: Signature-based prompt programming, ChainOfThought predictor,
              and BootstrapFewShot compilation (demo mode uses a mock LM).
Run:          python src/main.py
Run (demo):   DEMO_MODE=true python src/main.py
"""
import os
import sys

# Ensure emoji output works on Windows terminals (cp1252 → utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from dspy_core import run_demo_pipeline, run_live_pipeline


def _print_results(results: list[dict]) -> None:
    for i, r in enumerate(results, 1):
        rationale_preview = r["rationale"][:90] + "..." if len(r["rationale"]) > 90 else r["rationale"]
        print(f"Query {i}: {r['question']}")
        print(f"  Reasoning : {rationale_preview}")
        print(f"  Answer    : {r['answer']}")
        print(f"  Demos used: {r.get('num_demos', 0)} bootstrapped examples")
        print()


def main() -> None:
    config = Config()

    print("🚀 DSPy Programmatic Prompts Demo")
    print("=" * 38)

    if config.demo_mode:
        print("⚠️  Running in demo mode (no API key). Output is pre-computed.\n")
        results = run_demo_pipeline()
    else:
        print(f"🔑 Running live with model: {config.model}\n")
        results = run_live_pipeline(config)

    _print_results(results)

    print("✅ Concept demonstrated: DSPy separates program logic from prompt text,")
    print("   enabling automatic optimization via BootstrapFewShot teleprompter.")
    print()
    print("Key DSPy abstractions shown:")
    print("  • Signature       — typed I/O contract replacing raw prompt strings")
    print("  • ChainOfThought  — predictor that requires explicit reasoning steps")
    print("  • BootstrapFewShot — teleprompter that compiles optimal few-shot demos")


if __name__ == "__main__":
    main()
