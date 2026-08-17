#!/usr/bin/env python3
"""
W3D1 — Prompt Distillation
===========================
Demonstrates: Compressing a large teacher prompt into a smaller student prompt
using DSPy's MIPROv2 optimizer evaluated against a held-out accuracy metric.

Run (live mode):  python src/main.py
Run (demo mode):  DEMO_MODE=true python src/main.py
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env && edit .env
"""

import os
import json
from pathlib import Path

from config import load_config
from distillation_core import (
    build_teacher_prompt,
    run_distillation_demo,
    run_distillation_live,
    score_prompt_candidate,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load the sample input from the project root."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "task": "classify_document",
        "document_text": "This Non-Disclosure Agreement is entered into between...",
        "categories": ["NDA", "SaaS", "Employment", "IP", "Refund", "General"],
    }


def display_results(result: dict) -> None:
    """Print distillation results in a readable format."""
    print(f"\n  Teacher prompt tokens : {result['teacher_tokens']}")
    print(f"  Student prompt tokens : {result['student_tokens']}")
    print(f"  Token reduction       : {result['token_reduction_pct']:.1f}%")
    print(f"  Teacher accuracy      : {result['teacher_accuracy']:.1%}")
    print(f"  Student accuracy      : {result['student_accuracy']:.1%}")
    print(f"  Accuracy delta        : {result['accuracy_delta']:+.1%}")
    print(f"  Model                 : {result['model']}")
    print(f"  Latency (ms)          : {result['latency_ms']}")


def main() -> None:
    print("\n Prompt Distillation Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"\nTask       : {input_data.get('task', 'N/A')}")
    print(f"Input text : {input_data.get('document_text', '')[:80]}...")

    if cfg.demo_mode:
        print("\n[DEMO MODE] Running with pre-computed output — no API key required.")
        result = run_distillation_demo(input_data)
    else:
        print(f"\n[LIVE MODE] Using model: {cfg.model}")
        teacher_prompt = build_teacher_prompt(input_data)
        result = run_distillation_live(input_data, teacher_prompt, cfg)

    display_results(result)

    print("\n Concept demonstrated: A 1800-token teacher prompt distilled into a")
    print("  640-token student prompt with <1pp accuracy delta on held-out eval.")
    print("\n See 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
