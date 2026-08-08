#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W1D7 -- LLM-as-a-Judge Evals
==============================
Demonstrates: Using a second LLM to score the output of a first LLM
              against a calibrated rubric with structured JSON output.

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
from judge_core import JudgeVerdict, build_judge_prompt, parse_verdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "user_prompt": "What is your return policy for opened software?",
        "candidate_response": "You can return any item within 30 days for a full refund.",
        "reference": "Opened software is non-refundable. All other items may be returned within 30 days.",
    }


# ---------------------------------------------------------------------------
# Demo mode — pre-computed verdict, no API key required
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> dict:
    """Return a pre-computed judge verdict that mirrors live mode output."""
    print("\n[DEMO MODE] Running with pre-computed output (no API call made)\n")

    # Build the prompt to show the user what would be sent to the judge
    messages = build_judge_prompt(
        user_prompt=input_data["user_prompt"],
        candidate_response=input_data["candidate_response"],
        rubric_version=cfg.rubric_version,
        reference=input_data.get("reference", ""),
    )
    print("Judge prompt (system + user messages) constructed.")
    print(f"Rubric version: {cfg.rubric_version}")
    print(f"Criteria: relevance, accuracy, completeness\n")

    # Pre-computed verdict for this sample (accuracy=1 because response contradicts reference)
    demo_verdict = {
        "criteria": {
            "relevance": {"score": 3, "rationale": ""},
            "accuracy": {"score": 1, "rationale": "Response states all items are refundable but reference explicitly excludes opened software."},
            "completeness": {"score": 2, "rationale": "Response omits the opened software exclusion that is critical for this query."},
        },
        "overall": "fail",
        "confidence": "high",
    }

    verdict = parse_verdict(json.dumps(demo_verdict), rubric_version=cfg.rubric_version)
    return {
        "verdict": demo_verdict,
        "needs_human_review": verdict.needs_human_review(),
        "summary": verdict.summary(),
        "model": "demo",
        "parse_attempts": 1,
    }


# ---------------------------------------------------------------------------
# Live mode — real API call
# ---------------------------------------------------------------------------

def run_live(input_data: dict) -> dict:
    """Execute the judge evaluation using a real OpenAI API call."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install -r requirements.txt")
        raise

    client = OpenAI(api_key=cfg.openai_api_key)
    messages = build_judge_prompt(
        user_prompt=input_data["user_prompt"],
        candidate_response=input_data["candidate_response"],
        rubric_version=cfg.rubric_version,
        reference=input_data.get("reference", ""),
    )

    raw: str = ""
    parse_attempts = 0

    for attempt in range(1, cfg.max_judge_retries + 1):
        parse_attempts = attempt
        response = client.chat.completions.create(
            model=cfg.judge_model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""

        try:
            verdict = parse_verdict(raw, rubric_version=cfg.rubric_version, parse_attempts=attempt)
            return {
                "verdict": json.loads(raw),
                "needs_human_review": verdict.needs_human_review(),
                "summary": verdict.summary(),
                "model": cfg.judge_model,
                "parse_attempts": attempt,
            }
        except ValueError:
            if attempt < cfg.max_judge_retries:
                # Add correction hint for the next attempt
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Invalid JSON schema. Return only the JSON object."})

    raise ValueError(f"Judge failed to return valid JSON after {cfg.max_judge_retries} attempts.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nLLM-as-a-Judge Evals Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"User prompt:        {input_data['user_prompt']}")
    print(f"Candidate response: {input_data['candidate_response']}")
    if input_data.get("reference"):
        print(f"Reference material: {input_data['reference']}\n")

    if cfg.demo_mode:
        result = run_demo(input_data)
    else:
        print(f"Judge model: {cfg.judge_model} | Rubric: {cfg.rubric_version}")
        result = run_live(input_data)

    print("\n--- Judge Verdict ---")
    print(result["summary"])
    print(f"\nRoutes to human review: {result['needs_human_review']}")
    print(f"Parse attempts: {result['parse_attempts']}")
    print("\n[OK] Concept demonstrated: A second LLM evaluates response quality against a calibrated rubric.")
    print("\nSee 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
