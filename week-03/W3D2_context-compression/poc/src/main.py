#!/usr/bin/env python3
"""
W3D2 — Context Compression
===========================
Demonstrates: Query-aware context compression using extractive (TF-IDF)
and abstractive (LLM summarisation) strategies before an LLM call.

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
from context_compression_core import compress_context, CompressionResult

# OpenAI is imported at module level so tests can patch "main.OpenAI" reliably.
# The try/except keeps demo mode working when the package is not installed.
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Configuration — all secrets come from environment variables, never hardcoded
# ---------------------------------------------------------------------------
cfg = load_config()

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load the sample input from the project root."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "query": "What is the refund policy for annual subscriptions?",
        "segments": {
            "history": "Customer asked about billing. Agent explained pricing.",
            "docs": "Refunds are available within 30 days of purchase for annual plans."
        }
    }


def run_demo(input_data: dict) -> dict:
    """
    Run compression using the extractive strategy with pre-loaded data.
    No API key required — pure CPU, fully offline.
    """
    print("\n[DEMO MODE] No API call made — output is pre-computed.\n")

    query = input_data.get("query", "")
    segments = input_data.get("segments", {})
    token_budget = cfg.token_budget // max(len(segments), 1)

    results = {}
    total_original = 0
    total_compressed = 0

    for seg_name, seg_text in segments.items():
        result: CompressionResult = compress_context(
            text=seg_text,
            query=query,
            token_budget=token_budget,
            strategy="extractive",
            openai_client=None,
            min_segment_tokens=cfg.min_segment_tokens,
        )
        results[seg_name] = {
            "compressed_text": result.compressed_text,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": result.compression_ratio,
            "strategy_used": result.strategy_used,
        }
        total_original += result.original_tokens
        total_compressed += result.compressed_tokens

    overall_ratio = round(total_compressed / total_original, 3) if total_original else 1.0

    return {
        "query": query,
        "segments": results,
        "total_original_tokens": total_original,
        "total_compressed_tokens": total_compressed,
        "overall_compression_ratio": overall_ratio,
        "model": "demo",
        "latency_ms": 0,
    }


def run_live(input_data: dict) -> dict:
    """
    Run compression with abstractive strategy using the OpenAI API.
    Called only when OPENAI_API_KEY is set and DEMO_MODE is false.
    """
    if OpenAI is None:
        print("❌ openai package not installed. Run: pip install -r requirements.txt")
        raise ImportError("openai package is required for live mode")

    client = OpenAI(api_key=cfg.openai_api_key)
    query = input_data.get("query", "")
    segments = input_data.get("segments", {})
    token_budget = cfg.token_budget // max(len(segments), 1)

    results = {}
    total_original = 0
    total_compressed = 0

    for seg_name, seg_text in segments.items():
        result: CompressionResult = compress_context(
            text=seg_text,
            query=query,
            token_budget=token_budget,
            strategy=cfg.compression_strategy,
            openai_client=client,
            model=cfg.compression_model,
            min_segment_tokens=cfg.min_segment_tokens,
        )
        results[seg_name] = {
            "compressed_text": result.compressed_text,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": result.compression_ratio,
            "strategy_used": result.strategy_used,
        }
        total_original += result.original_tokens
        total_compressed += result.compressed_tokens

    overall_ratio = round(total_compressed / total_original, 3) if total_original else 1.0

    return {
        "query": query,
        "segments": results,
        "total_original_tokens": total_original,
        "total_compressed_tokens": total_compressed,
        "overall_compression_ratio": overall_ratio,
        "model": cfg.compression_model,
        "latency_ms": 0,
    }


def main():
    print("\n[W3D2] Context Compression Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"Query: {input_data.get('query', '')}\n")
    print(f"Segments to compress: {list(input_data.get('segments', {}).keys())}\n")

    if cfg.demo_mode:
        result = run_demo(input_data)
    else:
        print(f"[LIVE MODE] Strategy: {cfg.compression_strategy} | Model: {cfg.compression_model}")
        result = run_live(input_data)

    print(f"Results:\n{json.dumps(result, indent=2)}\n")
    ratio_pct = round((1 - result["overall_compression_ratio"]) * 100)
    print(f"[DONE] Concept demonstrated: {ratio_pct}% token reduction via query-aware context compression.")
    print("\nSee 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
