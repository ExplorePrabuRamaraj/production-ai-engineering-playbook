#!/usr/bin/env python3
"""
W2D2 — KV Caching & Token Trimming
=====================================
Demonstrates: Client-side token budget enforcement (sliding window eviction +
summary compression) as the client-side complement to server-side KV caching.

Run (live mode):  python src/main.py
Run (demo mode):  DEMO_MODE=true python src/main.py
Run tests:        pytest tests/ -v

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env && edit .env
"""

import json
import os
import time
from pathlib import Path

# Top-level import so @patch("main.OpenAI") works in tests.
# Graceful fallback when the package is not installed (demo mode only).
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Configuration — all secrets via environment variables, never hardcoded
# ---------------------------------------------------------------------------
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
MODEL              = os.getenv("MODEL", "gpt-4o-mini")
DEMO_MODE          = os.getenv("DEMO_MODE", "false").lower() == "true" or not OPENAI_API_KEY
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"

# ---------------------------------------------------------------------------
# Load sample input
# ---------------------------------------------------------------------------

def load_sample_input() -> dict:
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text(encoding="utf-8"))
    return {
        "system_prompt": "You are a helpful assistant.",
        "conversation": [
            {"role": "user",      "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user",      "content": "What is its population?"},
        ],
    }

# ---------------------------------------------------------------------------
# Demo mode — pre-computed output for offline demonstration
# ---------------------------------------------------------------------------

def run_demo(input_data: dict, budget: int = None) -> dict:
    """
    Return pre-computed trimming results without making any API call.
    Accepts an explicit budget to allow tests to override MAX_CONTEXT_TOKENS
    without mutating module-level state.
    """
    print("\n[DEMO MODE] Output is pre-computed (no API call made)\n")

    from kv_caching_core import prepare_context, count_messages_tokens

    effective_budget = budget if budget is not None else MAX_CONTEXT_TOKENS

    system_prompt = input_data.get("system_prompt", "You are a helpful assistant.")
    conversation  = input_data.get("conversation", [])

    messages        = [{"role": "system", "content": system_prompt}] + conversation
    original_tokens = count_messages_tokens(messages)

    result = prepare_context(
        messages=messages,
        budget=effective_budget,
        compression_threshold=0.5,
    )

    return {
        "original_tokens":  original_tokens,
        "final_tokens":     result["final_tokens"],
        "eviction_ratio":   round(result["eviction_ratio"], 3),
        "summary_injected": result["summary_injected"],
        "messages_before":  len(messages),
        "messages_after":   len(result["messages"]),
        "model":            "demo",
        "latency_ms":       0,
        "cache_note": (
            "In live mode, the system message would carry cache_control headers "
            "to enable server-side KV cache reuse on Anthropic/OpenAI."
        ),
    }

# ---------------------------------------------------------------------------
# Live mode — real OpenAI API call
# ---------------------------------------------------------------------------

def run_live(input_data: dict, budget: int = None) -> dict:
    """Trim the context to budget then make a real OpenAI API call."""
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install -r requirements.txt")

    from kv_caching_core import prepare_context, count_messages_tokens

    effective_budget = budget if budget is not None else MAX_CONTEXT_TOKENS

    system_prompt   = input_data.get("system_prompt", "You are a helpful assistant.")
    conversation    = input_data.get("conversation", [])
    messages        = [{"role": "system", "content": system_prompt}] + conversation
    original_tokens = count_messages_tokens(messages, MODEL)

    context = prepare_context(
        messages=messages,
        budget=effective_budget,
        compression_threshold=0.5,
        model=MODEL,
    )

    client   = OpenAI(api_key=OPENAI_API_KEY)
    start    = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=context["messages"],
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "500")),
    )
    latency_ms = int((time.time() - start) * 1000)

    return {
        "original_tokens":  original_tokens,
        "final_tokens":     context["final_tokens"],
        "eviction_ratio":   round(context["eviction_ratio"], 3),
        "summary_injected": context["summary_injected"],
        "messages_before":  len(messages),
        "messages_after":   len(context["messages"]),
        "llm_response":     response.choices[0].message.content,
        "tokens_used":      response.usage.total_tokens,
        "model":            response.model,
        "latency_ms":       latency_ms,
    }

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\nKV Caching & Token Trimming Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"Input conversation turns : {len(input_data.get('conversation', []))}")
    print(f"Token budget             : {MAX_CONTEXT_TOKENS} tokens")
    print(f"Mode                     : {'DEMO' if DEMO_MODE else 'LIVE'}\n")

    result = run_demo(input_data) if DEMO_MODE else run_live(input_data)

    print(f"Tokens before trimming   : {result['original_tokens']}")
    print(f"Tokens after trimming    : {result['final_tokens']}")
    print(f"Eviction ratio           : {result['eviction_ratio']:.1%}")
    print(f"Summary injected         : {result['summary_injected']}")
    print(f"Messages before / after  : {result['messages_before']} / {result['messages_after']}")

    if "llm_response" in result:
        print(f"\nLLM response             : {result['llm_response'][:200]}")

    if "cache_note" in result:
        print(f"\nCache note: {result['cache_note']}")

    print(f"\nLatency                  : {result['latency_ms']} ms")
    print("\n[OK] Concept demonstrated: token budget enforcement keeps context within limits")
    print("     See 02_technical-doc/technical-document.md for the full deep dive.")

    return result


if __name__ == "__main__":
    main()
