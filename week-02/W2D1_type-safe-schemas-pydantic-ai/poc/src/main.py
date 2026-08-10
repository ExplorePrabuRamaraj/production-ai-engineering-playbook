#!/usr/bin/env python3
"""
W2D1 — Type-Safe Schemas with Pydantic AI
==========================================
Demonstrates: Schema-enforced LLM output with automatic retry on validation failure.

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

# ---------------------------------------------------------------------------
# Configuration — all secrets from environment variables, never hardcoded
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true" or not OPENAI_API_KEY

# Add src/ to path so imports work whether invoked from project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from pydantic_schemas_core import (
    ReviewAnalysis,
    DEMO_REVIEW_OUTPUT,
    DEMO_TICKET_OUTPUT,
    SupportTicketTriage,
)

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    """Load sample input; fall back to inline fixture if file is missing."""
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "review": "Battery lasts 3 full days. Build feels premium. Great value at this price.",
        "ticket": "I was charged twice for order #88231. Please refund the duplicate charge ASAP.",
    }


def run_demo(input_data: dict) -> dict:
    """Return pre-computed output that mirrors what live mode produces.

    Pre-computing output here means the PoC runs completely offline
    and produces the same structured types as the live path — tests
    can validate the schema without any network calls.
    """
    print("\n  Running in DEMO MODE — output is pre-computed (no API call made)\n")
    return {
        "review_analysis": DEMO_REVIEW_OUTPUT.model_dump(),
        "ticket_triage": DEMO_TICKET_OUTPUT.model_dump(),
        "model": "demo",
        "retry_count": 0,
    }


def run_live(input_data: dict) -> dict:
    """Run schema-enforced extractions using Pydantic AI agents.

    Two agents are created — one per schema — to demonstrate that the same
    validation loop pattern applies to any Pydantic BaseModel subclass.
    """
    try:
        from pydantic_ai import Agent  # type: ignore[import]
    except ImportError:
        print("pydantic-ai not installed. Run: pip install -r requirements.txt")
        raise

    cfg = load_config()

    # Each agent is bound to one output schema. Pydantic AI injects the JSON
    # Schema into the system prompt and handles retry on validation failure.
    review_agent = Agent(
        model=cfg.model,
        result_type=ReviewAnalysis,
        system_prompt=(
            "You are a product review analyser. "
            "Extract the sentiment, confidence score (0.0-1.0), key topics, "
            "and a brief summary (max 150 chars) from the review text."
        ),
        retries=cfg.schema_retries,
    )

    ticket_agent = Agent(
        model=cfg.model,
        result_type=SupportTicketTriage,
        system_prompt=(
            "You are a customer support triage assistant. "
            "Extract the urgency level, department, whether a refund is involved, "
            "and a one-line summary (max 100 chars) from the support ticket."
        ),
        retries=cfg.schema_retries,
    )

    import asyncio

    async def _run() -> dict:
        review_result = await review_agent.run(input_data.get("review", ""))
        ticket_result = await ticket_agent.run(input_data.get("ticket", ""))
        return {
            "review_analysis": review_result.data.model_dump(),
            "ticket_triage": ticket_result.data.model_dump(),
            "model": cfg.model,
            "retry_count": 0,  # pydantic-ai handles retries internally
        }

    return asyncio.run(_run())


def main() -> None:
    print("\nType-Safe Schemas with Pydantic AI — W2D1 Demo")
    print("=" * 50)

    input_data = load_sample_input()
    print(f"Input (review): {input_data.get('review', '')[:80]}...")
    print(f"Input (ticket): {input_data.get('ticket', '')[:80]}...\n")

    result = run_demo(input_data) if DEMO_MODE else run_live(input_data)

    print("Review Analysis:")
    print(json.dumps(result["review_analysis"], indent=2))
    print("\nTicket Triage:")
    print(json.dumps(result["ticket_triage"], indent=2))
    print(f"\nModel: {result['model']}")
    print("\nConcept demonstrated: LLM output validated against Pydantic schemas")
    print("  with field-level validators and automatic retry on failure.")
    print("\nSee 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
