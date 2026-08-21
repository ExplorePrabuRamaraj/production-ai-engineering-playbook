#!/usr/bin/env python3
"""
W3D5 — Dynamic Skill Selection
================================
Demonstrates: Routing agent queries to a relevant skill subset using
embedding-based cosine similarity, cutting prompt token count by 60-80%.

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
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true" or not OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Bootstrap: add src/ to path for imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from skill_selection_core import (
    EmbeddingRouter,
    Skill,
    SkillInjector,
    SkillRegistry,
)

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def build_demo_registry(cfg) -> SkillRegistry:
    """Register a realistic 8-skill IT support toolset for demonstration."""
    registry = SkillRegistry(demo_mode=cfg.demo_mode)

    skills = [
        ("get_invoice",         "Retrieve billing invoices and charge history for a customer account",
         {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]},
         {"billing", "admin"}),
        ("process_refund",      "Process a billing refund for a customer",
         {"type": "object", "properties": {"account_id": {"type": "string"}, "amount": {"type": "number"}}, "required": ["account_id", "amount"]},
         {"billing", "admin"}),
        ("check_network_speed", "Run a network speed test for a customer connection",
         {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
         set()),
        ("run_ping_diagnostic", "Run ping and traceroute diagnostics to identify network latency issues",
         {"type": "object", "properties": {"ip_address": {"type": "string"}}, "required": ["ip_address"]},
         set()),
        ("reset_password",      "Reset the account password for an authenticated user",
         {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
         {"user", "admin"}),
        ("provision_access",    "Grant or revoke access permissions for a user account",
         {"type": "object", "properties": {"user_id": {"type": "string"}, "resource": {"type": "string"}, "action": {"type": "string", "enum": ["grant", "revoke"]}}, "required": ["user_id", "resource", "action"]},
         {"admin"}),
        ("create_it_ticket",    "Create a support ticket for IT infrastructure issues",
         {"type": "object", "properties": {"title": {"type": "string"}, "description": {"type": "string"}}, "required": ["title", "description"]},
         set()),
        ("general_response",    "Provide a general response when no specific tool is applicable",
         {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
         set()),
    ]

    for name, desc, schema, roles in skills:
        registry.register(name=name, description=desc, schema=schema, required_roles=roles)

    return registry


def run_demo(input_data: dict) -> dict:
    """Run the skill selection demonstration using pre-computed mock embeddings."""
    print("\n--- Running in DEMO MODE (no API key — embeddings are pre-computed) ---\n")

    cfg = load_config()
    registry = build_demo_registry(cfg)
    router = EmbeddingRouter(
        registry=registry,
        top_k=cfg.top_k,
        similarity_threshold=cfg.similarity_threshold,
        fallback_skills=cfg.fallback_skills,
        demo_mode=True,
    )
    injector = SkillInjector()

    results = []
    for scenario in input_data.get("scenarios", []):
        query = scenario["query"]
        roles = set(scenario.get("user_roles", []))
        selection = router.select(query=query, user_roles=roles)
        tool_block = injector.build_tool_block(selection.selected_skills)

        results.append({
            "query": query,
            "user_roles": list(roles),
            "selected_skills": [s.name for s in selection.selected_skills],
            "skill_count_injected": len(selection.selected_skills),
            "total_skills_registered": len(registry),
            "used_fallback": selection.used_fallback,
            "top_scores": {
                k: round(v, 3)
                for k, v in sorted(selection.scores.items(), key=lambda x: -x[1])[:3]
            },
        })

    return {
        "demo_mode": True,
        "model": "demo",
        "scenarios": results,
        "concept": "Dynamic Skill Selection — top-k cosine similarity routing",
    }


def run_live(input_data: dict) -> dict:
    """Run with real OpenAI embedding calls for query routing."""
    try:
        from openai import OpenAI  # noqa: F401 — validates package is installed
    except ImportError:
        print("openai package not installed. Run: pip install -r requirements.txt")
        raise

    cfg = load_config()
    registry = build_demo_registry(cfg)
    # In live mode, real embeddings would be pre-computed at startup via
    # client.embeddings.create() for each skill description, then stored.
    # For this PoC, we re-use the demo registry and swap to live query embedding.
    router = EmbeddingRouter(
        registry=registry,
        top_k=cfg.top_k,
        similarity_threshold=cfg.similarity_threshold,
        fallback_skills=cfg.fallback_skills,
        demo_mode=False,
        api_key=cfg.openai_api_key,
        embedding_model=cfg.embedding_model,
    )
    injector = SkillInjector()

    results = []
    for scenario in input_data.get("scenarios", []):
        query = scenario["query"]
        roles = set(scenario.get("user_roles", []))
        selection = router.select(query=query, user_roles=roles)
        tool_block = injector.build_tool_block(selection.selected_skills)

        results.append({
            "query": query,
            "user_roles": list(roles),
            "selected_skills": [s.name for s in selection.selected_skills],
            "skill_count_injected": len(selection.selected_skills),
            "total_skills_registered": len(registry),
            "used_fallback": selection.used_fallback,
        })

    return {
        "demo_mode": False,
        "model": cfg.embedding_model,
        "scenarios": results,
        "concept": "Dynamic Skill Selection — live embedding routing",
    }


def load_sample_input() -> dict:
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {
        "scenarios": [
            {"query": "Why is my internet so slow today?", "user_roles": ["user"]},
            {"query": "I need a refund on my last invoice", "user_roles": ["billing"]},
        ]
    }


def main():
    print("\nW3D5 Dynamic Skill Selection Demo")
    print("=" * 50)

    input_data = load_sample_input()
    result = run_demo(input_data) if DEMO_MODE else run_live(input_data)

    for scenario in result["scenarios"]:
        print(f"\nQuery:    {scenario['query']}")
        print(f"Roles:    {scenario['user_roles']}")
        print(f"Selected: {scenario['selected_skills']} ({scenario['skill_count_injected']} of {scenario['total_skills_registered']} tools)")
        if scenario.get("top_scores"):
            print(f"Scores:   {scenario['top_scores']}")
        if scenario["used_fallback"]:
            print("          [fallback activated — low similarity across all skills]")

    print("\n" + "=" * 50)
    print("Concept demonstrated: Routing injects only relevant skills per query,")
    print(f"reducing tool definitions from {result['scenarios'][0]['total_skills_registered']} to ~{result['scenarios'][0]['skill_count_injected']} per turn.")
    print("\nSee 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
