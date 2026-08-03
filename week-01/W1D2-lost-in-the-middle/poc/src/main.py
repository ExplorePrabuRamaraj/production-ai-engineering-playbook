#!/usr/bin/env python3
"""
W1D2 — Lost in the Middle: Context Position Decay
==================================================
Demonstrates: How document position in a context window affects LLM
              attention, and how position-aware ordering mitigates it.
Run:           python src/main.py
Run (demo):    DEMO_MODE=true python src/main.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from lost_in_middle_core import (
    Document,
    naive_ordering,
    relevance_sorted_ordering,
    lost_in_middle_aware_ordering,
    compute_effective_scores,
    summarise_effectiveness,
)
from config import load_config

cfg = load_config()

# Demo documents: high-relevance docs placed at middle positions (positions 2-3)
# to illustrate the Lost-in-the-Middle problem before correction is applied.
SAMPLE_DOCUMENTS = [
    Document(id="doc_1", text="FAQ: How to reset your password via email link.", relevance_score=0.10),
    Document(id="doc_2", text="Shipping destinations and estimated delivery times by region.", relevance_score=0.15),
    Document(id="doc_3", text="Checkout fails on Safari 16+ due to CSP header blocking WebKit fetch.", relevance_score=0.92),
    Document(id="doc_4", text="Payment gateway timeout when cart total exceeds $500 on mobile.", relevance_score=0.88),
    Document(id="doc_5", text="Promo codes expire at midnight UTC, not at midnight local time.", relevance_score=0.20),
    Document(id="doc_6", text="Payment retry: 3 attempts before surfacing error to the user.", relevance_score=0.75),
]


def _print_strategy(title: str, scores: list, summary: dict) -> None:
    print(f"\n  {title}")
    print(f"  {'Pos':>3}  {'ID':<7}  {'Relevance':>9}  {'Attention':>9}  {'Effective':>9}")
    print(f"  {'---':>3}  {'-------':<7}  {'---------':>9}  {'---------':>9}  {'---------':>9}")
    for r in scores:
        flag = "  ← dead zone" if r["relevance_score"] >= 0.75 and r["attention_weight"] < 0.7 else ""
        print(
            f"  {r['position']:>3}  {r['id']:<7}  "
            f"{r['relevance_score']:>9.2f}  {r['attention_weight']:>9.4f}  {r['effective_score']:>9.4f}{flag}"
        )
    print(f"  → mean={summary['mean_effective_score']:.4f}  min={summary['min_effective_score']:.4f}")


def run_demo() -> None:
    print("🚀 Lost in the Middle Demo")
    print("=" * 46)
    print(f"\nInput: {len(SAMPLE_DOCUMENTS)} docs | Query: 'Why does checkout fail on mobile Safari?'\n")

    runs = [
        ("Strategy 1: Naive (retrieval order)", naive_ordering(SAMPLE_DOCUMENTS)),
        ("Strategy 2: Relevance-sorted (best first)", relevance_sorted_ordering(SAMPLE_DOCUMENTS)),
        ("Strategy 3: LiTM-aware (best at edges)", lost_in_middle_aware_ordering(SAMPLE_DOCUMENTS)),
    ]

    summaries = []
    for title, ordered in runs:
        scores = compute_effective_scores(ordered)
        s = summarise_effectiveness(scores)
        summaries.append(s)
        _print_strategy(title, scores, s)

    naive_mean = summaries[0]["mean_effective_score"]
    litm_mean = summaries[2]["mean_effective_score"]
    pct = round((litm_mean - naive_mean) / naive_mean * 100, 1)

    print(
        f"\n📊 Naive={naive_mean:.4f} | "
        f"Sorted={summaries[1]['mean_effective_score']:.4f} | "
        f"LiTM={litm_mean:.4f}"
    )
    print(f"\n✅ Concept demonstrated: LiTM-aware ordering improves mean effective score by {pct}%.")
    print("   High-relevance docs now occupy positions 0 and N-1 where attention peaks.")


def run_live() -> None:
    """Live mode: extend this to call OpenAI with each ordering strategy."""
    print("ℹ️  Live mode not implemented. Set DEMO_MODE=true or add OPENAI_API_KEY.")
    run_demo()


def main() -> None:
    if cfg.demo_mode:
        print("⚠️  Running in demo mode (no API key required).")
    run_demo() if cfg.demo_mode else run_live()


if __name__ == "__main__":
    main()
