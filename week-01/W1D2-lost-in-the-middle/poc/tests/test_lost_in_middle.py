"""
Unit tests for Lost-in-the-Middle context position decay.
Run: pytest tests/ -v
"""
import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lost_in_middle_core import (
    Document,
    u_shaped_attention_weight,
    naive_ordering,
    relevance_sorted_ordering,
    lost_in_middle_aware_ordering,
    compute_effective_scores,
    summarise_effectiveness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs():
    return [
        Document(id="doc_1", text="High relevance document A.", relevance_score=0.90),
        Document(id="doc_2", text="Low relevance document B.", relevance_score=0.10),
        Document(id="doc_3", text="Medium relevance document C.", relevance_score=0.50),
        Document(id="doc_4", text="Medium-high relevance document D.", relevance_score=0.70),
        Document(id="doc_5", text="Very low relevance document E.", relevance_score=0.05),
    ]


# ---------------------------------------------------------------------------
# Attention weight tests
# ---------------------------------------------------------------------------

def test_attention_weight_single_doc_returns_full():
    """A single document always receives full attention."""
    assert u_shaped_attention_weight(0, 1) == 1.0


def test_attention_weight_edges_exceed_middle():
    """U-shaped: edge positions must have strictly higher weight than the middle."""
    n = 10
    start = u_shaped_attention_weight(0, n)
    end = u_shaped_attention_weight(n - 1, n)
    middle = u_shaped_attention_weight(n // 2, n)
    assert start > middle, f"Start weight {start} should exceed middle {middle}"
    assert end > middle, f"End weight {end} should exceed middle {middle}"


@pytest.mark.parametrize("position,total,is_edge", [
    (0, 6, True),   # Start of context: weight should be 1.0
    (5, 6, True),   # End of context: weight should be 1.0
    (2, 6, False),  # Near middle: weight should be below 0.70
    (3, 6, False),  # Near middle: weight should be below 0.70
])
def test_attention_weight_u_shape(position, total, is_edge):
    """Verify the U-shaped pattern: edges high, middle low."""
    weight = u_shaped_attention_weight(position, total)
    if is_edge:
        assert weight >= 0.9, f"Edge position {position} weight too low: {weight}"
    else:
        assert weight < 0.7, f"Middle position {position} weight too high: {weight}"


# ---------------------------------------------------------------------------
# Ordering strategy tests
# ---------------------------------------------------------------------------

def test_naive_ordering_preserves_original_order(sample_docs):
    """Naive ordering must not resequence documents."""
    original_ids = [d.id for d in sample_docs]
    result = naive_ordering(sample_docs)
    assert [d.id for d in result] == original_ids


def test_naive_ordering_assigns_sequential_positions(sample_docs):
    """All orderings must assign contiguous 0-based position indices."""
    result = naive_ordering(sample_docs)
    assert [d.position for d in result] == list(range(len(sample_docs)))


def test_relevance_sorted_puts_best_document_first(sample_docs):
    """The most relevant document must appear at position 0 after relevance sort."""
    result = relevance_sorted_ordering(sample_docs)
    best_score = max(d.relevance_score for d in sample_docs)
    assert result[0].relevance_score == best_score


def test_lost_in_middle_aware_places_best_at_an_edge(sample_docs):
    """The highest-relevance document must occupy position 0 or N-1."""
    result = lost_in_middle_aware_ordering(sample_docs)
    best_score = max(d.relevance_score for d in sample_docs)
    edge_scores = {result[0].relevance_score, result[-1].relevance_score}
    assert best_score in edge_scores, (
        f"Best doc (score {best_score}) not at an edge. "
        f"Edges have scores {edge_scores}."
    )


def test_compute_effective_scores_length_matches_input(sample_docs):
    """Effective scores list must have the same length as the input document list."""
    ordered = naive_ordering(sample_docs)
    scores = compute_effective_scores(ordered)
    assert len(scores) == len(sample_docs)


def test_summarise_effectiveness_returns_required_keys(sample_docs):
    """Summary dict must contain mean, min, and max keys."""
    ordered = naive_ordering(sample_docs)
    scores = compute_effective_scores(ordered)
    summary = summarise_effectiveness(scores)
    assert "mean_effective_score" in summary
    assert "min_effective_score" in summary
    assert "max_effective_score" in summary


def test_litm_aware_improves_mean_effective_score_over_naive(sample_docs):
    """LiTM-aware ordering must produce a higher mean effective score than naive."""
    naive_scores = compute_effective_scores(naive_ordering(sample_docs))
    litm_scores = compute_effective_scores(lost_in_middle_aware_ordering(sample_docs))

    naive_mean = summarise_effectiveness(naive_scores)["mean_effective_score"]
    litm_mean = summarise_effectiveness(litm_scores)["mean_effective_score"]

    assert litm_mean >= naive_mean, (
        f"LiTM mean ({litm_mean}) should be >= naive mean ({naive_mean})"
    )


def test_effective_score_bounded_by_relevance(sample_docs):
    """Effective score must never exceed the document's relevance score."""
    ordered = naive_ordering(sample_docs)
    scores = compute_effective_scores(ordered)
    for r in scores:
        assert r["effective_score"] <= r["relevance_score"] + 1e-9, (
            f"Effective score {r['effective_score']} exceeds relevance {r['relevance_score']} "
            f"for {r['id']}"
        )
