"""
W2D1 — Type-Safe Schemas with Pydantic AI — Unit Tests
=======================================================
Run: pytest tests/ -v

All external API calls are mocked so tests pass completely offline.
Tests cover: demo mode, core schema validation, field validators,
live mode (mocked), and sample file integrity.
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic_schemas_core import (
    DEMO_REVIEW_OUTPUT,
    DEMO_TICKET_OUTPUT,
    ReviewAnalysis,
    Sentiment,
    SupportTicketTriage,
    UrgencyLevel,
)
from main import load_sample_input, run_demo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    """Standard test input used across multiple tests."""
    return {
        "review": "Battery lasts 3 full days. Build feels premium. Great value.",
        "ticket": "I was charged twice for order #88231. Please refund immediately.",
    }


@pytest.fixture
def expected_review_keys():
    return {"sentiment", "confidence", "key_topics", "summary"}


@pytest.fixture
def expected_ticket_keys():
    return {"urgency", "department", "refund_involved", "one_line_summary"}


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo mode, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the offline demo mode — must pass without any API key."""

    def test_demo_returns_expected_top_level_keys(self, sample_input):
        """Demo output must contain all required top-level keys."""
        result = run_demo(sample_input)
        assert {"review_analysis", "ticket_triage", "model", "retry_count"}.issubset(result.keys())

    def test_demo_model_is_demo(self, sample_input):
        """Demo mode must indicate it is not a real model call."""
        result = run_demo(sample_input)
        assert result["model"] == "demo"

    def test_demo_review_analysis_has_correct_schema(self, sample_input, expected_review_keys):
        """Demo review_analysis must contain all required schema fields."""
        result = run_demo(sample_input)
        assert expected_review_keys.issubset(result["review_analysis"].keys())

    def test_demo_ticket_triage_has_correct_schema(self, sample_input, expected_ticket_keys):
        """Demo ticket_triage must contain all required schema fields."""
        result = run_demo(sample_input)
        assert expected_ticket_keys.issubset(result["ticket_triage"].keys())

    def test_demo_sentiment_is_valid_enum_value(self, sample_input):
        """Demo sentiment must be one of the valid Sentiment enum values."""
        result = run_demo(sample_input)
        valid_sentiments = {s.value for s in Sentiment}
        assert result["review_analysis"]["sentiment"] in valid_sentiments

    def test_demo_urgency_is_valid_enum_value(self, sample_input):
        """Demo urgency must be one of the valid UrgencyLevel enum values."""
        result = run_demo(sample_input)
        valid_levels = {u.value for u in UrgencyLevel}
        assert result["ticket_triage"]["urgency"] in valid_levels


# ---------------------------------------------------------------------------
# TestCoreConcept — Pydantic schema validation logic (pure, no API)
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for core schema validation — no LLM, pure Pydantic logic."""

    def test_review_analysis_valid_construction(self):
        """ReviewAnalysis must accept a fully valid input."""
        review = ReviewAnalysis(
            sentiment=Sentiment.POSITIVE,
            confidence=0.85,
            key_topics=["battery", "design"],
            summary="Great product overall.",
        )
        assert review.sentiment == Sentiment.POSITIVE
        assert review.confidence == 0.85

    def test_confidence_validator_rejects_out_of_range(self):
        """confidence > 1.0 must raise ValueError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="confidence must be a decimal between"):
            ReviewAnalysis(
                sentiment=Sentiment.POSITIVE,
                confidence=1.5,
                key_topics=["test"],
                summary="Short summary.",
            )

    def test_summary_validator_rejects_over_150_chars(self):
        """summary exceeding 150 characters must raise ValueError."""
        from pydantic import ValidationError
        long_summary = "A" * 151
        with pytest.raises(ValidationError, match="summary must be 150 characters or fewer"):
            ReviewAnalysis(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.5,
                key_topics=["test"],
                summary=long_summary,
            )

    def test_extra_fields_rejected(self):
        """Extra fields not in the schema must be rejected (extra='forbid')."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ReviewAnalysis(
                sentiment=Sentiment.POSITIVE,
                confidence=0.9,
                key_topics=["test"],
                summary="Fine.",
                injected_field="malicious_value",  # not in schema
            )

    def test_confidence_rounds_to_3_decimal_places(self):
        """confidence validator must round to 3 decimal places."""
        review = ReviewAnalysis(
            sentiment=Sentiment.POSITIVE,
            confidence=0.92345678,
            key_topics=["test"],
            summary="Summary.",
        )
        assert review.confidence == 0.923

    @pytest.mark.parametrize("sentiment_value,expected", [
        ("positive", Sentiment.POSITIVE),
        ("negative", Sentiment.NEGATIVE),
        ("neutral", Sentiment.NEUTRAL),
    ])
    def test_sentiment_enum_accepts_valid_string_values(self, sentiment_value, expected):
        """All valid Sentiment string values must be accepted and coerced to enum."""
        review = ReviewAnalysis(
            sentiment=sentiment_value,
            confidence=0.8,
            key_topics=["topic"],
            summary="OK.",
        )
        assert review.sentiment == expected

    def test_ticket_triage_valid_construction(self):
        """SupportTicketTriage must accept a fully valid input."""
        ticket = SupportTicketTriage(
            urgency=UrgencyLevel.HIGH,
            department="billing",
            refund_involved=True,
            one_line_summary="Duplicate charge, requesting refund.",
        )
        assert ticket.refund_involved is True

    @pytest.mark.parametrize("urgency,department,refund,summary", [
        ("low", "tech-support", False, "Password reset request."),
        ("medium", "shipping", False, "Order delayed by 3 days."),
        ("high", "billing", True, "Double charge on account."),
    ])
    def test_ticket_triage_with_varied_inputs(self, urgency, department, refund, summary):
        """SupportTicketTriage must handle all valid urgency/department combinations."""
        ticket = SupportTicketTriage(
            urgency=urgency,
            department=department,
            refund_involved=refund,
            one_line_summary=summary,
        )
        assert ticket is not None


# ---------------------------------------------------------------------------
# TestLiveMode — live mode with mocked Pydantic AI agent
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with all external calls mocked."""

    def test_live_mode_calls_agent_run(self, sample_input):
        """Live mode must invoke the Pydantic AI agent."""
        mock_review_result = MagicMock()
        mock_review_result.data = DEMO_REVIEW_OUTPUT
        mock_ticket_result = MagicMock()
        mock_ticket_result.data = DEMO_TICKET_OUTPUT

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=[mock_review_result, mock_ticket_result])

        # Mock at the import site inside run_live (lazy import) so the test
        # passes whether or not pydantic_ai is installed in the environment.
        mock_pydantic_ai = MagicMock()
        mock_pydantic_ai.Agent.return_value = mock_agent

        with patch("main.DEMO_MODE", False), \
             patch.dict("sys.modules", {"pydantic_ai": mock_pydantic_ai}):
            from main import run_live
            result = run_live(sample_input)

        assert result["review_analysis"]["sentiment"] == "positive"
        assert result["ticket_triage"]["refund_involved"] is True

    def test_live_mode_import_error_propagates(self, sample_input):
        """Missing pydantic_ai package must raise ImportError with a helpful message."""
        with patch.dict("sys.modules", {"pydantic_ai": None}):
            from main import run_live
            with pytest.raises((ImportError, TypeError)):
                run_live(sample_input)


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Tests that verify sample input/output files are valid and schema-compliant."""

    def test_sample_input_loads(self):
        """load_sample_input() must return a dict without raising."""
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_review_key(self):
        """sample_input.json must contain a 'review' key."""
        data = load_sample_input()
        assert "review" in data, "sample_input.json must contain a 'review' key"

    def test_sample_input_has_ticket_key(self):
        """sample_input.json must contain a 'ticket' key."""
        data = load_sample_input()
        assert "ticket" in data, "sample_input.json must contain a 'ticket' key"

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable JSON."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)

    def test_sample_output_review_matches_schema(self):
        """sample_output.json review_analysis must validate against ReviewAnalysis."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            review = ReviewAnalysis.model_validate(data["review_analysis"])
            assert review.sentiment in Sentiment

    def test_demo_fixtures_match_sample_output(self):
        """DEMO_REVIEW_OUTPUT and DEMO_TICKET_OUTPUT must match sample_output.json schema."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert data["review_analysis"]["sentiment"] == DEMO_REVIEW_OUTPUT.sentiment.value
            assert data["ticket_triage"]["urgency"] == DEMO_TICKET_OUTPUT.urgency.value
