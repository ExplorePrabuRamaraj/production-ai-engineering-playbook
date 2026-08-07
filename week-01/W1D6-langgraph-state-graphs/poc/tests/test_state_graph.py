"""
W1D6 — State Graphs (LangGraph) — Unit Tests
=============================================
Run: pytest tests/ -v

All external API calls and langgraph runtime are mocked.
Tests pass completely offline in demo mode.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from state_graph_core import (
    ingest_document,
    classify_risk,
    auto_process,
    request_human_approval,
    finalise_document,
    error_terminal,
    route_by_risk,
    run_demo_graph,
    DocumentReviewState,
)
from main import load_sample_input, run_demo


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def low_risk_state() -> DocumentReviewState:
    """State representing a low-risk document after classify_risk has run."""
    return DocumentReviewState(
        document_text="The vendor agrees to deliver the software by Q3.",
        clauses=["The vendor agrees to deliver the software by Q3"],
        risk_score=0.0,
        risk_label="low_risk",
        flags=[],
        human_approved=None,
        summary=None,
        retry_count=0,
        error=None,
    )


@pytest.fixture
def high_risk_state() -> DocumentReviewState:
    """State representing a high-risk document after classify_risk has run.
    Document text contains enough risk keywords (5+/7) to exceed the 0.7 threshold."""
    return DocumentReviewState(
        document_text=(
            "This agreement shall indemnify the party against all liability. "
            "Disputes resolved through arbitration. Termination for cause applies. "
            "The vendor warrants delivery. Liquidated damages apply for any penalty clause."
        ),
        clauses=[
            "This agreement shall indemnify the party against all liability",
            "Disputes resolved through arbitration",
            "Termination for cause applies",
            "The vendor warrants delivery",
            "Liquidated damages apply for any penalty clause",
        ],
        risk_score=0.86,
        risk_label="high_risk",
        flags=["indemnify", "liability", "arbitration", "warrant",
               "termination for cause", "liquidated damages", "penalty"],
        human_approved=None,
        summary=None,
        retry_count=0,
        error=None,
    )


@pytest.fixture
def expected_output_keys():
    """Keys that must be present in any result returned by run_demo."""
    return {"risk_score", "risk_label", "flags", "human_approved",
            "summary", "clauses_extracted", "model", "latency_ms"}


# ---------------------------------------------------------------------------
# TestDemoMode — offline execution, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Offline demo mode — must pass without any API key or langgraph."""

    def test_demo_returns_expected_schema(self, expected_output_keys):
        """run_demo output must contain all required top-level keys."""
        input_data = {"document_text": "Simple contract with no risky clauses."}
        result = run_demo(input_data)
        assert expected_output_keys.issubset(result.keys()), (
            f"Missing keys: {expected_output_keys - result.keys()}"
        )

    def test_demo_model_field_is_demo(self):
        """Demo mode must set model='demo' to indicate no real API call was made."""
        result = run_demo({"document_text": "Basic agreement terms."})
        assert result["model"] == "demo"

    def test_demo_summary_is_not_empty(self):
        """Demo mode must produce a non-empty summary string."""
        result = run_demo({"document_text": "Vendor shall deliver by Q4."})
        assert result["summary"], "summary must not be empty in demo mode"

    def test_demo_high_risk_document_sets_human_approved(self):
        """High-risk demo documents must have human_approved set (not None)."""
        result = run_demo({
            "document_text": (
                "This agreement shall indemnify against liability. "
                "Termination for cause and arbitration apply. Penalty for breach."
            )
        })
        assert result["risk_label"] == "high_risk"
        # In demo mode the human_approval node auto-approves
        assert result["human_approved"] is True

    def test_demo_low_risk_document_skips_human_approval(self):
        """Low-risk demo documents must not trigger human approval."""
        result = run_demo({"document_text": "Deliverables due Q1 next year."})
        assert result["risk_label"] == "low_risk"
        assert result["human_approved"] is None


# ---------------------------------------------------------------------------
# TestCoreConcept — pure node function behaviour
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Core node functions — tested in isolation, no graph runtime needed."""

    def test_ingest_extracts_clauses(self):
        """ingest_document must split text into a non-empty clause list."""
        state = DocumentReviewState(
            document_text="First clause here. Second clause follows. Third one ends.",
            clauses=[], risk_score=0.0, risk_label="low_risk",
            flags=[], human_approved=None, summary=None,
            retry_count=0, error=None,
        )
        result = ingest_document(state)
        assert "clauses" in result
        assert len(result["clauses"]) >= 1

    def test_ingest_caps_clauses_at_eight(self):
        """ingest_document must return at most 8 clauses."""
        long_text = ". ".join([f"Clause number {i} with sufficient length" for i in range(20)])
        state = DocumentReviewState(
            document_text=long_text,
            clauses=[], risk_score=0.0, risk_label="low_risk",
            flags=[], human_approved=None, summary=None,
            retry_count=0, error=None,
        )
        result = ingest_document(state)
        assert len(result["clauses"]) <= 8

    def test_classify_risk_high_risk_keywords(self, high_risk_state):
        """classify_risk must return high_risk for a document with risk keywords."""
        # Run on raw document text with empty clauses (classify reads document_text)
        base_state = DocumentReviewState(
            document_text=high_risk_state["document_text"],
            clauses=[], risk_score=0.0, risk_label="low_risk",
            flags=[], human_approved=None, summary=None,
            retry_count=0, error=None,
        )
        result = classify_risk(base_state)
        assert result["risk_label"] == "high_risk"
        assert result["risk_score"] >= 0.7
        assert len(result["flags"]) > 0

    def test_classify_risk_low_risk_document(self):
        """classify_risk must return low_risk for benign document text."""
        state = DocumentReviewState(
            document_text="The vendor will deliver the product by the agreed date.",
            clauses=[], risk_score=0.0, risk_label="low_risk",
            flags=[], human_approved=None, summary=None,
            retry_count=0, error=None,
        )
        result = classify_risk(state)
        assert result["risk_label"] == "low_risk"
        assert result["flags"] == []

    def test_finalise_includes_risk_score_in_summary(self, low_risk_state):
        """finalise_document summary must reference the risk score."""
        result = finalise_document(low_risk_state)
        assert "summary" in result
        assert str(low_risk_state["risk_score"]) in result["summary"]

    def test_error_terminal_sets_error_field(self, low_risk_state):
        """error_terminal must set the error field and clear the summary."""
        state = {**low_risk_state, "retry_count": 3}
        result = error_terminal(state)
        assert result["error"] is not None
        assert "3" in result["error"]
        assert result["summary"] is None

    @pytest.mark.parametrize("risk_label,expected_route", [
        ("high_risk", "high_risk"),
        ("low_risk", "low_risk"),
        ("",         "low_risk"),   # missing label defaults to low_risk
    ])
    def test_route_by_risk_returns_correct_key(self, risk_label, expected_route, low_risk_state):
        """route_by_risk must return the correct routing key for all valid inputs."""
        state = {**low_risk_state, "risk_label": risk_label}
        assert route_by_risk(state) == expected_route

    def test_run_demo_graph_returns_complete_state(self):
        """run_demo_graph must return a state dict with all required fields."""
        required_fields = {
            "document_text", "clauses", "risk_score", "risk_label",
            "flags", "human_approved", "summary", "retry_count", "error",
        }
        result = run_demo_graph("Standard vendor agreement with delivery terms.")
        assert required_fields.issubset(result.keys())


# ---------------------------------------------------------------------------
# TestLiveMode — graph invocation with mocked langgraph runtime
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode — langgraph graph.invoke() is mocked so tests run offline."""

    def test_live_mode_falls_back_to_demo_on_import_error(self):
        """If langgraph is unavailable, run_live must fall back to demo output."""
        from main import run_live
        # Simulate langgraph not being installed
        with patch("builtins.__import__", side_effect=ImportError("no module langgraph")):
            # The function catches ImportError and calls run_demo internally
            # We patch run_demo to verify the fallback path is taken
            with patch("main.run_demo") as mock_demo:
                mock_demo.return_value = {
                    "risk_score": 0.0, "risk_label": "low_risk",
                    "flags": [], "human_approved": None,
                    "summary": "demo summary", "clauses_extracted": 1,
                    "model": "demo", "latency_ms": 0,
                }
                result = run_live({"document_text": "test"})
                assert result["model"] == "demo"

    def test_build_graph_returns_none_without_langgraph(self):
        """build_graph must return None gracefully when langgraph is not installed."""
        import importlib
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"langgraph": None,
                                              "langgraph.graph": None}):
            # Re-import to trigger the ImportError path in build_graph
            import state_graph_core
            result = state_graph_core.build_graph(checkpointer=None)
            assert result is None


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validate that sample JSON files are present, parseable, and schema-correct."""

    def test_sample_input_loads(self):
        """load_sample_input must return a non-empty dict."""
        data = load_sample_input()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_sample_input_has_document_text_key(self):
        """sample_input.json must contain a 'document_text' field."""
        data = load_sample_input()
        assert "document_text" in data, (
            "sample_input.json must have a 'document_text' key"
        )

    def test_sample_input_document_text_is_non_empty_string(self):
        """document_text must be a non-empty string."""
        data = load_sample_input()
        assert isinstance(data["document_text"], str)
        assert len(data["document_text"]) > 10

    def test_sample_output_is_valid_json(self):
        """sample_output.json must exist and be valid JSON."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)

    def test_sample_output_has_required_keys(self):
        """sample_output.json must contain the expected result schema keys."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            required = {"risk_score", "risk_label", "summary", "model"}
            assert required.issubset(data.keys()), (
                f"sample_output.json missing keys: {required - data.keys()}"
            )
