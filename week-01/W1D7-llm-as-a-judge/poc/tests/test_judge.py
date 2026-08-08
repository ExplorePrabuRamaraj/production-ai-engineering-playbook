"""
W1D7 — LLM-as-a-Judge Evals — Unit Tests
==========================================
Run: pytest tests/ -v

All external API calls are mocked — tests pass completely offline.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allows importing from src/ without package installation
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from judge_core import (
    RUBRICS,
    CriterionVerdict,
    JudgeVerdict,
    build_judge_prompt,
    parse_verdict,
)
from main import load_sample_input, run_demo


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    return {
        "user_prompt": "What is your return policy for opened software?",
        "candidate_response": "You can return any item within 30 days for a full refund.",
        "reference": "Opened software is non-refundable. All other items may be returned within 30 days.",
    }


@pytest.fixture
def passing_verdict_json() -> str:
    return json.dumps({
        "criteria": {
            "relevance":    {"score": 3, "rationale": ""},
            "accuracy":     {"score": 3, "rationale": ""},
            "completeness": {"score": 3, "rationale": ""},
        },
        "overall": "pass",
        "confidence": "high",
    })


@pytest.fixture
def failing_verdict_json() -> str:
    return json.dumps({
        "criteria": {
            "relevance":    {"score": 3, "rationale": ""},
            "accuracy":     {"score": 1, "rationale": "Response contradicts the reference policy."},
            "completeness": {"score": 2, "rationale": "Missing the opened software exclusion."},
        },
        "overall": "fail",
        "confidence": "high",
    })


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo mode, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the pre-computed demo mode — must pass with no API key."""

    def test_demo_returns_required_keys(self, sample_input):
        result = run_demo(sample_input)
        for key in ("verdict", "needs_human_review", "summary", "model", "parse_attempts"):
            assert key in result, f"Missing key: {key}"

    def test_demo_model_field_is_demo(self, sample_input):
        result = run_demo(sample_input)
        assert result["model"] == "demo"

    def test_demo_verdict_is_fail_for_contradicting_response(self, sample_input):
        """The pre-computed demo verdict should be 'fail' because the response
        contradicts the reference (opened software is non-refundable)."""
        result = run_demo(sample_input)
        assert result["verdict"]["overall"] == "fail"

    def test_demo_routes_to_human_review(self, sample_input):
        """A fail verdict must always route to human review."""
        result = run_demo(sample_input)
        assert result["needs_human_review"] is True

    def test_demo_summary_is_non_empty_string(self, sample_input):
        result = run_demo(sample_input)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10


# ---------------------------------------------------------------------------
# TestCoreConcept — rubric, prompt-building, verdict parsing
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for the pure logic in judge_core — no network calls."""

    def test_rubric_v1_has_required_criteria(self):
        rubric = RUBRICS["v1.0"]
        for criterion in ("relevance", "accuracy", "completeness"):
            assert criterion in rubric, f"Missing criterion: {criterion}"

    def test_build_judge_prompt_returns_two_messages(self, sample_input):
        messages = build_judge_prompt(
            user_prompt=sample_input["user_prompt"],
            candidate_response=sample_input["candidate_response"],
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_judge_prompt_includes_sentinel_delimiters(self, sample_input):
        """Sentinel delimiters protect against prompt injection from the candidate response."""
        messages = build_judge_prompt(
            user_prompt=sample_input["user_prompt"],
            candidate_response=sample_input["candidate_response"],
        )
        user_content = messages[1]["content"]
        assert "<<<BEGIN_RESPONSE>>>" in user_content
        assert "<<<END_RESPONSE>>>" in user_content

    def test_build_judge_prompt_includes_reference_when_provided(self, sample_input):
        messages = build_judge_prompt(
            user_prompt=sample_input["user_prompt"],
            candidate_response=sample_input["candidate_response"],
            reference=sample_input["reference"],
        )
        assert "<<<BEGIN_REFERENCE>>>" in messages[1]["content"]

    def test_build_judge_prompt_omits_reference_block_when_empty(self, sample_input):
        messages = build_judge_prompt(
            user_prompt=sample_input["user_prompt"],
            candidate_response=sample_input["candidate_response"],
            reference="",
        )
        assert "<<<BEGIN_REFERENCE>>>" not in messages[1]["content"]

    def test_parse_verdict_pass(self, passing_verdict_json):
        verdict = parse_verdict(passing_verdict_json)
        assert verdict.overall == "pass"
        assert verdict.confidence == "high"
        assert all(cv.score == 3 for cv in verdict.criteria.values())

    def test_parse_verdict_fail(self, failing_verdict_json):
        verdict = parse_verdict(failing_verdict_json)
        assert verdict.overall == "fail"
        assert verdict.criteria["accuracy"].score == 1
        assert "reference policy" in verdict.criteria["accuracy"].rationale

    def test_parse_verdict_rejects_invalid_overall(self):
        bad_json = json.dumps({
            "criteria": {"relevance": {"score": 3, "rationale": ""}},
            "overall": "excellent",       # invalid value
            "confidence": "high",
        })
        with pytest.raises(ValueError, match="Invalid overall value"):
            parse_verdict(bad_json)

    def test_parse_verdict_rejects_out_of_range_score(self):
        bad_json = json.dumps({
            "criteria": {"relevance": {"score": 5, "rationale": ""}},  # score 5 invalid
            "overall": "pass",
            "confidence": "high",
        })
        with pytest.raises(ValueError):
            parse_verdict(bad_json)

    def test_criterion_verdict_requires_rationale_for_low_score(self):
        with pytest.raises(ValueError, match="Rationale is required"):
            CriterionVerdict(score=1, rationale="")

    def test_judge_verdict_needs_human_review_on_fail(self, failing_verdict_json):
        verdict = parse_verdict(failing_verdict_json)
        assert verdict.needs_human_review() is True

    def test_judge_verdict_no_human_review_on_clean_pass(self, passing_verdict_json):
        verdict = parse_verdict(passing_verdict_json)
        assert verdict.needs_human_review() is False

    def test_judge_verdict_summary_contains_overall(self, passing_verdict_json):
        verdict = parse_verdict(passing_verdict_json)
        assert "PASS" in verdict.summary()

    @pytest.mark.parametrize("overall,confidence,expect_review", [
        ("pass",   "high",   False),
        ("pass",   "low",    True),   # low confidence always routes to review
        ("review", "medium", True),
        ("fail",   "high",   True),
        ("fail",   "low",    True),
    ])
    def test_human_review_routing_matrix(self, overall, confidence, expect_review):
        """Routing logic covers all verdict/confidence combinations."""
        criteria = {"relevance": CriterionVerdict(score=3 if overall == "pass" else 2,
                                                   rationale="" if overall == "pass" else "needs work")}
        # For 'fail', force at least one score=1
        if overall == "fail":
            criteria["relevance"] = CriterionVerdict(score=1, rationale="critical failure")
        verdict = JudgeVerdict(
            criteria=criteria,
            overall=overall,
            confidence=confidence,
        )
        assert verdict.needs_human_review() is expect_review


# ---------------------------------------------------------------------------
# TestLiveMode — mocked OpenAI API
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with all API calls mocked.

    OpenAI is imported lazily inside run_live(), so we patch 'openai.OpenAI'
    (the source) rather than 'main.OpenAI' (which is never bound at module level).
    """

    def _make_mock_client(self, content: str) -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        mock_response.usage.total_tokens = 120
        mock_response.model = "gpt-4o-mini"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @patch("openai.OpenAI")
    def test_live_mode_calls_api_once_on_valid_response(
        self, mock_openai_class, sample_input, passing_verdict_json
    ):
        mock_openai_class.return_value = self._make_mock_client(passing_verdict_json)
        from main import run_live
        result = run_live(sample_input)
        mock_openai_class.return_value.chat.completions.create.assert_called_once()
        assert result["verdict"]["overall"] == "pass"

    @patch("openai.OpenAI")
    def test_live_mode_returns_correct_schema(
        self, mock_openai_class, sample_input, passing_verdict_json
    ):
        mock_openai_class.return_value = self._make_mock_client(passing_verdict_json)
        from main import run_live
        result = run_live(sample_input)
        for key in ("verdict", "needs_human_review", "summary", "model", "parse_attempts"):
            assert key in result

    @patch("openai.OpenAI")
    def test_live_mode_retries_on_invalid_json(self, mock_openai_class, sample_input, passing_verdict_json):
        """First call returns garbage; second call returns valid JSON — retry logic fires."""
        mock_client = MagicMock()
        bad_response = MagicMock()
        bad_response.choices[0].message.content = "I cannot evaluate this."
        good_response = MagicMock()
        good_response.choices[0].message.content = passing_verdict_json
        mock_client.chat.completions.create.side_effect = [bad_response, good_response]
        mock_openai_class.return_value = mock_client

        from main import run_live
        result = run_live(sample_input)
        assert mock_client.chat.completions.create.call_count == 2
        assert result["parse_attempts"] == 2

    @patch("openai.OpenAI")
    def test_live_mode_raises_after_max_retries(self, mock_openai_class, sample_input):
        """When all retries return invalid JSON, a ValueError is raised."""
        mock_client = MagicMock()
        bad_response = MagicMock()
        bad_response.choices[0].message.content = "not json at all"
        mock_client.chat.completions.create.return_value = bad_response
        mock_openai_class.return_value = mock_client

        from main import run_live
        with pytest.raises(ValueError, match="failed to return valid JSON"):
            run_live(sample_input)


# ---------------------------------------------------------------------------
# TestSampleFiles — validates sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validates that sample JSON files exist, load, and match expected schema."""

    def test_sample_input_loads_successfully(self):
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_required_keys(self):
        data = load_sample_input()
        assert "user_prompt" in data
        assert "candidate_response" in data

    def test_sample_output_is_valid_json(self):
        path = Path(__file__).parent.parent / "sample_output.json"
        if path.exists():
            data = json.loads(path.read_text())
            assert isinstance(data, dict)
            assert "verdict" in data

    @pytest.mark.parametrize("user_prompt,candidate_response,description", [
        (
            "Is express shipping available?",
            "Yes, express shipping takes 1-2 business days.",
            "on-topic accurate response",
        ),
        (
            "What is the refund window?",
            "Our policy depends on the product category.",
            "vague but on-topic response",
        ),
        (
            "Can I return a digital download?",
            "All sales are final for physical goods.",
            "off-topic response — wrong category",
        ),
    ])
    def test_build_prompt_succeeds_for_varied_inputs(
        self, user_prompt, candidate_response, description
    ):
        """Prompt builder handles varied inputs without raising exceptions."""
        messages = build_judge_prompt(
            user_prompt=user_prompt,
            candidate_response=candidate_response,
        )
        assert len(messages) == 2, f"Expected 2 messages for: {description}"
        assert candidate_response in messages[1]["content"]
