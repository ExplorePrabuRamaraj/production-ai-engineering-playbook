"""
W3D1 — Prompt Distillation — Unit Tests
=========================================
Run:  pytest tests/ -v
Run (demo mode):  DEMO_MODE=true pytest tests/ -v

All external API calls are mocked. Every test passes completely offline.

Test classes:
  TestDemoMode        — offline pre-computed path
  TestCoreConcept     — pure function logic (no API, no side effects)
  TestLiveMode        — live path with OpenAI mocked via unittest.mock
  TestSampleFiles     — validates sample_input.json and sample_output.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — tests run from the project root or tests/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config, load_config
from distillation_core import (
    _DEMO_EVAL_EXAMPLES,
    _approx_token_count,
    build_student_prompt,
    build_teacher_prompt,
    compute_token_savings,
    distill_prompt,
    run_distillation_demo,
    run_distillation_live,
    score_prompt_candidate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"
SAMPLE_OUTPUT_PATH = Path(__file__).parent.parent / "sample_output.json"


@pytest.fixture
def standard_input() -> dict:
    """Minimal valid input used across multiple test classes."""
    return {
        "task": "classify_document",
        "document_text": "This Non-Disclosure Agreement is entered into between Acme Corp and Beta LLC.",
        "categories": ["NDA", "SaaS", "Employment", "IP", "Refund", "General"],
    }


@pytest.fixture
def demo_result_schema() -> set[str]:
    """Required keys in every DistillationResult from demo or live mode."""
    return {
        "teacher_tokens",
        "student_tokens",
        "token_reduction_pct",
        "teacher_accuracy",
        "student_accuracy",
        "accuracy_delta",
        "model",
        "latency_ms",
    }


@pytest.fixture
def perfect_call_llm():
    """
    A mock LLM callable that always returns the correct label from
    _DEMO_EVAL_EXAMPLES, achieving 100% accuracy on the demo eval set.
    """
    label_map = {ex["input"]: ex["label"] for ex in _DEMO_EVAL_EXAMPLES}

    def _call(system_prompt: str, user_message: str) -> str:  # noqa: ARG001
        return label_map.get(user_message, "General")

    return _call


@pytest.fixture
def zero_accuracy_call_llm():
    """A mock LLM callable that always returns a wrong label."""
    def _call(system_prompt: str, user_message: str) -> str:  # noqa: ARG001
        return "WRONG"

    return _call


# ---------------------------------------------------------------------------
# TestDemoMode — all tests must pass with no API key
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Offline demo mode — no API key required, no network access."""

    def test_demo_returns_required_schema(self, standard_input, demo_result_schema):
        """run_distillation_demo must return all required keys."""
        result = run_distillation_demo(standard_input)
        missing = demo_result_schema - result.keys()
        assert not missing, f"Missing keys in demo result: {missing}"

    def test_demo_student_tokens_less_than_teacher(self, standard_input):
        """Distillation must always reduce token count."""
        result = run_distillation_demo(standard_input)
        assert result["student_tokens"] < result["teacher_tokens"], (
            "Student prompt must be shorter than teacher prompt"
        )

    def test_demo_token_reduction_above_50_percent(self, standard_input):
        """The demo scenario shows >50% reduction — a realistic outcome."""
        result = run_distillation_demo(standard_input)
        assert result["token_reduction_pct"] > 50.0, (
            f"Expected >50% reduction, got {result['token_reduction_pct']:.1f}%"
        )

    def test_demo_accuracy_delta_within_acceptable_range(self, standard_input):
        """Accuracy delta should be ≥ -2pp (student should stay close to teacher)."""
        result = run_distillation_demo(standard_input)
        assert result["accuracy_delta"] >= -0.02, (
            f"Accuracy drop too large: {result['accuracy_delta']:.1%}"
        )

    def test_demo_model_field_is_demo(self, standard_input):
        """Demo mode must not claim a real model was used."""
        result = run_distillation_demo(standard_input)
        assert result["model"] == "demo"

    def test_demo_latency_is_zero(self, standard_input):
        """No real API call means zero latency in demo mode."""
        result = run_distillation_demo(standard_input)
        assert result["latency_ms"] == 0

    def test_demo_result_contains_monthly_savings(self, standard_input):
        """Cost savings projection must be included in demo output."""
        result = run_distillation_demo(standard_input)
        assert "monthly_savings_usd" in result
        assert result["monthly_savings_usd"] > 0


# ---------------------------------------------------------------------------
# TestCoreConcept — pure function behaviour, no LLM, no side effects
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Core business logic — pure functions, type-checked, no I/O."""

    def test_approx_token_count_non_zero_for_non_empty_string(self):
        """Token count must be positive for any non-empty input."""
        assert _approx_token_count("Hello world this is a test sentence.") > 0

    def test_approx_token_count_empty_string_returns_one(self):
        """Empty string should return the minimum sentinel value of 1."""
        assert _approx_token_count("") == 1

    def test_build_teacher_prompt_contains_document_text(self, standard_input):
        """Teacher prompt must embed the document text."""
        prompt = build_teacher_prompt(standard_input)
        assert standard_input["document_text"] in prompt

    def test_build_student_prompt_shorter_than_teacher(self, standard_input):
        """Student prompt must be shorter than the teacher prompt."""
        teacher = build_teacher_prompt(standard_input)
        student = build_student_prompt(standard_input)
        assert len(student) < len(teacher), (
            "Student prompt should be significantly shorter than teacher prompt"
        )

    def test_score_prompt_candidate_perfect_accuracy(self, perfect_call_llm):
        """score_prompt_candidate returns 1.0 when the LLM is always correct."""
        score = score_prompt_candidate("any prompt", _DEMO_EVAL_EXAMPLES, perfect_call_llm)
        assert score == 1.0

    def test_score_prompt_candidate_zero_accuracy(self, zero_accuracy_call_llm):
        """score_prompt_candidate returns 0.0 when the LLM is always wrong."""
        score = score_prompt_candidate("any prompt", _DEMO_EVAL_EXAMPLES, zero_accuracy_call_llm)
        assert score == 0.0

    def test_score_prompt_candidate_empty_eval_set_returns_zero(self, perfect_call_llm):
        """Empty eval set should return 0.0 (no examples to score against)."""
        score = score_prompt_candidate("any prompt", [], perfect_call_llm)
        assert score == 0.0

    def test_distill_prompt_reduces_tokens_with_perfect_llm(
        self, standard_input, perfect_call_llm
    ):
        """Distillation should prune tokens when accuracy stays above the floor."""
        teacher = build_teacher_prompt(standard_input)
        result = distill_prompt(
            teacher_prompt=teacher,
            eval_examples=_DEMO_EVAL_EXAMPLES,
            call_llm_fn=perfect_call_llm,
            accuracy_floor=0.90,
            max_iterations=3,
        )
        assert result["student_tokens"] <= result["teacher_tokens"], (
            "Distillation must never increase token count"
        )

    def test_distill_prompt_respects_accuracy_floor(
        self, standard_input, zero_accuracy_call_llm
    ):
        """When every pruning attempt breaks accuracy, student == teacher."""
        teacher = build_teacher_prompt(standard_input)
        result = distill_prompt(
            teacher_prompt=teacher,
            eval_examples=_DEMO_EVAL_EXAMPLES,
            call_llm_fn=zero_accuracy_call_llm,
            accuracy_floor=0.90,
            max_iterations=5,
        )
        # With zero accuracy, no pruning should be accepted
        assert result["student_accuracy"] == result["teacher_accuracy"]

    @pytest.mark.parametrize(
        "teacher_tokens, student_tokens, daily_calls, expected_min_monthly",
        [
            (1800, 640, 6_667, 0.05),    # realistic document classifier scenario
            (3000, 500, 10_000, 0.50),   # aggressive distillation at high volume
            (500, 400, 1_000, 0.0),      # minimal savings at low volume
        ],
    )
    def test_compute_token_savings_projections(
        self, teacher_tokens, student_tokens, daily_calls, expected_min_monthly
    ):
        """compute_token_savings must return positive projections for real savings."""
        savings = compute_token_savings(
            teacher_tokens=teacher_tokens,
            student_tokens=student_tokens,
            daily_calls=daily_calls,
        )
        assert savings["monthly_savings_usd"] >= expected_min_monthly, (
            f"Expected >= ${expected_min_monthly}/month, "
            f"got ${savings['monthly_savings_usd']}"
        )
        assert savings["annual_savings_usd"] == pytest.approx(
            savings["monthly_savings_usd"] * 12, rel=0.01
        ), "Annual savings should be ~12x monthly savings"

    def test_compute_token_savings_no_reduction_returns_zero(self):
        """No token reduction should yield zero savings."""
        savings = compute_token_savings(
            teacher_tokens=1000,
            student_tokens=1000,
            daily_calls=10_000,
        )
        assert savings["daily_savings_usd"] == 0.0
        assert savings["tokens_saved_per_call"] == 0


# ---------------------------------------------------------------------------
# TestLiveMode — all OpenAI calls mocked via unittest.mock
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode with all LLM calls intercepted — no network access."""

    def test_run_distillation_live_calls_openai(self, standard_input):
        """Live mode must invoke the OpenAI client."""
        teacher = build_teacher_prompt(standard_input)
        cfg = Config(
            openai_api_key="sk-test-key",
            model="gpt-4o-mini",
            demo_mode=False,
        )

        with patch("distillation_core.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "NDA"
            mock_client.chat.completions.create.return_value = mock_response

            result = run_distillation_live(standard_input, teacher, cfg)

        assert mock_client.chat.completions.create.called, (
            "OpenAI API must be called in live mode"
        )
        assert "teacher_tokens" in result
        assert "student_tokens" in result

    def test_run_distillation_live_returns_correct_model(self, standard_input):
        """The result model field must match the config model."""
        teacher = build_teacher_prompt(standard_input)
        cfg = Config(
            openai_api_key="sk-test-key",
            model="gpt-4o-mini",
            demo_mode=False,
        )

        with patch("distillation_core.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "General"
            mock_client.chat.completions.create.return_value = mock_response

            result = run_distillation_live(standard_input, teacher, cfg)

        assert result["model"] == "gpt-4o-mini"

    def test_run_distillation_live_propagates_api_error(self, standard_input):
        """API errors must propagate so callers can apply retry/fallback logic."""
        teacher = build_teacher_prompt(standard_input)
        cfg = Config(openai_api_key="sk-test-key", demo_mode=False)

        with patch("distillation_core.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = RuntimeError(
                "Simulated API timeout"
            )

            with pytest.raises(RuntimeError, match="Simulated API timeout"):
                run_distillation_live(standard_input, teacher, cfg)


# ---------------------------------------------------------------------------
# TestSampleFiles — validates sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validates that sample JSON files are well-formed and schema-compliant."""

    def test_sample_input_file_exists(self):
        """sample_input.json must be present in the package root."""
        assert SAMPLE_INPUT_PATH.exists(), (
            f"sample_input.json not found at {SAMPLE_INPUT_PATH}"
        )

    def test_sample_input_is_valid_json(self):
        """sample_input.json must be parseable JSON."""
        data = json.loads(SAMPLE_INPUT_PATH.read_text())
        assert isinstance(data, dict)

    def test_sample_input_has_required_keys(self):
        """sample_input.json must contain the keys main.py expects."""
        data = json.loads(SAMPLE_INPUT_PATH.read_text())
        required_keys = {"task", "document_text", "categories"}
        missing = required_keys - data.keys()
        assert not missing, f"sample_input.json is missing keys: {missing}"

    def test_sample_input_document_text_is_non_empty(self):
        """The document_text field must contain actual text."""
        data = json.loads(SAMPLE_INPUT_PATH.read_text())
        assert data.get("document_text", "").strip(), (
            "document_text must not be empty"
        )

    def test_sample_output_file_exists(self):
        """sample_output.json must be present in the package root."""
        assert SAMPLE_OUTPUT_PATH.exists(), (
            f"sample_output.json not found at {SAMPLE_OUTPUT_PATH}"
        )

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable JSON."""
        data = json.loads(SAMPLE_OUTPUT_PATH.read_text())
        assert isinstance(data, dict)

    def test_sample_output_has_required_keys(self):
        """sample_output.json must reflect the DistillationResult schema."""
        data = json.loads(SAMPLE_OUTPUT_PATH.read_text())
        required_keys = {
            "teacher_tokens",
            "student_tokens",
            "token_reduction_pct",
            "teacher_accuracy",
            "student_accuracy",
        }
        missing = required_keys - data.keys()
        assert not missing, f"sample_output.json is missing keys: {missing}"

    def test_load_config_demo_mode_when_no_key(self, monkeypatch):
        """load_config must activate demo_mode when OPENAI_API_KEY is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEMO_MODE", "false")
        cfg = load_config()
        assert cfg.demo_mode is True, (
            "Config must default to demo_mode=True when no API key is set"
        )

    def test_load_config_demo_mode_forced_true(self, monkeypatch):
        """DEMO_MODE=true must override even when an API key is present."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
        monkeypatch.setenv("DEMO_MODE", "true")
        cfg = load_config()
        assert cfg.demo_mode is True

    def test_load_config_live_mode_when_key_present(self, monkeypatch):
        """load_config must set demo_mode=False when a key is set and DEMO_MODE=false."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
        monkeypatch.setenv("DEMO_MODE", "false")
        cfg = load_config()
        assert cfg.demo_mode is False
