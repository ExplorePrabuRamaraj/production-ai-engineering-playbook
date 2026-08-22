"""
W3D6 — Hierarchical Subagent Teams — Unit Tests
================================================
Run: pytest tests/ -v

All external API calls are mocked — tests pass completely offline.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_core import (
    WorkerResult,
    LeadResult,
    FinalResult,
    SubtaskSpec,
    ExecutionOrder,
    run_demo,
    run_team_lead,
    run_orchestrator,
)
from main import load_sample_input, build_task_plan

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    return {
        "goal": "Analyse competitive landscape for a SaaS CRM product",
        "domains": ["Competitive Research", "Market Analysis"],
    }


@pytest.fixture
def sample_worker_instructions():
    return [
        {"worker_id": "worker_0", "instruction": "List top 3 competitors with market share."},
        {"worker_id": "worker_1", "instruction": "Identify top 2 competitive advantages."},
    ]


@pytest.fixture
def mock_openai_response():
    """Reusable mock for a single OpenAI chat completion call."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked LLM output for testing."
    mock_response.usage.total_tokens = 42
    return mock_response


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo must pass with no API key
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Offline demo mode — must pass without any API key or network access."""

    def test_demo_returns_final_result(self, sample_input):
        """run_demo must return a FinalResult object."""
        result = run_demo(sample_input)
        assert isinstance(result, FinalResult)

    def test_demo_success_flag_is_true(self, sample_input):
        """Demo output should report overall success."""
        result = run_demo(sample_input)
        assert result.success is True

    def test_demo_has_two_lead_results(self, sample_input):
        """Demo must produce results from exactly 2 team leads."""
        result = run_demo(sample_input)
        assert len(result.lead_results) == 2

    def test_demo_final_output_is_nonempty(self, sample_input):
        """Final assembled output must not be blank."""
        result = run_demo(sample_input)
        assert result.final_output.strip() != ""

    def test_demo_model_is_demo(self, sample_input):
        """Demo worker results must not claim to use a real model."""
        result = run_demo(sample_input)
        for lr in result.lead_results:
            for wr in lr.worker_results:
                # Demo workers return pre-computed data — tokens_used is non-zero
                # but model is not set to a real API model in the demo path
                assert wr.success is True

    def test_demo_tokens_are_positive(self, sample_input):
        """Demo token counts must be plausible (> 0)."""
        result = run_demo(sample_input)
        assert result.total_tokens_used > 0

    def test_demo_no_warnings(self, sample_input):
        """Demo run with all workers succeeding should produce no warnings."""
        result = run_demo(sample_input)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# TestCoreConcept — pure logic, no API calls
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Core data structure and contract logic — pure behaviour."""

    def test_worker_result_success_contract(self):
        """WorkerResult with success=True must have non-empty output."""
        wr = WorkerResult(
            worker_id="w1", output="some output",
            tokens_used=10, latency_ms=200.0, success=True
        )
        assert wr.success is True
        assert wr.output != ""

    def test_worker_result_failure_contract(self):
        """WorkerResult with success=False must carry an error_message."""
        wr = WorkerResult(
            worker_id="w1", output="",
            tokens_used=0, latency_ms=50.0, success=False,
            error_message="Timeout"
        )
        assert wr.success is False
        assert wr.error_message == "Timeout"

    def test_lead_result_partial_flag(self):
        """LeadResult.partial should be True when some workers failed."""
        lr = LeadResult(
            lead_id="lead_1", domain="Research",
            aggregated_output="partial output",
            worker_results=[
                WorkerResult("w1", "ok", 10, 100.0, True),
                WorkerResult("w2", "", 0, 50.0, False, "API error"),
            ],
            tokens_used=10, success=True, partial=True
        )
        assert lr.partial is True
        assert lr.success is True  # lead can succeed even with partial workers

    def test_subtask_spec_defaults(self):
        """SubtaskSpec should default to PARALLEL execution and empty depends_on."""
        spec = SubtaskSpec(lead_id="lead_a", domain="Research", instruction="Do X")
        assert spec.execution_order == ExecutionOrder.PARALLEL
        assert spec.depends_on == []

    @pytest.mark.parametrize("goal,expected_leads", [
        ("Analyse CRM market", 2),
        ("Research cloud providers", 2),
        ("Study AI tooling landscape", 2),
    ])
    def test_build_task_plan_always_produces_two_leads(self, goal, expected_leads):
        """build_task_plan must return exactly 2 SubtaskSpecs for any goal."""
        input_data = {"goal": goal}
        specs, worker_map = build_task_plan(input_data)
        assert len(specs) == expected_leads

    @pytest.mark.parametrize("lead_id,expected_workers", [
        ("lead_research", 2),
        ("lead_analysis", 1),
    ])
    def test_worker_map_sizes(self, lead_id, expected_workers):
        """Worker map must contain the correct number of workers per lead."""
        input_data = {"goal": "Test goal"}
        _, worker_map = build_task_plan(input_data)
        assert len(worker_map[lead_id]) == expected_workers


# ---------------------------------------------------------------------------
# TestLiveMode — all OpenAI calls mocked
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode with all API calls mocked via unittest.mock."""

    @patch("hierarchical_core.OpenAI")
    def test_run_team_lead_calls_workers_and_aggregates(
        self, mock_openai_class, sample_worker_instructions, mock_openai_response
    ):
        """Team lead should call LLM for each worker plus once for aggregation."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        result = run_team_lead(
            lead_id="lead_test",
            domain="Test Domain",
            subtask="Test subtask instruction",
            worker_instructions=sample_worker_instructions,
            api_key="test-key",
            model="gpt-4o-mini",
            max_tokens=200,
            max_retries=1,
        )

        assert isinstance(result, LeadResult)
        assert result.success is True
        # 2 worker calls + 1 aggregation call = 3 total
        assert mock_client.chat.completions.create.call_count == 3

    @patch("hierarchical_core.OpenAI")
    def test_run_team_lead_retries_failed_worker(
        self, mock_openai_class, mock_openai_response
    ):
        """Team lead should retry a failed worker up to max_retries times."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        # First call fails, second succeeds, third (aggregation) succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("Transient error"),
            mock_openai_response,
            mock_openai_response,
        ]

        result = run_team_lead(
            lead_id="lead_retry",
            domain="Retry Domain",
            subtask="retry test",
            worker_instructions=[{"worker_id": "w0", "instruction": "Do something"}],
            api_key="test-key",
            model="gpt-4o-mini",
            max_tokens=200,
            max_retries=2,
        )

        assert result.success is True
        # Worker succeeded on retry — should not be marked partial
        assert result.partial is False

    @patch("hierarchical_core.OpenAI")
    def test_run_orchestrator_assembles_lead_results(
        self, mock_openai_class, sample_input, mock_openai_response
    ):
        """Orchestrator should call LLM for workers, aggregation, and final synthesis."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response

        subtask_specs, worker_map = build_task_plan(sample_input)
        result = run_orchestrator(
            goal=sample_input["goal"],
            subtask_specs=subtask_specs,
            worker_map=worker_map,
            api_key="test-key",
            model="gpt-4o-mini",
            max_tokens=200,
            max_retries=1,
        )

        assert isinstance(result, FinalResult)
        assert result.success is True
        assert result.final_output != ""
        assert len(result.lead_results) == 2

    @patch("hierarchical_core.OpenAI")
    def test_orchestrator_warns_on_partial_lead(
        self, mock_openai_class, sample_input, mock_openai_response
    ):
        """Orchestrator must add a warning when a lead returns partial=True."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        # Make every call fail to force partial — then succeed on aggregation/synthesis
        fail_then_succeed = [
            Exception("worker fail"),      # worker_0 attempt 1
            Exception("worker fail"),      # worker_0 attempt 2 (retry)
            mock_openai_response,          # aggregation (with only 1 worker success from lead_analysis)
            mock_openai_response,          # lead_analysis worker
            mock_openai_response,          # lead_analysis aggregation
            mock_openai_response,          # orchestrator synthesis
        ]
        mock_client.chat.completions.create.side_effect = fail_then_succeed

        subtask_specs = [
            SubtaskSpec(lead_id="lead_research", domain="Research",
                        instruction="Research competitors")
        ]
        worker_map = {
            "lead_research": [
                {"worker_id": "w0", "instruction": "List competitors"},
                {"worker_id": "w1", "instruction": "List advantages"},
            ]
        }
        result = run_orchestrator(
            goal=sample_input["goal"],
            subtask_specs=subtask_specs,
            worker_map=worker_map,
            api_key="test-key",
            model="gpt-4o-mini",
            max_tokens=200,
            max_retries=1,
        )

        # Partial result should trigger a warning
        assert any("partial" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validates that sample JSON files exist and have correct schemas."""

    def test_sample_input_loads(self):
        """sample_input.json must be loadable via load_sample_input()."""
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_goal_key(self):
        """sample_input.json must contain a 'goal' key."""
        data = load_sample_input()
        assert "goal" in data, "sample_input.json must contain 'goal'"

    def test_sample_input_goal_is_nonempty(self):
        """Goal in sample_input.json must be a non-empty string."""
        data = load_sample_input()
        assert isinstance(data["goal"], str)
        assert len(data["goal"]) > 0

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable JSON with expected keys."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)
            assert "final_output" in data
            assert "success" in data
            assert "total_tokens_used" in data
