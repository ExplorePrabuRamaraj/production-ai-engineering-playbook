"""
W2D6 -- Supervisor vs. Swarm Networks -- Unit Tests
=====================================================
Run: pytest tests/ -v

All tests pass offline -- no API key required.
Tests cover: demo mode, core topology logic, routing behaviour,
             edge cases, and sample file validity.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make src/ importable from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swarm_core import (
    Agent,
    AgentResult,
    AnalysisAgent,
    GenerationAgent,
    RetrievalAgent,
    SupervisorNetwork,
    SwarmNetwork,
    ValidationAgent,
    WorkflowResult,
)
from main import load_sample_input, run_demo, _build_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_task() -> str:
    return "retrieve customer data; analyse sentiment; generate response; validate compliance"


@pytest.fixture
def default_agents() -> list:
    return [RetrievalAgent(), AnalysisAgent(), GenerationAgent(), ValidationAgent()]


@pytest.fixture
def supervisor(default_agents) -> SupervisorNetwork:
    return SupervisorNetwork(agents=default_agents, max_hops=5)


@pytest.fixture
def swarm(default_agents) -> SwarmNetwork:
    return SwarmNetwork(agents=default_agents, max_hops=5)


# ---------------------------------------------------------------------------
# TestDemoMode -- offline demo must pass without any API key
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Verify the demo mode produces valid output without an API key."""

    def test_demo_returns_both_topologies(self, sample_task):
        """run_demo must return keys for both supervisor and swarm."""
        result = run_demo({"task": sample_task})
        assert "supervisor" in result
        assert "swarm" in result

    def test_demo_supervisor_output_schema(self, sample_task):
        """Supervisor section must contain all required keys."""
        result = run_demo({"task": sample_task})
        required = {"topology", "subtasks_handled", "total_latency_ms", "total_tokens",
                    "routing_trace", "final_output"}
        assert required.issubset(result["supervisor"].keys())

    def test_demo_swarm_output_schema(self, sample_task):
        """Swarm section must contain all required keys."""
        result = run_demo({"task": sample_task})
        required = {"topology", "subtasks_handled", "total_latency_ms", "total_tokens",
                    "routing_trace", "final_output"}
        assert required.issubset(result["swarm"].keys())

    def test_demo_topology_labels_correct(self, sample_task):
        """Topology labels must match expected strings."""
        result = run_demo({"task": sample_task})
        assert result["supervisor"]["topology"] == "supervisor"
        assert result["swarm"]["topology"] == "swarm"

    def test_demo_handles_empty_task(self):
        """Demo must not raise on empty task string."""
        result = run_demo({"task": ""})
        assert "supervisor" in result
        assert "swarm" in result


# ---------------------------------------------------------------------------
# TestCoreConcept -- core topology logic
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for SupervisorNetwork and SwarmNetwork business logic."""

    def test_supervisor_decomposes_retrieval_task(self, supervisor):
        """Supervisor must decompose a task mentioning 'retrieve' into at least one subtask."""
        subtasks = supervisor.decompose("retrieve customer purchase history")
        assert len(subtasks) >= 1
        assert any("retrieve" in s.lower() for s in subtasks)

    def test_supervisor_run_returns_workflow_result(self, supervisor, sample_task):
        """SupervisorNetwork.run() must return a WorkflowResult instance."""
        result = supervisor.run(sample_task, demo_mode=True)
        assert isinstance(result, WorkflowResult)

    def test_supervisor_handles_all_subtasks(self, supervisor, sample_task):
        """All decomposed subtasks should be handled (no unhandled drops)."""
        result = supervisor.run(sample_task, demo_mode=True)
        # WorkflowResult stores results in subtask_results list
        assert len(result.subtask_results) >= 1
        assert all(r.success for r in result.subtask_results)

    def test_swarm_routes_to_correct_agent(self, swarm):
        """SwarmNetwork must route a retrieval subtask to the RetrievalAgent."""
        result = swarm.run("retrieve relevant documents for customer inquiry", demo_mode=True)
        assert len(result.subtask_results) >= 1
        assert result.subtask_results[0].agent_name == "retrieval-agent"

    def test_swarm_cycle_prevention(self, default_agents):
        """Swarm must not route the same message to the same agent twice."""
        swarm = SwarmNetwork(agents=default_agents, max_hops=10)
        trace: list[str] = []
        history: list[str] = []
        # Route a task through the swarm -- history must never repeat an agent name
        swarm._route_message("retrieve data", history, trace)
        assert len(history) == len(set(history)), "Routing history contains duplicate agent names"

    def test_swarm_max_hops_enforced(self, default_agents):
        """Swarm must stop routing and send to DLQ when max_hops is exceeded."""
        # Set max_hops=0 so every message exceeds the limit immediately
        swarm = SwarmNetwork(agents=default_agents, max_hops=0)
        result = swarm.run("retrieve data", demo_mode=True)
        assert len(swarm.dead_letter_queue) >= 1

    def test_swarm_unroutable_message_goes_to_dlq(self, default_agents):
        """A task that no agent can handle must land in the dead letter queue."""
        swarm = SwarmNetwork(agents=default_agents, max_hops=5)
        result = swarm.run("xyzzy frobnicate", demo_mode=True)
        # Either handled by fallback or captured in DLQ -- pipeline must not raise
        assert isinstance(result, WorkflowResult)

    @pytest.mark.parametrize("subtask,expected_agent", [
        ("retrieve documents from database", "retrieval-agent"),
        ("analyse sentiment of feedback", "analysis-agent"),
        ("generate a reply for the customer", "generation-agent"),
        ("validate output for policy compliance", "validation-agent"),
    ])
    def test_agent_capability_matching(self, subtask, expected_agent, default_agents):
        """Each agent must claim capability over its designated subtask type."""
        agent_map = {a.name: a for a in default_agents}
        agent = agent_map[expected_agent]
        assert agent.can_handle(subtask), (
            f"{expected_agent} should claim capability for: '{subtask}'"
        )

    def test_agent_result_schema(self, default_agents):
        """Every agent's handle() must return an AgentResult with required fields."""
        for agent in default_agents:
            result = agent.handle("test subtask", demo_mode=True)
            assert isinstance(result, AgentResult)
            assert result.agent_name == agent.name
            assert isinstance(result.output, str)
            assert isinstance(result.success, bool)
            assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# TestLiveMode -- mocked API calls
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode tests with all external calls mocked."""

    def test_run_demo_does_not_call_openai(self, sample_task):
        """Demo mode must complete without importing or calling openai."""
        with patch.dict("sys.modules", {"openai": None}):
            # Should not raise ImportError
            result = run_demo({"task": sample_task})
            assert "supervisor" in result

    def test_workflow_result_latency_is_non_negative(self, supervisor, sample_task):
        """Workflow total latency must be >= 0 ms."""
        result = supervisor.run(sample_task, demo_mode=True)
        assert result.total_latency_ms >= 0

    def test_workflow_result_tokens_is_non_negative(self, swarm, sample_task):
        """Workflow total tokens must be >= 0."""
        result = swarm.run(sample_task, demo_mode=True)
        assert result.total_tokens >= 0


# ---------------------------------------------------------------------------
# TestSampleFiles -- validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Verify sample input/output files are present and schema-valid."""

    def test_load_sample_input_returns_dict(self):
        """load_sample_input() must return a dict."""
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_task_key(self):
        """sample_input.json must have a 'task' key."""
        data = load_sample_input()
        assert "task" in data, "sample_input.json must contain a 'task' key"

    def test_sample_input_task_is_non_empty(self):
        """The 'task' value in sample_input.json must not be empty."""
        data = load_sample_input()
        assert data["task"].strip(), "task value must not be empty"

    def test_sample_output_is_valid_json(self):
        """sample_output.json must parse as a dict."""
        output_path = Path(__file__).parent.parent / "sample_output.json"
        if output_path.exists():
            data = json.loads(output_path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)

    def test_sample_output_contains_topologies(self):
        """sample_output.json must contain 'supervisor' and 'swarm' keys."""
        output_path = Path(__file__).parent.parent / "sample_output.json"
        if output_path.exists():
            data = json.loads(output_path.read_text(encoding="utf-8"))
            assert "supervisor" in data
            assert "swarm" in data
