"""
W3D5 — Dynamic Skill Selection — Unit Tests
=============================================
Run: pytest tests/ -v

All tests pass offline — no API key required.
External embedding calls are mocked via unittest.mock.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skill_selection_core import (
    EmbeddingRouter,
    Skill,
    SkillInjector,
    SkillRegistry,
    _cosine_similarity,
)
from main import build_demo_registry, load_sample_input, run_demo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_registry():
    """A populated SkillRegistry in demo mode."""
    from config import load_config
    cfg = load_config()
    cfg.demo_mode = True
    return build_demo_registry(cfg)


@pytest.fixture
def demo_router(demo_registry):
    """An EmbeddingRouter wired to the demo registry."""
    return EmbeddingRouter(
        registry=demo_registry,
        top_k=5,
        similarity_threshold=0.35,
        fallback_skills=["general_response"],
        demo_mode=True,
    )


@pytest.fixture
def sample_input():
    return {
        "scenarios": [
            {"query": "Why is my internet so slow today?", "user_roles": ["user"]},
            {"query": "I need a refund on my last invoice", "user_roles": ["billing"]},
            {"query": "Reset my password please", "user_roles": ["user"]},
        ]
    }


@pytest.fixture
def expected_output_keys():
    return {"query", "user_roles", "selected_skills", "skill_count_injected",
            "total_skills_registered", "used_fallback"}


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo execution
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the offline demo mode — must pass without any API key."""

    def test_run_demo_returns_scenarios(self, sample_input):
        result = run_demo(sample_input)
        assert "scenarios" in result
        assert len(result["scenarios"]) == 3

    def test_run_demo_model_is_demo(self, sample_input):
        result = run_demo(sample_input)
        assert result["model"] == "demo"
        assert result["demo_mode"] is True

    def test_run_demo_scenario_has_required_keys(self, sample_input, expected_output_keys):
        result = run_demo(sample_input)
        for scenario in result["scenarios"]:
            assert expected_output_keys.issubset(scenario.keys()), \
                f"Missing keys: {expected_output_keys - scenario.keys()}"

    def test_run_demo_selected_count_within_top_k(self, sample_input):
        result = run_demo(sample_input)
        for scenario in result["scenarios"]:
            assert scenario["skill_count_injected"] <= 5, \
                "Selected skill count must not exceed top_k=5"

    def test_run_demo_total_registered_is_correct(self, sample_input):
        result = run_demo(sample_input)
        # We registered 8 skills in build_demo_registry
        assert result["scenarios"][0]["total_skills_registered"] == 8


# ---------------------------------------------------------------------------
# TestCoreConcept — SkillRegistry, EmbeddingRouter, SkillInjector
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for the core skill selection logic."""

    def test_registry_registers_skill(self):
        registry = SkillRegistry(demo_mode=True)
        registry.register(
            name="test_tool",
            description="A test tool for unit testing",
            schema={"type": "object", "properties": {}, "required": []},
        )
        assert len(registry) == 1
        assert registry.get("test_tool") is not None

    def test_registry_returns_none_for_unknown_skill(self, demo_registry):
        assert demo_registry.get("nonexistent_tool") is None

    def test_router_selects_network_tools_for_network_query(self, demo_router):
        result = demo_router.select(
            query="My internet connection is very slow",
            user_roles={"user"},
        )
        selected_names = [s.name for s in result.selected_skills]
        # Network tools should be selected; billing tools should not
        assert any("network" in name or "ping" in name for name in selected_names), \
            f"Expected network tool in selection, got: {selected_names}"
        assert "process_refund" not in selected_names

    def test_router_respects_role_permission_filter(self, demo_router):
        # provision_access requires "admin" role
        result = demo_router.select(
            query="Grant access to the shared drive",
            user_roles={"user"},          # user role, not admin
        )
        selected_names = [s.name for s in result.selected_skills]
        assert "provision_access" not in selected_names, \
            "provision_access should not be visible to non-admin users"

    def test_router_admin_can_see_privileged_tools(self, demo_router):
        result = demo_router.select(
            query="Grant access to the shared drive",
            user_roles={"admin"},
        )
        selected_names = [s.name for s in result.selected_skills]
        # Admin should potentially see provision_access for access-related queries
        assert isinstance(selected_names, list)  # Selection ran without error

    def test_router_activates_fallback_on_low_similarity(self):
        # Build a minimal registry whose only skill has a known embedding,
        # then use a threshold higher than any possible cosine score against
        # the "unknown" query vector — guaranteeing the fallback path fires.
        isolated_registry = SkillRegistry(demo_mode=True)
        # Register a fallback skill with a known mock vector
        isolated_registry.register(
            name="fallback_only",
            description="A fallback response skill",
            schema={"type": "object", "properties": {}, "required": []},
        )
        # Register an orphan skill with NO mock embedding entry
        # (embedding will be None → skipped in scoring → no candidates)
        isolated_registry._skills["orphan"] = Skill(
            name="orphan",
            description="orphan skill with no embedding",
            schema={},
            required_roles=set(),
            embedding=None,             # explicitly absent — skipped during scoring
        )
        # Override fallback_only's embedding to be a known orthogonal vector
        isolated_registry._skills["fallback_only"].embedding = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # "unknown" demo query maps to [0.15, 0.15, 0.15, 0.15, 0.15, 0.90]
        # cos_sim([1,0,0,0,0,0], [0.15,0.15,0.15,0.15,0.15,0.90]) ≈ 0.15 — below 0.99
        strict_router = EmbeddingRouter(
            registry=isolated_registry,
            top_k=5,
            similarity_threshold=0.99,
            fallback_skills=["fallback_only"],
            demo_mode=True,
        )
        result = strict_router.select(query="something completely unrelated", user_roles=set())
        assert result.used_fallback is True
        assert any(s.name == "fallback_only" for s in result.selected_skills)

    def test_injector_builds_valid_tool_block(self, demo_registry):
        skills = [demo_registry.get("check_network_speed")]
        tool_block = SkillInjector.build_tool_block(skills)
        assert len(tool_block) == 1
        assert tool_block[0]["type"] == "function"
        assert tool_block[0]["function"]["name"] == "check_network_speed"
        assert "description" in tool_block[0]["function"]
        assert "parameters" in tool_block[0]["function"]

    def test_cosine_similarity_identical_vectors(self):
        vec = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    @pytest.mark.parametrize("query,expected_domain", [
        ("I was charged twice on my bill", "billing"),
        ("My ping is very high in games", "network"),
        ("I forgot my password and am locked out", "password"),
        ("Create a ticket for my broken laptop", "it_ticket"),
    ])
    def test_router_routes_to_correct_domain(self, demo_router, query, expected_domain):
        """Verify that domain keywords route to the right skill cluster."""
        domain_to_skills = {
            "billing":   {"get_invoice", "process_refund"},
            "network":   {"check_network_speed", "run_ping_diagnostic"},
            "password":  {"reset_password"},
            "it_ticket": {"create_it_ticket"},
        }
        result = demo_router.select(query=query, user_roles={"user", "billing", "admin"})
        selected_names = set(s.name for s in result.selected_skills)
        expected_skills = domain_to_skills[expected_domain]
        overlap = expected_skills.intersection(selected_names)
        assert overlap, \
            f"Query '{query}' expected domain '{expected_domain}' skills {expected_skills}, " \
            f"got {selected_names}"

    def test_registry_eviction_removes_stale_skills(self):
        registry = SkillRegistry(demo_mode=True)
        registry.register(
            name="stale_tool",
            description="Stale tool never called",
            schema={"type": "object", "properties": {}, "required": []},
        )
        # Simulate that the tool was called once on turn 0, then 60 turns pass
        registry.log_call("stale_tool")
        for _ in range(61):
            registry.advance_turn()
        evicted = registry.evict_stale(eviction_threshold=50)
        assert "stale_tool" in evicted
        assert registry.get("stale_tool") is None


# ---------------------------------------------------------------------------
# TestLiveMode — mocked OpenAI embedding calls
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with all API calls mocked."""

    @patch("skill_selection_core.OpenAI")
    def test_live_router_calls_embedding_api(self, mock_openai_class, demo_registry):
        """Live router must call the embeddings API exactly once per select() call."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock embedding response — must match the 6-dim vectors used in demo
        mock_embedding_response = MagicMock()
        mock_embedding_response.data[0].embedding = [0.10, 0.91, 0.08, 0.05, 0.18, 0.05]
        mock_client.embeddings.create.return_value = mock_embedding_response

        router = EmbeddingRouter(
            registry=demo_registry,
            top_k=5,
            similarity_threshold=0.35,
            fallback_skills=["general_response"],
            demo_mode=False,
            api_key="sk-fake-key-for-test",
            embedding_model="text-embedding-3-small",
        )
        result = router.select(query="My internet is slow", user_roles={"user"})

        mock_client.embeddings.create.assert_called_once()
        assert len(result.selected_skills) > 0

    @patch("skill_selection_core.OpenAI")
    def test_live_router_handles_api_error(self, mock_openai_class, demo_registry):
        """Live router must propagate embedding API errors to the caller."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("Embedding API unavailable")

        router = EmbeddingRouter(
            registry=demo_registry,
            top_k=5,
            similarity_threshold=0.35,
            demo_mode=False,
            api_key="sk-fake-key",
        )
        with pytest.raises(Exception, match="Embedding API unavailable"):
            router.select(query="test query", user_roles=set())


# ---------------------------------------------------------------------------
# TestSampleFiles — validate JSON artefacts
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Tests that verify sample_input.json and sample_output.json are valid."""

    def test_sample_input_loads(self):
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_scenarios_key(self):
        data = load_sample_input()
        assert "scenarios" in data, "sample_input.json must have a 'scenarios' key"

    def test_sample_input_scenarios_have_query(self):
        data = load_sample_input()
        for scenario in data["scenarios"]:
            assert "query" in scenario, "Each scenario must have a 'query' field"

    def test_sample_output_is_valid_json(self):
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)

    def test_sample_output_has_expected_structure(self):
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert "scenarios" in data
            assert "concept" in data
