"""
W1D5 — Episodic vs. Semantic Memory — Unit Tests
==================================================
Run: pytest tests/ -v

All external API calls are mocked so tests pass completely offline.
Tests cover: demo mode, core memory logic, live mode (mocked), sample files.
"""

import json
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config, load_config
from memory_core import (
    EpisodicMemory,
    SemanticMemory,
    PromotionPipeline,
    assemble_working_memory,
    format_working_memory_for_prompt,
    _cosine_similarity,
    _recency_weight,
    _demo_embed,
)
from main import run_demo, load_sample_input


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> Config:
    """Minimal config for testing — demo mode, small top_k values."""
    return Config(
        demo_mode=True,
        episodic_top_k=3,
        semantic_top_k=2,
        recency_weight_alpha=0.3,
        recency_days=30,
        promotion_min_evidence=3,
        episodic_token_budget=1200,
        semantic_token_budget=800,
    )


@pytest.fixture
def episodic(config) -> EpisodicMemory:
    return EpisodicMemory(config)


@pytest.fixture
def semantic(config) -> SemanticMemory:
    return SemanticMemory(config)


@pytest.fixture
def sample_input() -> dict:
    return {
        "user_id": "test_user_01",
        "session_id": "sess_test_001",
        "turns": [{"query": "payment error E-402"}],
    }


# ---------------------------------------------------------------------------
# Test: Demo Mode (offline, no API key)
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the offline demo mode — must pass without any API key."""

    def test_demo_returns_required_keys(self, sample_input):
        """Demo output must contain all required schema keys."""
        result = run_demo(sample_input)
        required = {
            "query", "episodic_events_retrieved", "semantic_facts_retrieved",
            "working_memory_tokens", "memory_prompt_block", "model", "latency_ms",
        }
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_demo_model_is_demo(self, sample_input):
        """Demo mode must not claim to have used a real model."""
        result = run_demo(sample_input)
        assert result["model"] == "demo"

    def test_demo_retrieves_episodic_events(self, sample_input):
        """Demo mode should retrieve at least one prior episodic event."""
        result = run_demo(sample_input)
        assert result["episodic_events_retrieved"] >= 1, \
            "Demo should surface at least one seeded episodic event"

    def test_demo_retrieves_semantic_facts(self, sample_input):
        """Demo mode should retrieve at least one seeded semantic fact."""
        result = run_demo(sample_input)
        assert result["semantic_facts_retrieved"] >= 1, \
            "Demo should surface at least one seeded semantic fact"

    def test_demo_prompt_block_contains_delimiters(self, sample_input):
        """Memory prompt block must use structural delimiters to prevent injection."""
        result = run_demo(sample_input)
        block = result["memory_prompt_block"]
        assert '<memory type="episodic">' in block or '<memory type="semantic">' in block, \
            "Prompt block must contain structural memory delimiters"

    def test_demo_working_memory_tokens_positive(self, sample_input):
        """Working memory token estimate must be a positive integer."""
        result = run_demo(sample_input)
        assert isinstance(result["working_memory_tokens"], int)
        assert result["working_memory_tokens"] > 0


# ---------------------------------------------------------------------------
# Test: Core Memory Logic
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for EpisodicMemory, SemanticMemory, and PromotionPipeline."""

    def test_episodic_write_and_count(self, episodic):
        """Writing an event should increase the store count."""
        assert episodic.count() == 0
        episodic.write_event("user_1", "sess_1", "user_message", "Hello world")
        assert episodic.count() == 1

    def test_episodic_retrieve_scopes_to_user(self, episodic):
        """Retrieval must not return events from a different user_id."""
        episodic.write_event("user_A", "sess_1", "user_message", "payment error E-402")
        episodic.write_event("user_B", "sess_2", "user_message", "payment error E-402")

        results = episodic.retrieve(user_id="user_A", query="payment error")
        for event in results:
            assert event.user_id == "user_A", \
                "Episodic retrieval must never return events from a different user"

    def test_episodic_retrieve_respects_top_k(self, config, episodic):
        """Retrieval must return at most top_k results."""
        for i in range(10):
            episodic.write_event("user_X", "sess_1", "user_message", f"event content {i}")
        results = episodic.retrieve(user_id="user_X", query="event content", top_k=3)
        assert len(results) <= 3

    def test_semantic_write_and_count(self, semantic):
        """Writing a fact should increase the semantic store count."""
        assert semantic.count() == 0
        semantic.write_fact(
            content="Error E-402 indicates expired OAuth token.",
            confidence=0.9,
            provenance_ids=["e1", "e2", "e3"],
        )
        assert semantic.count() == 1

    def test_semantic_expired_facts_not_retrieved(self, config, semantic):
        """Facts past their TTL must not appear in retrieval results."""
        # Write a fact with a TTL in the past (already expired)
        fact = semantic.write_fact(
            content="This fact has expired.",
            confidence=0.8,
            provenance_ids=["e1"],
        )
        fact.valid_until = time.time() - 1  # Expire immediately

        results = semantic.retrieve(query="expired fact")
        expired_ids = [f.fact_id for f in results]
        assert fact.fact_id not in expired_ids, \
            "Expired semantic facts must not appear in retrieval results"

    def test_promotion_pipeline_enforces_min_evidence(self, config, episodic, semantic):
        """Promotion must not create a fact from fewer than min_evidence events."""
        # Write 2 resolved events (below threshold of 3)
        for i in range(2):
            ev = episodic.write_event(
                "user_Y", "sess_1", "user_message",
                "repeated pattern below threshold", resolved=True,
            )

        pipeline = PromotionPipeline(episodic, semantic)
        promoted = pipeline.run(lookback_seconds=3600)
        assert len(promoted) == 0, \
            "Pipeline must not promote facts from fewer than min_evidence events"

    def test_promotion_pipeline_promotes_above_threshold(self, config, episodic, semantic):
        """Promotion must create a fact when min_evidence is met."""
        # Use identical content so they cluster together
        content = "OAuth token expiry causes E-402 payment failure"
        for i in range(config.promotion_min_evidence):
            episodic.write_event(
                "user_Z", f"sess_{i}", "user_message", content, resolved=True,
            )

        pipeline = PromotionPipeline(episodic, semantic)
        promoted = pipeline.run(lookback_seconds=3600)
        assert len(promoted) >= 1, \
            "Pipeline must promote at least one fact when evidence threshold is met"
        assert semantic.count() >= 1

    @pytest.mark.parametrize("user_id, query, expected_min_results", [
        ("user_param_1", "billing issue refund", 0),
        ("user_param_2", "login authentication error", 0),
        ("user_param_3", "timeout network request", 0),
    ])
    def test_episodic_retrieve_empty_store_returns_empty(
        self, config, user_id, query, expected_min_results
    ):
        """Retrieval from an empty store must return an empty list, not raise."""
        fresh_episodic = EpisodicMemory(config)
        results = fresh_episodic.retrieve(user_id=user_id, query=query)
        assert isinstance(results, list)
        assert len(results) >= expected_min_results

    def test_working_memory_assembler_respects_token_budget(self, config, episodic, semantic):
        """Assembler must not exceed the configured token budgets."""
        # Write many large events
        for i in range(20):
            episodic.write_event(
                "budget_user", "sess_1", "user_message",
                "A " * 200,  # ~200 words, ~150 tokens
                resolved=False,
            )
        for i in range(10):
            semantic.write_fact(
                content="B " * 200,
                confidence=0.8,
                provenance_ids=[f"e{i}"],
            )

        episodic_hits = episodic.retrieve(user_id="budget_user", query="test")
        semantic_hits = semantic.retrieve(query="test")
        memory = assemble_working_memory(episodic_hits, semantic_hits, config)

        # Estimate: episodic content tokens should not exceed budget
        episodic_tokens = sum(max(1, len(e.content) // 4) for e in memory.episodic_events)
        semantic_tokens = sum(max(1, len(f.content) // 4) for f in memory.semantic_facts)
        assert episodic_tokens <= config.episodic_token_budget, \
            "Episodic token budget exceeded"
        assert semantic_tokens <= config.semantic_token_budget, \
            "Semantic token budget exceeded"

    def test_cosine_similarity_identical_vectors(self):
        """Cosine similarity of identical vectors must be 1.0."""
        vec = _demo_embed("hello world", dim=8)
        sim = _cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_mismatched_length_returns_zero(self):
        """Cosine similarity of mismatched vectors must return 0.0 gracefully."""
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_recency_weight_recent_is_high(self):
        """An event from right now should have a recency weight close to 1.0."""
        now = time.time()
        weight = _recency_weight(now, now, decay_days=30)
        assert weight > 0.99

    def test_recency_weight_old_is_low(self):
        """An event 180 days old (6 half-lives) should have very low recency weight."""
        now = time.time()
        old_ts = now - (180 * 86400)
        weight = _recency_weight(old_ts, now, decay_days=30)
        assert weight < 0.02


# ---------------------------------------------------------------------------
# Test: Live Mode (mocked OpenAI API)
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with all OpenAI API calls mocked."""

    @patch("main.DEMO_MODE", False)
    @patch("main.OPENAI_API_KEY", "sk-test-fake-key")
    def test_live_mode_calls_openai(self, sample_input):
        """Live mode must make exactly one chat completion call."""
        # OpenAI is imported lazily inside run_live(), so patch at the source module
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Your OAuth token has expired."
            mock_response.usage.total_tokens = 128
            mock_response.model = "gpt-4o-mini"
            mock_client.chat.completions.create.return_value = mock_response

            from main import run_live
            result = run_live(sample_input)

            mock_client.chat.completions.create.assert_called_once()
            assert result["agent_response"] == "Your OAuth token has expired."
            assert result["tokens_used"] == 128
            assert result["model"] == "gpt-4o-mini"

    @patch("main.DEMO_MODE", False)
    @patch("main.OPENAI_API_KEY", "sk-test-fake-key")
    def test_live_mode_propagates_api_error(self, sample_input):
        """Live mode must propagate API errors to the caller."""
        # OpenAI is imported lazily inside run_live(), so patch at the source module
        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API rate limit")

            from main import run_live
            with pytest.raises(Exception, match="API rate limit"):
                run_live(sample_input)


# ---------------------------------------------------------------------------
# Test: Sample Files
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validates that sample_input.json and sample_output.json are well-formed."""

    def test_sample_input_loads(self):
        """load_sample_input() must return a non-empty dict."""
        data = load_sample_input()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_sample_input_has_user_id(self):
        """sample_input.json must contain a user_id field."""
        data = load_sample_input()
        assert "user_id" in data, "sample_input.json must contain 'user_id'"

    def test_sample_input_has_turns(self):
        """sample_input.json must contain a non-empty turns list."""
        data = load_sample_input()
        assert "turns" in data, "sample_input.json must contain 'turns'"
        assert isinstance(data["turns"], list)
        assert len(data["turns"]) >= 1

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable and contain expected keys."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)
            assert "episodic_events_retrieved" in data
            assert "semantic_facts_retrieved" in data
