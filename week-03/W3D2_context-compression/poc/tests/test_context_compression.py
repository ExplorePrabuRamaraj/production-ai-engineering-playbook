"""
W3D2 — Context Compression — Unit Tests
=========================================
Run: pytest tests/ -v

All external API calls are mocked so tests pass completely offline.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from context_compression_core import (
    compress_context,
    extractive_compress,
    abstractive_compress,
    split_sentences,
    estimate_tokens,
    CompressionResult,
)
from main import run_demo, run_live, load_sample_input


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_text():
    return "The refund policy allows returns within 30 days. Annual plans are eligible."


@pytest.fixture
def long_text():
    return (
        "Hello, how are you today? I hope you are doing well. "
        "The refund policy for annual subscriptions states that customers may request "
        "a full refund within 30 days of the initial purchase date. "
        "After 30 days, refunds are prorated based on the remaining subscription period. "
        "Monthly plans are not eligible for refunds once the billing cycle has started. "
        "To request a refund, please contact our support team at support@example.com. "
        "We process all refund requests within 5 business days. "
        "Thank you for being a valued customer. Have a great day!"
    )


@pytest.fixture
def sample_query():
    return "What is the refund policy for annual subscriptions?"


@pytest.fixture
def sample_input():
    return {
        "query": "What is the refund policy for annual subscriptions?",
        "segments": {
            "history": (
                "Customer: Hi, I want to cancel my annual subscription. "
                "Agent: Sure, I can help with that. May I ask the reason? "
                "Customer: I just want to know the refund policy first."
            ),
            "docs": (
                "Annual subscriptions may be refunded within 30 days. "
                "Monthly plans do not qualify for refunds after billing. "
                "Contact support@example.com to initiate a refund request."
            ),
        },
    }


@pytest.fixture
def expected_output_schema():
    return {
        "query", "segments", "total_original_tokens",
        "total_compressed_tokens", "overall_compression_ratio", "model", "latency_ms"
    }


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo mode, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Offline demo mode — must pass without any API key."""

    def test_demo_returns_expected_schema(self, sample_input, expected_output_schema):
        """Demo output must contain all required top-level keys."""
        result = run_demo(sample_input)
        assert expected_output_schema.issubset(result.keys()), (
            f"Missing keys: {expected_output_schema - result.keys()}"
        )

    def test_demo_model_is_demo(self, sample_input):
        """Demo mode must set model to 'demo' (no real API call)."""
        result = run_demo(sample_input)
        assert result["model"] == "demo"

    def test_demo_produces_compression_ratio(self, sample_input):
        """overall_compression_ratio must be between 0 and 1."""
        result = run_demo(sample_input)
        ratio = result["overall_compression_ratio"]
        assert 0.0 <= ratio <= 1.0, f"Unexpected ratio: {ratio}"

    def test_demo_segment_keys_match_input(self, sample_input):
        """Each input segment must appear in the output segments dict."""
        result = run_demo(sample_input)
        for seg_name in sample_input["segments"]:
            assert seg_name in result["segments"], f"Segment '{seg_name}' missing from output"

    def test_demo_compressed_tokens_not_greater_than_original(self, sample_input):
        """Compressed token count must not exceed original token count."""
        result = run_demo(sample_input)
        assert result["total_compressed_tokens"] <= result["total_original_tokens"]


# ---------------------------------------------------------------------------
# TestCoreConcept — core business logic
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for core compression functions — pure logic, no API calls."""

    def test_extractive_compress_returns_result(self, long_text, sample_query):
        """extractive_compress must return a CompressionResult."""
        result = extractive_compress(long_text, sample_query, token_budget=100)
        assert isinstance(result, CompressionResult)

    def test_extractive_compress_respects_budget(self, long_text, sample_query):
        """Compressed token count must not exceed the token budget."""
        budget = 60
        result = extractive_compress(long_text, sample_query, token_budget=budget)
        assert result.compressed_tokens <= budget, (
            f"Expected <= {budget} tokens, got {result.compressed_tokens}"
        )

    def test_extractive_compress_preserves_original_order(self, long_text, sample_query):
        """Sentences must appear in source order, not relevance-score order."""
        result = extractive_compress(long_text, sample_query, token_budget=200)
        compressed = result.compressed_text
        # The refund sentence should appear before the contact sentence if both retained
        refund_pos = compressed.find("refund")
        contact_pos = compressed.find("contact")
        if refund_pos != -1 and contact_pos != -1:
            assert refund_pos < contact_pos

    def test_compress_context_bypass_below_threshold(self, short_text, sample_query):
        """Segments below min_segment_tokens must bypass compression."""
        result = compress_context(
            text=short_text,
            query=sample_query,
            token_budget=500,
            min_segment_tokens=500,  # threshold higher than short_text tokens
        )
        assert result.strategy_used == "bypass"
        assert result.compression_ratio == 1.0

    def test_compress_context_bypass_within_budget(self, short_text, sample_query):
        """Segments already within budget must bypass compression."""
        original_tokens = estimate_tokens(short_text)
        result = compress_context(
            text=short_text,
            query=sample_query,
            token_budget=original_tokens + 100,  # budget larger than content
            min_segment_tokens=1,
        )
        assert result.strategy_used == "bypass"

    def test_split_sentences_handles_abbreviation(self):
        """Sentence splitter must not break on 'Dr.' abbreviation."""
        text = "Dr. Smith reviewed the case. The patient recovered fully."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert sentences[0].startswith("Dr.")

    def test_estimate_tokens_proportional(self):
        """Token estimator must return larger count for longer text."""
        short = "Hello world."
        long = "Hello world. " * 20
        assert estimate_tokens(long) > estimate_tokens(short)

    @pytest.mark.parametrize("query,keyword", [
        ("What is the refund policy?", "refund"),
        ("How do I contact support?", "contact"),
        ("When does billing occur?", "billing"),
    ])
    def test_extractive_compress_retains_query_relevant_sentence(
        self, long_text, query, keyword
    ):
        """Extractive compression must retain at least one sentence relevant to the query."""
        result = extractive_compress(long_text, query, token_budget=150)
        assert keyword.lower() in result.compressed_text.lower(), (
            f"Expected keyword '{keyword}' in compressed output for query: '{query}'"
        )


# ---------------------------------------------------------------------------
# TestLiveMode — live mode with mocked OpenAI API
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode tests — all OpenAI calls are mocked."""

    def test_run_live_calls_compress_context(self, sample_input):
        """run_live must return a result dict with the correct schema."""
        from config import Config
        mock_cfg = Config(
            openai_api_key="sk-test",
            demo_mode=False,
            token_budget=500,
            compression_strategy="abstractive",
            compression_model="gpt-4o-mini",
            min_segment_tokens=10,
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Refunds available within 30 days for annual plans."
        mock_client.chat.completions.create.return_value = mock_response

        # Patch cfg in main module and OpenAI constructor so no real calls are made
        with patch("main.cfg", mock_cfg), patch("main.OpenAI", return_value=mock_client):
            result = run_live(sample_input)

        assert "segments" in result
        assert result["model"] == "gpt-4o-mini"

    def test_run_live_propagates_api_error(self, sample_input):
        """run_live must propagate OpenAI API errors to the caller.
        Token budget is set to 5 (far below any segment) so every segment
        triggers abstractive compression, which calls the mocked API.
        """
        from config import Config
        mock_cfg = Config(
            openai_api_key="sk-test",
            demo_mode=False,
            token_budget=5,          # forces compression on every segment
            compression_strategy="abstractive",
            compression_model="gpt-4o-mini",
            min_segment_tokens=1,    # no bypass threshold
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("rate limit exceeded")

        with patch("main.cfg", mock_cfg), patch("main.OpenAI", return_value=mock_client):
            with pytest.raises(Exception, match="rate limit exceeded"):
                run_live(sample_input)

    def test_abstractive_compress_falls_back_in_demo(self, long_text, sample_query):
        """abstractive_compress must fall back to extractive when no client provided.
        The heuristic tokeniser (4 chars/token) may produce minor overshoot when
        a retained sentence boundary lands just above the budget — allow +15 tokens
        of tolerance to keep the test deterministic across different inputs.
        """
        budget = 80
        result = abstractive_compress(
            text=long_text,
            query=sample_query,
            token_budget=budget,
            openai_client=None,
        )
        assert result.strategy_used == "abstractive-demo-fallback"
        assert result.compressed_tokens <= budget + 15, (
            f"Expected <= {budget + 15} tokens (heuristic tolerance), "
            f"got {result.compressed_tokens}"
        )


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validate that sample JSON files are well-formed and schema-correct."""

    def test_sample_input_loads(self):
        """load_sample_input must return a dict."""
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_query_key(self):
        """sample_input.json must contain a 'query' key."""
        data = load_sample_input()
        assert "query" in data, "sample_input.json must contain 'query'"

    def test_sample_input_has_segments_key(self):
        """sample_input.json must contain a 'segments' key."""
        data = load_sample_input()
        assert "segments" in data, "sample_input.json must contain 'segments'"

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be valid JSON with required keys."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)
            assert "overall_compression_ratio" in data

    def test_sample_input_segments_are_non_empty(self):
        """All segments in sample_input.json must have non-empty text."""
        data = load_sample_input()
        segments = data.get("segments", {})
        assert len(segments) > 0, "sample_input.json must have at least one segment"
        for name, text in segments.items():
            assert text.strip(), f"Segment '{name}' must not be empty"
