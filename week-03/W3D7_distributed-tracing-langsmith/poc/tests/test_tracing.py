"""
W3D7 — Distributed Tracing (LangSmith) — Unit Tests
=====================================================
Run: pytest tests/ -v

All external API calls (OpenAI, LangSmith) are mocked.
Tests pass completely offline — set DEMO_MODE=true or run without any API key.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracing_core import (
    retrieve_documents,
    rerank_documents,
    assemble_context,
    validate_answer,
    run_demo_pipeline,
    RetrievedDocument,
    DEMO_DOCUMENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_query() -> str:
    return "What is the return window for electronics?"


@pytest.fixture
def sample_docs() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            doc_id="t-001",
            content="Electronics must be returned within 15 days in original packaging.",
            score=0.88,
            source="electronics_policy.md",
            last_updated="2024-11-01",
        ),
        RetrievedDocument(
            doc_id="t-002",
            content="Standard return window is 30 days from purchase date.",
            score=0.72,
            source="return_policy.md",
            last_updated="2024-10-01",
        ),
    ]


@pytest.fixture
def expected_result_schema() -> set:
    return {"run_id", "answer", "retrieved_docs", "tokens_used", "latency_ms", "model", "spans_captured"}


# ---------------------------------------------------------------------------
# TestDemoMode — offline pipeline, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Offline demo mode — must pass without any API key."""

    def test_demo_pipeline_returns_valid_schema(self, sample_query, expected_result_schema):
        """Demo pipeline output must contain all required fields."""
        result = run_demo_pipeline(sample_query)
        result_keys = set(vars(result).keys())
        assert expected_result_schema.issubset(result_keys), \
            f"Missing fields: {expected_result_schema - result_keys}"

    def test_demo_pipeline_answer_is_not_empty(self, sample_query):
        """Demo answer must be a non-empty string."""
        result = run_demo_pipeline(sample_query)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 10

    def test_demo_pipeline_model_is_demo(self, sample_query):
        """Demo mode must set model='demo' to distinguish from live runs."""
        result = run_demo_pipeline(sample_query)
        assert result.model == "demo"

    def test_demo_pipeline_captures_five_spans(self, sample_query):
        """Demo pipeline must record exactly 5 spans: retrieve, rerank, assemble, generate, validate."""
        result = run_demo_pipeline(sample_query)
        assert result.spans_captured == 5, \
            f"Expected 5 spans, got {result.spans_captured}"

    def test_demo_pipeline_run_id_is_uuid_format(self, sample_query):
        """Each demo run must produce a unique UUID run_id."""
        import re
        result = run_demo_pipeline(sample_query)
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, result.run_id), \
            f"run_id '{result.run_id}' is not a valid UUID"


# ---------------------------------------------------------------------------
# TestCoreConcept — pure function behaviour
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Core pipeline function tests — no external calls."""

    def test_retrieve_documents_returns_correct_count(self, sample_query):
        """retrieve_documents(top_k=2) must return exactly 2 documents."""
        docs = retrieve_documents(sample_query, top_k=2)
        assert len(docs) == 2

    def test_retrieve_documents_returns_retrieved_document_objects(self, sample_query):
        """retrieve_documents must return RetrievedDocument instances."""
        docs = retrieve_documents(sample_query)
        assert all(isinstance(d, RetrievedDocument) for d in docs)

    def test_rerank_documents_sorts_by_score_descending(self, sample_docs):
        """rerank_documents must return docs in descending score order."""
        # Shuffle order before reranking
        shuffled = list(reversed(sample_docs))
        ranked = rerank_documents("test query", shuffled)
        scores = [d.score for d in ranked]
        assert scores == sorted(scores, reverse=True), \
            f"Expected descending scores, got: {scores}"

    def test_assemble_context_respects_max_chars(self, sample_docs):
        """assemble_context must not exceed max_chars."""
        context = assemble_context(sample_docs, max_chars=50)
        assert len(context) <= 50

    def test_assemble_context_includes_source_label(self, sample_docs):
        """assemble_context must include the source filename for auditability."""
        context = assemble_context(sample_docs)
        assert "electronics_policy.md" in context or "return_policy.md" in context

    @pytest.mark.parametrize("answer, expected_pass", [
        ("Electronics must be returned within 15 days in original packaging.", True),
        ("Standard return window is 30 days from purchase date.", True),
        ("The moon is made of green cheese and returns are free forever.", False),
        ("", False),
    ])
    def test_validate_answer_with_varied_inputs(self, answer, expected_pass, sample_docs):
        """validate_answer should correctly classify grounded vs. ungrounded answers."""
        result = validate_answer(answer, sample_docs)
        assert result["passed"] == expected_pass, \
            f"For answer='{answer[:40]}...', expected passed={expected_pass}, got {result['passed']}"

    def test_validate_answer_returns_required_keys(self, sample_docs):
        """validate_answer output must include 'passed' and 'reason' keys."""
        result = validate_answer("Electronics must be returned within 15 days.", sample_docs)
        assert "passed" in result
        assert "reason" in result

    def test_validate_answer_empty_string_fails(self, sample_docs):
        """Empty answer must fail validation."""
        result = validate_answer("", sample_docs)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# TestLiveMode — all external calls mocked
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode tests with OpenAI and LangSmith APIs mocked."""

    @patch("main.OpenAI")
    @patch("main.traceable", side_effect=lambda **kwargs: (lambda fn: fn))
    def test_live_mode_calls_openai(self, mock_traceable, mock_openai_cls, sample_query):
        """Live mode must invoke the OpenAI chat completions endpoint."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Return window is 15 days for electronics."
        mock_resp.usage.total_tokens = 54
        mock_resp.model = "gpt-4o-mini"
        mock_client.chat.completions.create.return_value = mock_resp

        import importlib
        import main as main_mod
        # Patch cfg to disable demo mode for this test
        with patch.object(main_mod.cfg, "demo_mode", False), \
             patch.object(main_mod.cfg, "openai_api_key", "sk-test"):
            result = main_mod.run_live(sample_query)

        assert mock_client.chat.completions.create.called
        assert result.tokens_used == 54
        assert "15 days" in result.answer

    @patch("main.OpenAI")
    @patch("main.traceable", side_effect=lambda **kwargs: (lambda fn: fn))
    def test_live_mode_propagates_api_error(self, mock_traceable, mock_openai_cls, sample_query):
        """Live mode must propagate API errors to the caller."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("rate_limit_exceeded")

        import main as main_mod
        with patch.object(main_mod.cfg, "demo_mode", False), \
             patch.object(main_mod.cfg, "openai_api_key", "sk-test"):
            with pytest.raises(Exception, match="rate_limit_exceeded"):
                main_mod.run_live(sample_query)


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validate that sample JSON files are present and correctly structured."""

    def test_sample_input_loads_as_dict(self):
        """sample_input.json must load as a dict with a 'query' key."""
        path = Path(__file__).parent.parent / "sample_input.json"
        assert path.exists(), "sample_input.json is missing"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert "query" in data, "sample_input.json must contain a 'query' key"

    def test_sample_output_loads_as_dict(self):
        """sample_output.json must load as a dict with required keys."""
        path = Path(__file__).parent.parent / "sample_output.json"
        assert path.exists(), "sample_output.json is missing"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        required = {"run_id", "answer", "tokens_used", "model", "spans_captured"}
        missing = required - set(data.keys())
        assert not missing, f"sample_output.json missing keys: {missing}"

    def test_sample_output_spans_captured_is_five(self):
        """sample_output.json must record 5 spans matching the pipeline depth."""
        path = Path(__file__).parent.parent / "sample_output.json"
        if path.exists():
            data = json.loads(path.read_text())
            assert data.get("spans_captured") == 5
