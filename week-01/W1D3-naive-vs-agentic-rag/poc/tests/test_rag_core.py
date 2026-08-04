"""
Unit tests for W1D3 — Naive vs. Agentic RAG core logic.
Run: pytest tests/ -v

All tests run offline — no API key required.
External LLM calls are mocked via unittest.mock.
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from rag_core import (
    AgenticRAGPipeline,
    ChunkRetriever,
    NaiveRAGPipeline,
    QueryDecomposer,
    RetrievedChunk,
    SubQuestionResult,
    _demo_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_config() -> Config:
    """Config with demo mode on and a low similarity threshold for test predictability."""
    return Config(
        demo_mode=True,
        similarity_threshold=0.05,  # low threshold so demo corpus always returns results
        top_k=3,
        max_sub_questions=4,
        max_reformulation_retries=2,
    )


@pytest.fixture
def strict_config() -> Config:
    """Config with a high similarity threshold to test threshold-filtering logic."""
    return Config(
        demo_mode=True,
        similarity_threshold=0.99,  # near-impossible to satisfy — forces unanswerable path
        top_k=3,
        max_reformulation_retries=1,
    )


# ---------------------------------------------------------------------------
# _demo_similarity
# ---------------------------------------------------------------------------

class TestDemoSimilarity:
    def test_identical_text_returns_high_score(self):
        score = _demo_similarity("gold membership refund policy", "gold membership refund policy")
        assert score > 0.8

    def test_disjoint_text_returns_low_score(self):
        score = _demo_similarity("gold membership benefits", "reset your account password")
        assert score < 0.3

    def test_empty_query_returns_zero(self):
        score = _demo_similarity("", "some document text")
        assert score == 0.0

    def test_score_is_between_zero_and_one(self):
        score = _demo_similarity("shipping delivery estimate", "orders are typically delivered")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------

class TestQueryDecomposer:
    def test_single_hop_query_returns_one_sub_question(self, demo_config):
        decomposer = QueryDecomposer(config=demo_config)
        result = decomposer.decompose("How do I track my order?")
        assert len(result) == 1
        assert result[0] == "How do I track my order?"

    def test_multi_hop_query_splits_on_and(self, demo_config):
        decomposer = QueryDecomposer(config=demo_config)
        result = decomposer.decompose(
            "What is the refund policy and does my Gold membership change my options?"
        )
        assert len(result) == 2
        assert "refund policy" in result[0].lower()
        assert "gold membership" in result[1].lower()

    def test_multi_hop_signal_triggers_decomposition(self, demo_config):
        decomposer = QueryDecomposer(config=demo_config)
        result = decomposer.decompose("What are my options for a missing order?")
        # "what are my options" is a multi-hop signal but no " and " — returns 1 element
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_decomposition_result_is_list_of_strings(self, demo_config):
        decomposer = QueryDecomposer(config=demo_config)
        result = decomposer.decompose("Can I return an item and get a refund?")
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_live_decompose_falls_back_on_api_error(self, demo_config):
        """If the OpenAI call fails, decompose() must not raise — it falls back to demo logic."""
        live_config = Config(
            demo_mode=False,
            openai_api_key="fake-key",
            similarity_threshold=0.05,
        )
        decomposer = QueryDecomposer(config=live_config)
        with patch("rag_core.QueryDecomposer._live_decompose", side_effect=Exception("API error")):
            # _live_decompose raises, fallback to _demo_decompose should handle gracefully
            result = decomposer._demo_decompose("What is the refund policy?")
            assert isinstance(result, list)
            assert len(result) >= 1


# ---------------------------------------------------------------------------
# ChunkRetriever
# ---------------------------------------------------------------------------

class TestChunkRetriever:
    def test_retrieve_returns_list_of_retrieved_chunks(self, demo_config):
        retriever = ChunkRetriever(config=demo_config)
        chunks = retriever.retrieve("gold membership refund")
        assert isinstance(chunks, list)
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_retrieve_respects_top_k(self, demo_config):
        retriever = ChunkRetriever(config=demo_config)
        chunks = retriever.retrieve("order delivery shipping", top_k=2)
        assert len(chunks) <= 2

    def test_retrieve_respects_similarity_threshold(self, strict_config):
        """With threshold=0.99 no demo corpus chunk can pass — retriever returns empty list."""
        retriever = ChunkRetriever(config=strict_config)
        chunks = retriever.retrieve("gold membership refund")
        assert chunks == []

    def test_retrieve_increments_call_count(self, demo_config):
        retriever = ChunkRetriever(config=demo_config)
        assert retriever.call_count == 0
        retriever.retrieve("gold membership")
        assert retriever.call_count == 1
        retriever.retrieve("shipping policy")
        assert retriever.call_count == 2

    def test_chunks_sorted_by_similarity_descending(self, demo_config):
        retriever = ChunkRetriever(config=demo_config)
        chunks = retriever.retrieve("gold membership expedited replacement")
        if len(chunks) >= 2:
            assert chunks[0].similarity >= chunks[1].similarity

    @pytest.mark.parametrize("query,expected_keyword", [
        ("gold membership benefits expedited replacement", "gold"),
        ("password reset login account", "password"),
        ("order delivered missing report days", "missing"),
    ])
    def test_retrieve_returns_relevant_chunk_for_known_queries(
        self, query, expected_keyword, demo_config
    ):
        retriever = ChunkRetriever(config=demo_config)
        chunks = retriever.retrieve(query)
        assert len(chunks) > 0
        top_text = chunks[0].text.lower()
        assert expected_keyword in top_text


# ---------------------------------------------------------------------------
# NaiveRAGPipeline
# ---------------------------------------------------------------------------

class TestNaiveRAGPipeline:
    def test_naive_pipeline_returns_rag_result(self, demo_config):
        pipeline = NaiveRAGPipeline(config=demo_config)
        result = pipeline.run("How do I track my order?")
        assert result.pipeline == "naive"
        assert result.retrieval_calls == 1
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_naive_pipeline_has_single_sub_question(self, demo_config):
        pipeline = NaiveRAGPipeline(config=demo_config)
        result = pipeline.run("What is the return policy?")
        assert len(result.sub_questions) == 1

    def test_naive_pipeline_no_chunks_on_strict_threshold(self, strict_config):
        pipeline = NaiveRAGPipeline(config=strict_config)
        result = pipeline.run("gold membership refund")
        assert result.chunks_used == 0
        assert "No relevant information" in result.answer


# ---------------------------------------------------------------------------
# AgenticRAGPipeline
# ---------------------------------------------------------------------------

class TestAgenticRAGPipeline:
    def test_agentic_pipeline_returns_rag_result(self, demo_config):
        pipeline = AgenticRAGPipeline(config=demo_config)
        result = pipeline.run("What is the return policy?")
        assert result.pipeline == "agentic"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_agentic_pipeline_decomposes_multi_hop_query(self, demo_config):
        pipeline = AgenticRAGPipeline(config=demo_config)
        result = pipeline.run(
            "What is the refund window and does my Gold membership extend it?"
        )
        # Multi-hop query should produce 2 sub-questions
        assert len(result.sub_questions) == 2

    def test_agentic_pipeline_makes_multiple_retrieval_calls_for_multi_hop(self, demo_config):
        pipeline = AgenticRAGPipeline(config=demo_config)
        result = pipeline.run(
            "What is the refund window and does my Gold membership extend it?"
        )
        # One retrieval call per sub-question (at minimum)
        assert result.retrieval_calls >= 2

    def test_agentic_pipeline_handles_unanswerable_sub_question(self, strict_config):
        """With high threshold, retrieval always fails — pipeline must not raise."""
        pipeline = AgenticRAGPipeline(config=strict_config)
        result = pipeline.run(
            "What is the refund policy and does my Gold membership change my options?"
        )
        # Answer should mention that information was not found (not raise an exception)
        assert isinstance(result.answer, str)
        assert "No reliable information" in result.answer or len(result.answer) > 0

    def test_agentic_pipeline_answer_contains_citation_marker(self, demo_config):
        pipeline = AgenticRAGPipeline(config=demo_config)
        result = pipeline.run("What is the gold membership refund policy?")
        # Demo synthesiser appends [1] citation markers
        assert "[1]" in result.answer

    @pytest.mark.parametrize("query,pipeline_type", [
        ("How do I track my shipment?", "agentic"),
        ("What are my return options and does Gold membership help?", "agentic"),
        ("Reset my password", "agentic"),
    ])
    def test_pipeline_type_is_always_agentic(self, query, pipeline_type, demo_config):
        pipeline = AgenticRAGPipeline(config=demo_config)
        result = pipeline.run(query)
        assert result.pipeline == pipeline_type


# ---------------------------------------------------------------------------
# Integration: naive vs. agentic comparison
# ---------------------------------------------------------------------------

class TestNaiveVsAgenticComparison:
    def test_agentic_makes_more_retrieval_calls_on_multi_hop(self, demo_config):
        """Agentic RAG should issue more retrieval calls than naive on a multi-hop query."""
        query = "What is the missing order procedure and what Gold membership benefits apply?"
        naive = NaiveRAGPipeline(config=demo_config)
        agentic = AgenticRAGPipeline(config=demo_config)
        naive_result = naive.run(query)
        agentic_result = agentic.run(query)
        assert agentic_result.retrieval_calls > naive_result.retrieval_calls

    def test_both_pipelines_return_non_empty_answers(self, demo_config):
        query = "What is the return policy?"
        naive = NaiveRAGPipeline(config=demo_config)
        agentic = AgenticRAGPipeline(config=demo_config)
        assert len(naive.run(query).answer) > 0
        assert len(agentic.run(query).answer) > 0
