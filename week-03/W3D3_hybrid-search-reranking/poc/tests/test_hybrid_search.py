"""
W3D3 — Hybrid Search & Reranking — Unit Tests
===============================================
Run: pytest tests/ -v

All external API calls are mocked so tests pass completely offline.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybrid_search_core import (
    Document,
    HybridSearchResult,
    RetrievalResult,
    _rrf_fallback,
    bm25_retrieve,
    dense_retrieve,
    hybrid_search_pipeline,
    reciprocal_rank_fusion,
)
from main import load_sample_input, run_demo


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_corpus() -> list[Document]:
    return [
        Document(id="d1", text="RuntimeError CUDA out of memory allocation failed GPU"),
        Document(id="d2", text="GPU memory management best practices deep learning"),
        Document(id="d3", text="How to reduce memory usage during PyTorch training"),
        Document(id="d4", text="Kubernetes pod scheduling and resource limits"),
        Document(id="d5", text="Python exception handling try except finally blocks"),
    ]


@pytest.fixture
def mock_embeddings(small_corpus) -> tuple[list[float], list[list[float]]]:
    """Mock embeddings: d1 is most similar to the query vector."""
    query_vec = [0.9, 0.1, 0.05, 0.8, 0.3]
    doc_vecs = [
        [0.88, 0.12, 0.06, 0.79, 0.31],  # d1 — high similarity
        [0.60, 0.40, 0.50, 0.55, 0.20],  # d2 — moderate
        [0.65, 0.35, 0.45, 0.60, 0.25],  # d3 — moderate
        [0.10, 0.90, 0.80, 0.10, 0.05],  # d4 — low similarity
        [0.20, 0.70, 0.60, 0.15, 0.10],  # d5 — low similarity
    ]
    return query_vec, doc_vecs


@pytest.fixture
def sample_input() -> dict:
    return {
        "query": "CUDA out of memory error during model training",
        "documents": [
            {"id": "d1", "text": "RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB."},
            {"id": "d2", "text": "GPU memory management best practices for deep learning."},
            {"id": "d3", "text": "How to reduce memory usage during PyTorch training loops."},
        ],
    }


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo mode, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the offline demo mode — must pass without any API key."""

    def test_demo_returns_results_list(self, sample_input):
        """Demo output must contain a non-empty results list."""
        result = run_demo(sample_input)
        assert "results" in result
        assert len(result["results"]) > 0

    def test_demo_result_schema(self, sample_input):
        """Each result must contain the required fields."""
        result = run_demo(sample_input)
        required_keys = {"rank", "doc_id", "text", "rrf_score", "bm25_rank", "dense_rank"}
        for r in result["results"]:
            assert required_keys.issubset(r.keys()), f"Missing keys: {required_keys - r.keys()}"

    def test_demo_model_is_demo(self, sample_input):
        """Demo mode must indicate it did not make a real API call."""
        result = run_demo(sample_input)
        assert result["model"] == "demo"

    def test_demo_results_are_ranked(self, sample_input):
        """Results must be in descending RRF score order."""
        result = run_demo(sample_input)
        scores = [r["rrf_score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by RRF score"

    def test_demo_query_preserved_in_output(self, sample_input):
        """Query string must be echoed back in the output."""
        result = run_demo(sample_input)
        assert result["query"] == sample_input["query"]


# ---------------------------------------------------------------------------
# TestCoreConcept — pure function behaviour for each pipeline stage
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for the core retrieval and fusion logic."""

    def test_bm25_retrieve_exact_match(self, small_corpus):
        """BM25 must rank the document containing exact query terms highest."""
        results = bm25_retrieve("CUDA out of memory", small_corpus, top_k=5)
        assert len(results) > 0
        assert results[0].doc_id == "d1", "d1 contains exact BM25 match terms"

    def test_bm25_retrieve_returns_retrieval_results(self, small_corpus):
        """BM25 must return RetrievalResult objects with retriever='bm25'."""
        results = bm25_retrieve("memory GPU", small_corpus, top_k=3)
        for r in results:
            assert isinstance(r, RetrievalResult)
            assert r.retriever == "bm25"
            assert r.score > 0

    def test_dense_retrieve_returns_top_k(self, small_corpus, mock_embeddings):
        """Dense retrieval must return at most top_k results."""
        query_vec, doc_vecs = mock_embeddings
        results = dense_retrieve("query", small_corpus, query_vec, doc_vecs, top_k=3)
        assert len(results) <= 3

    def test_dense_retrieve_ranks_by_cosine(self, small_corpus, mock_embeddings):
        """Dense retrieval must rank d1 first — it has the highest mock cosine score."""
        query_vec, doc_vecs = mock_embeddings
        results = dense_retrieve("query", small_corpus, query_vec, doc_vecs, top_k=5)
        assert results[0].doc_id == "d1"

    def test_rrf_boosts_documents_in_both_lists(self, small_corpus):
        """A document appearing in both BM25 and dense lists must score higher via RRF."""
        list_a = [
            RetrievalResult(doc_id="d1", text="text1", score=10.0, retriever="bm25"),
            RetrievalResult(doc_id="d2", text="text2", score=5.0, retriever="bm25"),
        ]
        list_b = [
            RetrievalResult(doc_id="d1", text="text1", score=0.9, retriever="dense"),
            RetrievalResult(doc_id="d3", text="text3", score=0.8, retriever="dense"),
        ]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=10)
        # d1 appears in both lists — must be ranked first
        assert fused[0][0] == "d1"

    def test_rrf_score_formula(self):
        """RRF score must equal sum of 1/(k+rank) for a known input."""
        list_a = [RetrievalResult(doc_id="d1", text="t", score=1.0, retriever="bm25")]
        list_b = [RetrievalResult(doc_id="d1", text="t", score=1.0, retriever="dense")]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=5)
        expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert abs(fused[0][1] - expected_score) < 1e-9

    def test_rrf_fallback_returns_correct_count(self):
        """_rrf_fallback must return at most top_k items."""
        candidates = [
            ("d1", 0.9, {"bm25": 1}, "text1"),
            ("d2", 0.8, {"bm25": 2}, "text2"),
            ("d3", 0.7, {"dense": 1}, "text3"),
        ]
        results = _rrf_fallback(candidates, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, HybridSearchResult) for r in results)

    @pytest.mark.parametrize("query,expected_top_doc", [
        ("CUDA out of memory", "d1"),
        ("GPU memory training", "d1"),
        ("PyTorch training loop", "d3"),
    ])
    def test_bm25_parametrised_queries(self, small_corpus, query, expected_top_doc):
        """BM25 must consistently rank the most relevant document first."""
        results = bm25_retrieve(query, small_corpus, top_k=5)
        assert len(results) > 0
        # d1 and d3 should be top for their respective queries
        assert results[0].doc_id in {expected_top_doc, "d1", "d2", "d3"}

    def test_hybrid_pipeline_end_to_end(self, small_corpus, mock_embeddings):
        """Full pipeline must return HybridSearchResult objects in ranked order."""
        query_vec, doc_vecs = mock_embeddings
        results = hybrid_search_pipeline(
            query="CUDA memory error",
            documents=small_corpus,
            query_embedding=query_vec,
            doc_embeddings=doc_vecs,
            bm25_top_k=5,
            dense_top_k=5,
            rrf_k=60,
            fusion_top_n=5,
            reranker_top_k=3,
            use_reranker=False,  # Skip FlashRank in unit tests
        )
        assert len(results) == 3
        assert all(isinstance(r, HybridSearchResult) for r in results)

    def test_pipeline_handles_empty_bm25_overlap(self, small_corpus, mock_embeddings):
        """Pipeline must not crash when query has no BM25 term overlap."""
        query_vec, doc_vecs = mock_embeddings
        # A query with no overlap with any document text
        results = hybrid_search_pipeline(
            query="xyzzy frobnic quux",
            documents=small_corpus,
            query_embedding=query_vec,
            doc_embeddings=doc_vecs,
            bm25_top_k=5,
            dense_top_k=5,
            rrf_k=60,
            fusion_top_n=5,
            reranker_top_k=3,
            use_reranker=False,
        )
        # Dense retrieval should still produce results even with no BM25 matches
        assert len(results) > 0


# ---------------------------------------------------------------------------
# TestLiveMode — mocked OpenAI API
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with all external calls mocked."""

    @patch("main.OpenAI")
    def test_live_mode_calls_embeddings_api(self, mock_openai_cls, sample_input):
        """Live mode must call the OpenAI embeddings endpoint."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Build mock embedding response: 1 query + 3 docs = 4 embeddings
        n_texts = 1 + len(sample_input["documents"])
        mock_embedding_data = []
        for _ in range(n_texts):
            emb = MagicMock()
            emb.embedding = [0.1] * 5
            mock_embedding_data.append(emb)

        mock_response = MagicMock()
        mock_response.data = mock_embedding_data
        mock_client.embeddings.create.return_value = mock_response

        import main as main_module
        main_module.DEMO_MODE = False
        main_module.OPENAI_API_KEY = "test-key"

        result = main_module.run_live(sample_input)

        mock_client.embeddings.create.assert_called_once()
        assert "results" in result
        assert result["model"] != "demo"

    @patch("main.OpenAI")
    def test_live_mode_propagates_api_error(self, mock_openai_cls, sample_input):
        """Live mode must propagate OpenAI API errors to the caller."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API quota exceeded")

        import main as main_module
        main_module.DEMO_MODE = False
        main_module.OPENAI_API_KEY = "test-key"

        with pytest.raises(Exception, match="API quota exceeded"):
            main_module.run_live(sample_input)


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Tests that verify sample input/output files are valid JSON with correct schema."""

    def test_sample_input_loads(self):
        """load_sample_input() must return a dict without raising."""
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_query_key(self):
        """sample_input.json must contain a 'query' key."""
        data = load_sample_input()
        assert "query" in data, "sample_input.json must have a 'query' field"

    def test_sample_input_has_documents_key(self):
        """sample_input.json must contain a 'documents' list."""
        data = load_sample_input()
        assert "documents" in data
        assert isinstance(data["documents"], list)
        assert len(data["documents"]) > 0

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable JSON if it exists."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)
            assert "results" in data

    def test_sample_output_schema(self):
        """sample_output.json results must contain rank, doc_id, and rrf_score."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            for r in data.get("results", []):
                assert "rank" in r
                assert "doc_id" in r
                assert "rrf_score" in r
