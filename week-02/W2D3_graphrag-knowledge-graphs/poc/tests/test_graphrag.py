"""
W2D3 — GraphRAG & Knowledge Graphs — Unit Tests
=================================================
Run: pytest tests/ -v

All external API calls are mocked so tests pass completely offline.
The graph logic (build, detect, traverse, merge) is pure Python with no
network dependency, so most tests run without any mocking at all.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allows importing src modules from tests/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag_core import (
    Community,
    Entity,
    RetrievalResult,
    Relationship,
    build_knowledge_graph,
    detect_communities,
    hybrid_retrieve,
    rrf_merge,
    traverse_graph,
)
from main import load_sample_input, run_demo


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_triples():
    """Three triples forming a small connected graph: Alice -> Contract -> Policy."""
    return [
        ("Alice", "APPROVED", "Contract_42"),
        ("Contract_42", "GOVERNED_BY", "Policy_GDPR_17"),
        ("Alice", "REPORTS_TO", "Bob"),
        ("Bob", "MANAGES", "Legal_Department"),
    ]


@pytest.fixture
def simple_graph(simple_triples):
    """Pre-built graph from simple_triples fixture."""
    return build_knowledge_graph(simple_triples)


@pytest.fixture
def sample_query_input():
    """Standard query input used across multiple tests."""
    return {
        "query": "Who approved the contract and which policy governs it?",
        "seed_entities": ["Alice", "Contract_42"],
        "triples": [
            ["Alice", "APPROVED", "Contract_42"],
            ["Contract_42", "GOVERNED_BY", "Policy_GDPR_17"],
        ],
    }


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo must run without any API key
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the offline demo mode — no API key required."""

    def test_demo_returns_expected_schema(self, sample_query_input):
        """Demo output must contain all required top-level keys."""
        result = run_demo(sample_query_input)
        required_keys = {"query", "graph_nodes", "graph_edges", "communities", "results", "model"}
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

    def test_demo_model_field_is_demo(self, sample_query_input):
        """Demo mode must not report a live model name."""
        result = run_demo(sample_query_input)
        assert result["model"] == "demo"

    def test_demo_results_is_list(self, sample_query_input):
        """Demo results field must be a non-empty list."""
        result = run_demo(sample_query_input)
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0

    def test_demo_result_items_have_provenance(self, sample_query_input):
        """Each result item must have a provenance field (enables audit trail)."""
        result = run_demo(sample_query_input)
        for item in result["results"]:
            assert "provenance" in item, "Each result must carry a provenance field"

    def test_demo_runs_with_minimal_triples(self):
        """Demo mode must not crash when given a single-triple graph."""
        minimal_input = {
            "query": "What is Alice's role?",
            "seed_entities": ["Alice"],
            "triples": [["Alice", "IS_A", "Manager"]],
        }
        result = run_demo(minimal_input)
        assert result["graph_nodes"] >= 1


# ---------------------------------------------------------------------------
# TestCoreConcept — pure graph logic, no mocking needed
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for the core GraphRAG logic: build, detect, traverse, merge."""

    def test_build_graph_creates_correct_node_count(self, simple_triples):
        """Graph must have exactly as many unique entities as in the triples."""
        graph = build_knowledge_graph(simple_triples)
        # Unique entities: Alice, Contract_42, Policy_GDPR_17, Bob, Legal_Department
        assert len(graph["nodes"]) == 5

    def test_build_graph_creates_correct_edge_count(self, simple_triples):
        """Graph must have one edge per unique (source, rel, target) triple."""
        graph = build_knowledge_graph(simple_triples)
        assert len(graph["edges"]) == 4

    def test_duplicate_triples_increment_edge_weight(self):
        """Repeated triples must increase edge weight, not create duplicate edges."""
        triples = [
            ("Alice", "APPROVED", "Contract_42"),
            ("Alice", "APPROVED", "Contract_42"),  # duplicate
        ]
        graph = build_knowledge_graph(triples)
        assert len(graph["edges"]) == 1
        edge = list(graph["edges"].values())[0]
        assert edge.weight == 2

    def test_detect_communities_returns_all_nodes(self, simple_graph):
        """Every node in the graph must appear in exactly one community."""
        communities = detect_communities(simple_graph)
        all_members = [nid for c in communities for nid in c.member_ids]
        assert set(all_members) == set(simple_graph["nodes"].keys())

    def test_traverse_graph_respects_hop_depth(self, simple_graph):
        """Traversal with hop_depth=1 must not reach nodes 2 hops away."""
        # Alice -> Contract_42 (1 hop) -> Policy_GDPR_17 (2 hops)
        results_depth1 = traverse_graph(simple_graph, ["alice"], hop_depth=1)
        result_texts = [r.text for r in results_depth1]
        # policy_gdpr_17 is 2 hops from alice via contract_42 — must not appear
        assert not any("policy_gdpr_17" in t.lower() for t in result_texts), \
            "Hop depth=1 traversal must not reach 2-hop nodes"

    def test_traverse_graph_respects_max_nodes(self, simple_graph):
        """Traversal must not return more nodes than max_nodes."""
        results = traverse_graph(simple_graph, ["alice"], hop_depth=5, max_nodes=2)
        assert len(results) <= 2

    def test_rrf_merge_deduplicates_overlapping_results(self):
        """RRF merge must not return duplicate entries for the same text."""
        shared_text = "Alice approved Contract_42."
        vec_results = [RetrievalResult(text=shared_text, source="vector", score=0.9)]
        graph_results = [RetrievalResult(text=shared_text, source="graph", score=0.8)]
        merged = rrf_merge(vec_results, graph_results, k=60, top_m=10)
        texts = [r.text[:80] for r in merged]
        assert len(texts) == len(set(texts)), "Merged results must not contain duplicates"

    def test_rrf_merge_higher_ranked_gets_higher_score(self):
        """Item ranked 1st in both lists must outscore item ranked 2nd in both."""
        top = RetrievalResult(text="Top result with unique content A.", source="vector", score=0.99)
        bottom = RetrievalResult(text="Lower result with unique content B.", source="vector", score=0.50)
        vec = [top, bottom]
        graph = [top, bottom]
        merged = rrf_merge(vec, graph, k=60, top_m=2)
        assert merged[0].text[:80] == top.text[:80], \
            "Item ranked 1st in both lists must be top after RRF merge"

    @pytest.mark.parametrize("hop_depth,expected_min_nodes", [
        (0, 1),   # Only seed node itself
        (1, 2),   # Seed + direct neighbours
        (2, 3),   # Seed + 2-hop chain
    ])
    def test_traverse_graph_depth_expands_results(self, simple_graph, hop_depth, expected_min_nodes):
        """Deeper traversal must retrieve at least as many nodes as shallower traversal."""
        results = traverse_graph(simple_graph, ["alice"], hop_depth=hop_depth, max_nodes=50)
        assert len(results) >= expected_min_nodes, \
            f"hop_depth={hop_depth} must yield at least {expected_min_nodes} nodes"

    def test_hybrid_retrieve_returns_results(self, simple_graph):
        """hybrid_retrieve must return a non-empty list from a valid graph."""
        communities = detect_communities(simple_graph)
        results = hybrid_retrieve(
            query="Who approved the contract?",
            graph=simple_graph,
            communities=communities,
            seed_entities=["Alice", "Contract_42"],
            vector_results=None,
            hop_depth=2,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_retrieve_handles_unknown_seed_entity(self, simple_graph):
        """hybrid_retrieve must not crash when a seed entity is not in the graph."""
        communities = detect_communities(simple_graph)
        results = hybrid_retrieve(
            query="What did Zara do?",
            graph=simple_graph,
            communities=communities,
            seed_entities=["Zara"],  # not in graph
            vector_results=None,
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestLiveMode — live mode with all OpenAI calls mocked
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with mocked OpenAI API calls."""

    @patch("main.OpenAI")
    def test_live_mode_calls_chat_completions(self, mock_openai_class, sample_query_input):
        """Live mode must make exactly one chat.completions.create call."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Alice approved the contract under GDPR Article 17."
        mock_response.usage.total_tokens = 120
        mock_response.model = "gpt-4o-mini"
        mock_client.chat.completions.create.return_value = mock_response

        # Temporarily set API key so run_live is reachable
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "DEMO_MODE": "false"}):
            from main import run_live
            result = run_live(sample_query_input)

        mock_client.chat.completions.create.assert_called_once()
        assert result["tokens_used"] == 120
        assert "Alice" in result["answer"]

    @patch("main.OpenAI")
    def test_live_mode_propagates_api_errors(self, mock_openai_class, sample_query_input):
        """Live mode must propagate API errors to the caller."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "DEMO_MODE": "false"}):
            from main import run_live
            with pytest.raises(Exception, match="Rate limit exceeded"):
                run_live(sample_query_input)


# ---------------------------------------------------------------------------
# TestSampleFiles — validates sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Tests that verify sample input/output files exist and are valid JSON."""

    def test_sample_input_loads(self):
        """load_sample_input must return a dict without raising."""
        data = load_sample_input()
        assert isinstance(data, dict)

    def test_sample_input_has_query_key(self):
        """sample_input.json must contain a 'query' key."""
        data = load_sample_input()
        assert "query" in data, "sample_input.json must have a 'query' field"

    def test_sample_input_has_triples(self):
        """sample_input.json must contain a 'triples' list."""
        data = load_sample_input()
        assert "triples" in data, "sample_input.json must have a 'triples' field"
        assert isinstance(data["triples"], list)
        assert len(data["triples"]) > 0, "triples list must not be empty"

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable JSON with required keys."""
        sample_output_path = Path(__file__).parent.parent / "sample_output.json"
        if sample_output_path.exists():
            data = json.loads(sample_output_path.read_text())
            assert isinstance(data, dict)
            assert "results" in data, "sample_output.json must have a 'results' key"
