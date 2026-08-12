"""
graphrag_core.py — Core GraphRAG logic for W2D3.

This module demonstrates the three key steps of GraphRAG:
  1. build_knowledge_graph  — construct an entity relationship graph from chunks
  2. detect_communities     — cluster connected nodes (simplified Leiden proxy)
  3. hybrid_retrieve        — merge vector search results with graph traversal via RRF

All functions are pure (no side effects) and fully typed so they are easy to test
and import into larger pipelines.

Design note: We use networkx.DiGraph for the in-process demo. For corpora with
> 500,000 nodes, replace GraphStore with a Neo4j or Amazon Neptune backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity extracted from a document chunk."""
    id: str          # Normalised lowercase identifier, e.g. "alice_smith"
    label: str       # Display name, e.g. "Alice Smith"
    entity_type: str # Controlled type: Person, Organization, Policy, Product, Event


@dataclass
class Relationship:
    """A directed, typed relationship between two entities."""
    source_id: str       # Entity.id of the source node
    target_id: str       # Entity.id of the target node
    rel_type: str        # Relationship type, e.g. REPORTS_TO, GOVERNED_BY, CAUSES
    weight: int = 1      # Occurrence count — incremented on duplicate extraction
    source_chunk: str = ""  # Chunk ID for provenance tracking


@dataclass
class Community:
    """A cluster of closely connected entities with an LLM-generated summary."""
    community_id: str
    member_ids: list[str]
    summary: str


@dataclass
class RetrievalResult:
    """A single result from the hybrid retrieval pipeline."""
    text: str
    source: str          # "vector" or "graph"
    score: float         # RRF score (higher = more relevant)
    provenance: str = "" # For graph results: traversal path as human-readable string


# ---------------------------------------------------------------------------
# Step 1: Build knowledge graph
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    triples: list[tuple[str, str, str]],
    entity_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build an in-memory knowledge graph from (subject, predicate, object) triples.

    We use a plain dict-based adjacency representation rather than requiring
    networkx as a hard dependency, keeping the module importable in restricted
    environments. The graph dict has two keys:
      - "nodes": {entity_id: Entity}
      - "edges": {(source_id, target_id, rel_type): Relationship}

    Args:
        triples: List of (subject_label, predicate, object_label) strings.
        entity_types: Optional mapping from entity label to type string.
                      Defaults to "Unknown" for unlabelled entities.

    Returns:
        Graph dict with "nodes" and "edges" sub-dicts.
    """
    if entity_types is None:
        entity_types = {}

    nodes: dict[str, Entity] = {}
    edges: dict[tuple[str, str, str], Relationship] = {}

    for subject_label, predicate, object_label in triples:
        # Normalise entity IDs to lowercase-underscore form
        subj_id = subject_label.lower().replace(" ", "_")
        obj_id = object_label.lower().replace(" ", "_")

        # Register nodes if not already present
        if subj_id not in nodes:
            nodes[subj_id] = Entity(
                id=subj_id,
                label=subject_label,
                entity_type=entity_types.get(subject_label, "Unknown"),
            )
        if obj_id not in nodes:
            nodes[obj_id] = Entity(
                id=obj_id,
                label=object_label,
                entity_type=entity_types.get(object_label, "Unknown"),
            )

        # Register or increment edge weight
        edge_key = (subj_id, obj_id, predicate)
        if edge_key in edges:
            edges[edge_key].weight += 1
        else:
            edges[edge_key] = Relationship(
                source_id=subj_id,
                target_id=obj_id,
                rel_type=predicate,
                weight=1,
            )

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Step 2: Community detection (simplified Leiden proxy)
# ---------------------------------------------------------------------------

def detect_communities(
    graph: dict[str, Any],
    gamma: float = 1.0,
    min_edge_weight: int = 1,
) -> list[Community]:
    """
    Partition the graph into communities using a greedy connected-components
    approach as a simplified proxy for the Leiden algorithm.

    Production note: Replace this with the `leidenalg` Python package for true
    Leiden modularity maximisation. This demo version uses connected components
    to illustrate the concept without requiring a graph-science dependency.

    Each connected component becomes one community, with a pre-computed summary
    that describes the entities in that cluster.

    Args:
        graph: Output of build_knowledge_graph().
        gamma: Resolution parameter (ignored in this simplified version;
               included for API compatibility with production Leiden).
        min_edge_weight: Edges with weight < this value are excluded before
                         clustering, filtering likely extraction noise.

    Returns:
        List of Community objects with member IDs and auto-generated summaries.
    """
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Filter low-weight edges before community detection
    active_edges = {
        key: rel for key, rel in edges.items()
        if rel.weight >= min_edge_weight
    }

    # Build adjacency list (undirected for community detection)
    adjacency: dict[str, set[str]] = {nid: set() for nid in nodes}
    for (src, tgt, _), rel in active_edges.items():
        if src in adjacency and tgt in adjacency:
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)

    # BFS to find connected components
    visited: set[str] = set()
    communities: list[Community] = []
    community_counter = 0

    for start_node in nodes:
        if start_node in visited:
            continue
        # BFS from start_node
        component: list[str] = []
        queue = [start_node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            queue.extend(adjacency[current] - visited)

        # Generate a plain-text summary for this community
        entity_labels = [nodes[nid].label for nid in component if nid in nodes]
        summary = _generate_community_summary(entity_labels, community_counter)

        communities.append(Community(
            community_id=f"c{community_counter:03d}",
            member_ids=component,
            summary=summary,
        ))
        community_counter += 1

    return communities


def _generate_community_summary(entity_labels: list[str], community_id: int) -> str:
    """
    Generate a deterministic summary for a community.

    In production, this would be an LLM call with a structured prompt.
    Here we produce a template-based summary so the demo runs offline.
    """
    if not entity_labels:
        return f"Community {community_id}: empty cluster."
    if len(entity_labels) == 1:
        return f"Community {community_id}: single entity — {entity_labels[0]}."
    entity_list = ", ".join(entity_labels[:5])
    suffix = f" and {len(entity_labels) - 5} more" if len(entity_labels) > 5 else ""
    return (
        f"Community {community_id} ({len(entity_labels)} entities): "
        f"Cluster containing {entity_list}{suffix}. "
        f"These entities are densely interconnected in the source corpus."
    )


# ---------------------------------------------------------------------------
# Step 3: Graph traversal
# ---------------------------------------------------------------------------

def traverse_graph(
    graph: dict[str, Any],
    seed_entities: list[str],
    hop_depth: int = 2,
    max_nodes: int = 30,
) -> list[RetrievalResult]:
    """
    Collect the N-hop neighbourhood of seed entities from the graph.

    Each visited node and its connecting edges become a RetrievalResult with
    a provenance string describing the traversal path.

    Args:
        graph: Output of build_knowledge_graph().
        seed_entities: Normalised entity IDs to start traversal from.
        hop_depth: Maximum number of hops to traverse (hard-capped for safety).
        max_nodes: Hard cap on total nodes retrieved (prevents context explosion).

    Returns:
        List of RetrievalResult objects ranked by traversal proximity
        (closer = higher score).
    """
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Build outgoing adjacency: source_id -> [(target_id, rel_type, weight)]
    out_adj: dict[str, list[tuple[str, str, int]]] = {nid: [] for nid in nodes}
    for (src, tgt, rel_type), rel in edges.items():
        if src in out_adj:
            out_adj[src].append((tgt, rel_type, rel.weight))

    visited: dict[str, int] = {}   # entity_id -> hop distance
    results: list[RetrievalResult] = []
    queue: list[tuple[str, int, str]] = []  # (entity_id, depth, path_so_far)

    for seed in seed_entities:
        seed_norm = seed.lower().replace(" ", "_")
        if seed_norm in nodes:
            queue.append((seed_norm, 0, seed_norm))

    while queue and len(results) < max_nodes:
        current_id, depth, path = queue.pop(0)

        if current_id in visited:
            continue
        visited[current_id] = depth

        if current_id in nodes:
            node = nodes[current_id]
            # Score by proximity: nodes closer to seed score higher
            proximity_score = 1.0 / (1.0 + depth)
            results.append(RetrievalResult(
                text=(
                    f"Entity: {node.label} (type: {node.entity_type}). "
                    f"Reached via path: {path.replace('_', ' ')}."
                ),
                source="graph",
                score=proximity_score,
                provenance=path,
            ))

            # Enqueue neighbours if within hop budget
            if depth < hop_depth:
                for tgt_id, rel_type, weight in out_adj.get(current_id, []):
                    if tgt_id not in visited:
                        new_path = f"{path} -[{rel_type}]-> {tgt_id}"
                        queue.append((tgt_id, depth + 1, new_path))

    return results


# ---------------------------------------------------------------------------
# Step 4: RRF merge
# ---------------------------------------------------------------------------

def rrf_merge(
    vector_results: list[RetrievalResult],
    graph_results: list[RetrievalResult],
    k: int = 60,
    top_m: int = 8,
) -> list[RetrievalResult]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    RRF score: sum of 1 / (k + rank_i) across all result sets.
    k=60 is the standard default from Cormack et al. (SIGIR 2009).

    This formula is rank-based, so it is robust to score-scale differences
    between vector cosine similarity (0-1) and graph proximity scores.

    Args:
        vector_results: Results from vector search, pre-sorted by score desc.
        graph_results: Results from graph traversal, pre-sorted by score desc.
        k: RRF smoothing constant.
        top_m: Number of merged results to return.

    Returns:
        Top-m results sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    result_map: dict[str, RetrievalResult] = {}

    def _update(results: list[RetrievalResult]) -> None:
        for rank, result in enumerate(results, start=1):
            key = result.text[:80]  # Use truncated text as deduplication key
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in result_map:
                result_map[key] = result

    _update(sorted(vector_results, key=lambda r: r.score, reverse=True))
    _update(sorted(graph_results, key=lambda r: r.score, reverse=True))

    # Sort by RRF score and return top-m
    sorted_keys = sorted(rrf_scores, key=lambda k_: rrf_scores[k_], reverse=True)
    merged = []
    for key in sorted_keys[:top_m]:
        result = result_map[key]
        merged.append(RetrievalResult(
            text=result.text,
            source=result.source,
            score=round(rrf_scores[key], 6),
            provenance=result.provenance,
        ))
    return merged


# ---------------------------------------------------------------------------
# Step 5: High-level hybrid retrieval entry point
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    graph: dict[str, Any],
    communities: list[Community],
    seed_entities: list[str],
    vector_results: list[RetrievalResult] | None = None,
    hop_depth: int = 2,
    max_nodes: int = 30,
    rrf_k: int = 60,
    top_m: int = 8,
) -> list[RetrievalResult]:
    """
    Run the full hybrid retrieval pipeline: graph traversal + vector merge via RRF.

    In a production system, vector_results would come from a live vector index
    (e.g., LanceDB, Pinecone, pgvector). In the PoC demo, they are pre-supplied
    as static fixtures to enable offline execution.

    Args:
        query: The user's natural language question.
        graph: Output of build_knowledge_graph().
        communities: Output of detect_communities().
        seed_entities: Entity labels extracted from the query.
        vector_results: Pre-ranked vector search results (or None for demo).
        hop_depth: Max hops for graph traversal.
        max_nodes: Max nodes retrieved per traversal.
        rrf_k: RRF smoothing constant.
        top_m: Number of final merged results to return.

    Returns:
        Top-m merged results sorted by RRF score.
    """
    # Graph traversal
    graph_results = traverse_graph(
        graph=graph,
        seed_entities=seed_entities,
        hop_depth=hop_depth,
        max_nodes=max_nodes,
    )

    # Add community summaries for seed entities' communities as additional context
    seed_norms = {e.lower().replace(" ", "_") for e in seed_entities}
    for community in communities:
        if seed_norms.intersection(set(community.member_ids)):
            graph_results.append(RetrievalResult(
                text=community.summary,
                source="graph_community",
                score=0.5,  # Community summaries get a fixed mid-tier score
                provenance=f"community:{community.community_id}",
            ))

    # Fall back to empty vector results if none provided (demo mode)
    if vector_results is None:
        vector_results = []

    return rrf_merge(
        vector_results=vector_results,
        graph_results=graph_results,
        k=rrf_k,
        top_m=top_m,
    )
