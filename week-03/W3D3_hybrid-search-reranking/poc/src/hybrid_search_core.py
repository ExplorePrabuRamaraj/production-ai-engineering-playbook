"""
W3D3 — Hybrid Search & Reranking — Core Logic
===============================================
Reusable, side-effect-free functions implementing:
  1. BM25 sparse retrieval
  2. Dense retrieval (bi-encoder cosine similarity)
  3. Reciprocal Rank Fusion (RRF)
  4. Cross-encoder reranking via FlashRank

All functions are independently testable and importable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single document in the corpus."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """One ranked result from a single retriever."""
    doc_id: str
    text: str
    score: float
    retriever: str  # "bm25" | "dense" | "rrf" | "reranker"


@dataclass
class HybridSearchResult:
    """Final result returned to the caller after all pipeline stages."""
    doc_id: str
    text: str
    rrf_score: float
    rerank_score: float | None
    bm25_rank: int | None   # Original rank in BM25 list (None if not retrieved)
    dense_rank: int | None  # Original rank in dense list (None if not retrieved)


# ---------------------------------------------------------------------------
# Stage 1a: BM25 sparse retrieval
# ---------------------------------------------------------------------------

def bm25_retrieve(
    query: str,
    documents: list[Document],
    top_k: int = 100,
) -> list[RetrievalResult]:
    """
    Retrieve documents using BM25 (Best Match 25) scoring.

    BM25 excels at exact-match queries: error codes, product IDs, version strings.
    It operates on raw term frequency with saturation and length normalisation.

    Uses rank-bm25 library (BM25Okapi variant) with default k1=1.5, b=0.75.
    Falls back to a simple TF-based scorer if rank-bm25 is not installed.
    """
    corpus_texts = [doc.text for doc in documents]

    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        tokenised_corpus = [text.lower().split() for text in corpus_texts]
        bm25 = BM25Okapi(tokenised_corpus)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

    except ImportError:
        # Fallback: simple TF scoring without IDF — sufficient for demo purposes
        query_tokens = set(query.lower().split())
        scores = []
        for text in corpus_texts:
            doc_tokens = text.lower().split()
            tf = sum(1 for t in doc_tokens if t in query_tokens)
            scores.append(float(tf))

    # Rank by score descending, return top_k
    ranked = sorted(
        zip(range(len(documents)), scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        RetrievalResult(
            doc_id=documents[idx].id,
            text=documents[idx].text,
            score=score,
            retriever="bm25",
        )
        for idx, score in ranked
        if score > 0  # BM25 score of 0 means no query term overlap
    ]


# ---------------------------------------------------------------------------
# Stage 1b: Dense retrieval (cosine similarity, no external vector DB)
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def dense_retrieve(
    query: str,
    documents: list[Document],
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    top_k: int = 100,
) -> list[RetrievalResult]:
    """
    Retrieve documents using dense cosine similarity.

    Embeddings are pre-computed and passed in — this function is pure
    and does not call any embedding API itself. Caller is responsible
    for generating embeddings (live or mock).

    Dense retrieval excels at semantic/conceptual queries and paraphrases.
    It fails on exact-match queries where the embedding averages out rare terms.
    """
    scores = [
        (i, _cosine_similarity(query_embedding, doc_emb))
        for i, doc_emb in enumerate(doc_embeddings)
    ]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    return [
        RetrievalResult(
            doc_id=documents[idx].id,
            text=documents[idx].text,
            score=score,
            retriever="dense",
        )
        for idx, score in ranked
    ]


# ---------------------------------------------------------------------------
# Stage 2: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_n: int = 50,
) -> list[tuple[str, float, dict]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over all lists of 1 / (k + rank(d))
    where rank is 1-indexed and k=60 is the Cormack (2009) smoothing constant.

    RRF is preferred over weighted linear combination because:
    - It is parameter-free (no score normalisation needed)
    - BM25 scores are unbounded; cosine scores are bounded 0-1
      mixing them raw skews results — RRF avoids this entirely
    - It is robust to outlier top-1 results via the k smoothing term

    Returns: list of (doc_id, rrf_score, {retriever: rank}) tuples
    """
    # Accumulate RRF scores and track per-retriever rank
    rrf_scores: dict[str, float] = {}
    rank_tracking: dict[str, dict[str, int]] = {}
    text_lookup: dict[str, str] = {}

    for ranked_list in ranked_lists:
        retriever_name = ranked_list[0].retriever if ranked_list else "unknown"
        for rank_0based, result in enumerate(ranked_list):
            doc_id = result.doc_id
            rank_1based = rank_0based + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank_1based)
            if doc_id not in rank_tracking:
                rank_tracking[doc_id] = {}
            rank_tracking[doc_id][retriever_name] = rank_1based
            text_lookup[doc_id] = result.text

    # Sort by RRF score descending and return top_n
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        (doc_id, score, rank_tracking.get(doc_id, {}), text_lookup.get(doc_id, ""))
        for doc_id, score in sorted_docs
    ]


# ---------------------------------------------------------------------------
# Stage 3: Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank_with_flashrank(
    query: str,
    candidates: list[tuple[str, float, dict, str]],  # (doc_id, rrf_score, ranks, text)
    top_k: int = 5,
    model_name: str = "ms-marco-MiniLM-L-6-v2",
) -> list[HybridSearchResult]:
    """
    Rerank fusion candidates using a cross-encoder via FlashRank.

    Cross-encoders outperform bi-encoders on precision because they process
    the query and document jointly — the model can attend to specific
    query-relevant phrases within the document.

    FlashRank uses MiniLM-based cross-encoders optimised for CPU inference.
    Falls back to returning RRF-ordered results if FlashRank is unavailable.
    """
    try:
        from flashrank import Ranker, RerankRequest  # type: ignore

        ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
        passages = [{"id": doc_id, "text": text} for doc_id, _, _, text in candidates]
        request = RerankRequest(query=query, passages=passages)
        reranked = ranker.rerank(request)

        # Build lookup for RRF metadata
        meta_lookup = {doc_id: (rrf_score, ranks) for doc_id, rrf_score, ranks, _ in candidates}

        results = []
        for item in reranked[:top_k]:
            doc_id = item["id"]
            rrf_score, ranks = meta_lookup.get(doc_id, (0.0, {}))
            results.append(HybridSearchResult(
                doc_id=doc_id,
                text=item["text"],
                rrf_score=rrf_score,
                rerank_score=item.get("score"),
                bm25_rank=ranks.get("bm25"),
                dense_rank=ranks.get("dense"),
            ))
        return results

    except ImportError:
        # Graceful fallback: return RRF-ordered results without reranking
        return _rrf_fallback(candidates, top_k)


def _rrf_fallback(
    candidates: list[tuple[str, float, dict, str]],
    top_k: int,
) -> list[HybridSearchResult]:
    """Return top-K results ordered by RRF score (no cross-encoder reranking)."""
    return [
        HybridSearchResult(
            doc_id=doc_id,
            text=text,
            rrf_score=rrf_score,
            rerank_score=None,
            bm25_rank=ranks.get("bm25"),
            dense_rank=ranks.get("dense"),
        )
        for doc_id, rrf_score, ranks, text in candidates[:top_k]
    ]


# ---------------------------------------------------------------------------
# Pipeline orchestrator: all three stages in sequence
# ---------------------------------------------------------------------------

def hybrid_search_pipeline(
    query: str,
    documents: list[Document],
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    bm25_top_k: int = 100,
    dense_top_k: int = 100,
    rrf_k: int = 60,
    fusion_top_n: int = 50,
    reranker_top_k: int = 5,
    use_reranker: bool = True,
    reranker_model: str = "ms-marco-MiniLM-L-6-v2",
) -> list[HybridSearchResult]:
    """
    Full three-stage hybrid search pipeline.

    Stage 1: Parallel BM25 + dense retrieval
    Stage 2: RRF fusion
    Stage 3: Cross-encoder reranking (optional, falls back gracefully)
    """
    # Stage 1: parallel retrieval
    bm25_results = bm25_retrieve(query, documents, top_k=bm25_top_k)
    dense_results = dense_retrieve(
        query, documents, query_embedding, doc_embeddings, top_k=dense_top_k
    )

    # Stage 2: RRF fusion
    fused = reciprocal_rank_fusion(
        [bm25_results, dense_results], k=rrf_k, top_n=fusion_top_n
    )

    # Stage 3: reranking (or fallback)
    if use_reranker:
        return rerank_with_flashrank(query, fused, top_k=reranker_top_k, model_name=reranker_model)
    else:
        return _rrf_fallback(fused, reranker_top_k)
