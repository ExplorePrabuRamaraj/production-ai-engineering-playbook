"""
W3D7 — Distributed Tracing (LangSmith) — Core Logic
=====================================================
Reusable pipeline functions that demonstrate the @traceable span pattern.
Each function represents one logical step in a RAG pipeline.

In live mode: functions emit spans to LangSmith automatically via the decorator.
In demo mode: the same functions run but span emission is a no-op (no API key).
"""

import time
import uuid
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures representing what each span captures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDocument:
    """Represents one document returned by the retriever span."""
    doc_id: str
    content: str
    score: float          # Relevance score from the vector store (0.0–1.0)
    source: str = ""      # Document source / filename for audit trail
    last_updated: str = ""  # ISO date — used to detect stale documents


@dataclass
class PipelineResult:
    """The structured output of the full RAG pipeline run."""
    run_id: str
    answer: str
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    model: str = "demo"
    spans_captured: int = 0   # How many child spans were emitted


# ---------------------------------------------------------------------------
# Simulated retriever (used in demo mode and as a stand-in for a real vector store)
# ---------------------------------------------------------------------------

DEMO_DOCUMENTS = [
    RetrievedDocument(
        doc_id="policy-001",
        content="Standard return window is 30 days from purchase date for all items.",
        score=0.91,
        source="return_policy_v3.md",
        last_updated="2024-11-01",
    ),
    RetrievedDocument(
        doc_id="policy-002",
        content="Electronics must be returned within 15 days and in original packaging.",
        score=0.84,
        source="electronics_policy_v2.md",
        last_updated="2024-10-15",
    ),
    RetrievedDocument(
        doc_id="policy-003",
        content="Refunds are issued within 5 business days to the original payment method.",
        score=0.77,
        source="refund_process_v1.md",
        last_updated="2024-09-01",
    ),
]


def retrieve_documents(query: str, top_k: int = 3) -> list[RetrievedDocument]:
    """
    Simulate a vector store retrieval step.

    In a real pipeline this would call Chroma, Pinecone, or similar.
    Returns top_k documents ranked by mock relevance score.
    This function is decorated with @traceable in live mode (see main.py).
    """
    # Simulate retrieval latency
    time.sleep(0.01)

    # Return the demo documents, filtered to top_k
    return DEMO_DOCUMENTS[:top_k]


def rerank_documents(
    query: str, docs: list[RetrievedDocument]
) -> list[RetrievedDocument]:
    """
    Simulate a cross-encoder reranker.

    In a real pipeline this would call a reranking model (e.g., Cohere Rerank).
    Here we simply sort by score descending to simulate reranking behaviour.
    This function is a separate span — its inputs (pre-rank) and outputs
    (post-rank) are both captured, making rank inversions visible in traces.
    """
    time.sleep(0.005)
    return sorted(docs, key=lambda d: d.score, reverse=True)


def assemble_context(docs: list[RetrievedDocument], max_chars: int = 800) -> str:
    """
    Assemble retrieved documents into a context string for the LLM prompt.

    Truncates to max_chars to avoid token budget overruns.
    Captures the assembled context as span output — essential for
    diagnosing cases where the LLM had the right documents but
    the assembled prompt cut off the key information.
    """
    context_parts = []
    total = 0
    for doc in docs:
        entry = f"[{doc.source}] {doc.content}"
        if total + len(entry) > max_chars:
            break
        context_parts.append(entry)
        total += len(entry)
    return "\n".join(context_parts)


def validate_answer(answer: str, docs: list[RetrievedDocument]) -> dict:
    """
    Deterministic guardrail: check that the answer does not introduce
    content not present in any retrieved document.

    Returns a dict with 'passed' (bool) and 'reason' (str).
    This is a separate span so validator bugs are visible independently
    of LLM generation bugs — a common source of confusion in production.
    """
    if not answer.strip():
        return {"passed": False, "reason": "Empty answer"}

    # Simple heuristic: check at least one content fragment appears in answer
    # In production, replace with an LLM-as-a-Judge grounding check
    doc_content_combined = " ".join(d.content.lower() for d in docs)
    answer_words = set(answer.lower().split())
    doc_words = set(doc_content_combined.split())

    # Overlap ratio as a crude grounding proxy
    overlap = len(answer_words & doc_words) / max(len(answer_words), 1)
    passed = overlap >= 0.3

    return {
        "passed": passed,
        "overlap_ratio": round(overlap, 3),
        "reason": "Sufficient overlap with retrieved context" if passed
                  else "Answer content diverges from retrieved documents",
    }


# ---------------------------------------------------------------------------
# Demo-mode pipeline runner (no API calls, no tracing)
# ---------------------------------------------------------------------------

def run_demo_pipeline(query: str) -> PipelineResult:
    """
    Execute the full RAG pipeline in demo mode.
    All steps run with local mock data; no external calls are made.
    Span capture is simulated — span count is tracked manually.
    """
    run_id = str(uuid.uuid4())
    start = time.time()
    spans = 0

    # Step 1: Retrieve
    docs = retrieve_documents(query, top_k=3)
    spans += 1  # retriever span

    # Step 2: Rerank
    ranked_docs = rerank_documents(query, docs)
    spans += 1  # reranker span

    # Step 3: Assemble context
    context = assemble_context(ranked_docs)
    spans += 1  # assemble_context span

    # Step 4: Generate (demo answer — no LLM call)
    answer = (
        "Based on the return policy, standard items can be returned within "
        "30 days. Electronics must be returned within 15 days in original "
        "packaging. Refunds are processed within 5 business days."
    )
    spans += 1  # llm_call span

    # Step 5: Validate
    validation = validate_answer(answer, ranked_docs)
    spans += 1  # validate_answer span

    latency_ms = round((time.time() - start) * 1000, 2)

    return PipelineResult(
        run_id=run_id,
        answer=answer,
        retrieved_docs=ranked_docs,
        tokens_used=87,       # Pre-computed estimate for demo
        latency_ms=latency_ms,
        model="demo",
        spans_captured=spans,
    )
