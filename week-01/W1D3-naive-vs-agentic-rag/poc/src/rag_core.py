"""
W1D3 — Naive vs. Agentic RAG — Core Logic
==========================================
Contains the three building blocks of agentic RAG:
  - QueryDecomposer: breaks a complex query into atomic sub-questions
  - ChunkRetriever: wraps a vector store with similarity-threshold filtering
  - AgenticRAGPipeline: orchestrates the decompose → retrieve → validate → synthesise loop

All classes support demo mode (no API key required).
"""
import json
import math
from dataclasses import dataclass, field
from typing import Optional

from config import Config, load_config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single document chunk with its similarity score."""
    text: str
    source: str
    similarity: float


@dataclass
class SubQuestionResult:
    """Evidence gathered for one sub-question."""
    sub_question: str
    chunks: list[RetrievedChunk]
    answerable: bool
    confidence: str  # "high" | "medium" | "low" | "none"


@dataclass
class RAGResult:
    """Final output of either the naive or agentic pipeline."""
    query: str
    answer: str
    sub_questions: list[str]
    retrieval_calls: int
    chunks_used: int
    pipeline: str  # "naive" | "agentic"


# ---------------------------------------------------------------------------
# Demo corpus — a minimal in-memory document store for offline testing
# ---------------------------------------------------------------------------

DEMO_CORPUS: list[dict] = [
    {
        "id": "refund-policy-001",
        "text": (
            "Standard members must report missing or undelivered orders within 7 days "
            "of the expected delivery date to qualify for a replacement or refund."
        ),
        "keywords": ["refund", "missing", "undelivered", "report", "7 days", "standard"],
    },
    {
        "id": "gold-membership-001",
        "text": (
            "Gold members receive an extended 30-day dispute window for lost or undelivered "
            "orders and qualify for expedited replacement shipping at no additional charge."
        ),
        "keywords": ["gold", "membership", "30 days", "dispute", "expedited", "replacement"],
    },
    {
        "id": "shipping-faq-001",
        "text": (
            "Orders are typically delivered within 3-5 business days. Delivery estimates "
            "may vary during peak seasons. You can track your order via the order status page."
        ),
        "keywords": ["shipping", "delivery", "track", "business days", "estimate"],
    },
    {
        "id": "account-reset-001",
        "text": (
            "To reset your account password, click 'Forgot Password' on the login page "
            "and follow the instructions sent to your registered email address."
        ),
        "keywords": ["reset", "password", "account", "login", "forgot", "email"],
    },
    {
        "id": "return-policy-001",
        "text": (
            "Items may be returned within 30 days of purchase in original condition. "
            "Gold members receive free return shipping labels."
        ),
        "keywords": ["return", "30 days", "purchase", "condition", "gold", "free"],
    },
]


def _demo_similarity(query: str, chunk_text: str) -> float:
    """
    Lightweight keyword-overlap similarity for demo mode.
    Returns a float in [0.0, 1.0] — not a true cosine score, but sufficient
    to demonstrate the threshold filtering logic without a real encoder.
    """
    query_tokens = set(query.lower().split())
    chunk_tokens = set(chunk_text.lower().split())
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    # Jaccard-inspired score, scaled to roughly match cosine similarity range
    score = overlap / math.sqrt(len(query_tokens) * len(chunk_tokens))
    return min(score * 3.5, 1.0)  # scale factor tuned for demo corpus


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """
    Breaks a complex query into a list of atomic sub-questions.
    Each sub-question is narrow enough to have a single best-matching chunk.

    In demo mode, uses keyword heuristics to decide whether to decompose.
    In live mode, uses an LLM call with a structured decomposition prompt.
    """

    # Indicators that a query likely spans multiple topics
    MULTI_HOP_SIGNALS = [
        " and ", " also ", " as well as ", " both ", " additionally ",
        "what are my options", "does my membership", "how does my account",
    ]

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()

    def decompose(self, query: str) -> list[str]:
        """Return a list of sub-questions. Single-hop queries return a 1-element list."""
        if self.config.demo_mode:
            return self._demo_decompose(query)
        return self._live_decompose(query)

    def _demo_decompose(self, query: str) -> list[str]:
        """
        Heuristic decomposition for demo mode.
        Detects multi-hop signals and splits on common conjunction patterns.
        """
        q_lower = query.lower()
        is_multi_hop = any(signal in q_lower for signal in self.MULTI_HOP_SIGNALS)

        if not is_multi_hop:
            return [query]

        # Simple split on " and " to produce two sub-questions
        parts = query.split(" and ", 1)
        if len(parts) == 2:
            sub1 = parts[0].strip().rstrip("?") + "?"
            sub2 = parts[1].strip().rstrip("?") + "?"
            # Capitalise the second sub-question since it was mid-sentence
            sub2 = sub2[0].upper() + sub2[1:]
            return [sub1, sub2]

        return [query]

    def _live_decompose(self, query: str) -> list[str]:
        """LLM-based decomposition. Requires OPENAI_API_KEY."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            prompt = (
                f"Break the following user query into a list of at most "
                f"{self.config.max_sub_questions} atomic sub-questions, each answerable "
                f"by searching a single document.\n\n"
                f"Query: {query}\n\n"
                f"Output a JSON array of strings only. No explanation."
            )
            response = client.chat.completions.create(
                model=self.config.decomposition_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            sub_questions = json.loads(raw)
            # Guard: enforce list of strings, cap at max_sub_questions
            if isinstance(sub_questions, list):
                return [str(q) for q in sub_questions[: self.config.max_sub_questions]]
        except Exception:
            pass
        # Fallback to demo decomposition if LLM call fails
        return self._demo_decompose(query)


# ---------------------------------------------------------------------------
# ChunkRetriever
# ---------------------------------------------------------------------------

class ChunkRetriever:
    """
    Wraps a document corpus with similarity search and threshold filtering.

    In demo mode, uses keyword-overlap similarity against DEMO_CORPUS.
    In live mode, calls a real vector store (stub shown — replace with
    your FAISS/Pinecone/Weaviate client).
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self._call_count = 0  # tracks total retrieval calls for reporting

    @property
    def call_count(self) -> int:
        return self._call_count

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        """
        Return up to top_k chunks above the similarity threshold.
        Returns an empty list if no chunks meet the threshold — the caller
        must handle this as a retrieval failure, not silently use empty evidence.
        """
        k = top_k or self.config.top_k
        self._call_count += 1

        if self.config.demo_mode:
            return self._demo_retrieve(query, k)
        return self._live_retrieve(query, k)

    def _demo_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scored = []
        for doc in DEMO_CORPUS:
            score = _demo_similarity(query, doc["text"])
            scored.append(RetrievedChunk(
                text=doc["text"],
                source=doc["id"],
                similarity=round(score, 3),
            ))
        # Sort by similarity descending, apply threshold, take top_k
        scored.sort(key=lambda c: c.similarity, reverse=True)
        return [c for c in scored[:top_k] if c.similarity >= self.config.similarity_threshold]

    def _live_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        Stub for a real vector store client.
        Replace with your FAISS, Pinecone, or Weaviate retrieval call.
        Must return List[RetrievedChunk] with real similarity scores.
        """
        # Example structure (not executed — requires real vector store):
        # results = vector_store.query(
        #     query_texts=[query], n_results=top_k,
        #     where={"tenant_id": self.config.tenant_id}  # always filter by tenant
        # )
        raise NotImplementedError(
            "Live retrieval requires a vector store client. "
            "Set DEMO_MODE=true to run without one."
        )


# ---------------------------------------------------------------------------
# AgenticRAGPipeline
# ---------------------------------------------------------------------------

class AgenticRAGPipeline:
    """
    Orchestrates the full agentic RAG loop:
      1. Decompose query into sub-questions
      2. Retrieve evidence per sub-question
      3. Validate evidence (similarity threshold + semantic check)
      4. Reformulate and retry on validation failure
      5. Synthesise final answer from validated evidence

    Compare with NaiveRAGPipeline which skips steps 1, 3, and 4.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.decomposer = QueryDecomposer(config=self.config)
        self.retriever = ChunkRetriever(config=self.config)

    def run(self, query: str) -> RAGResult:
        """Execute the full agentic pipeline and return a RAGResult."""
        sub_questions = self.decomposer.decompose(query)
        evidence: list[SubQuestionResult] = []

        for sq in sub_questions:
            result = self._retrieve_with_retry(sq)
            evidence.append(result)

        answer = self._synthesise(query, evidence)
        chunks_used = sum(len(e.chunks) for e in evidence if e.answerable)

        return RAGResult(
            query=query,
            answer=answer,
            sub_questions=sub_questions,
            retrieval_calls=self.retriever.call_count,
            chunks_used=chunks_used,
            pipeline="agentic",
        )

    def _retrieve_with_retry(self, sub_question: str) -> SubQuestionResult:
        """
        Attempt retrieval for a sub-question with up to max_reformulation_retries retries.
        Returns a SubQuestionResult indicating whether answerable evidence was found.
        """
        current_query = sub_question
        for attempt in range(self.config.max_reformulation_retries + 1):
            chunks = self.retriever.retrieve(current_query)
            if chunks:
                confidence = "high" if chunks[0].similarity >= 0.80 else "medium"
                return SubQuestionResult(
                    sub_question=sub_question,
                    chunks=chunks,
                    answerable=True,
                    confidence=confidence,
                )
            # Reformulate by broadening the query (remove question word for demo)
            current_query = self._reformulate(current_query, attempt)

        # All retries exhausted — mark as unanswerable
        return SubQuestionResult(
            sub_question=sub_question,
            chunks=[],
            answerable=False,
            confidence="none",
        )

    def _reformulate(self, query: str, attempt: int) -> str:
        """
        Simple reformulation strategy for demo mode:
        strip question words and punctuation to broaden the search.
        """
        stop_words = {"what", "how", "when", "where", "why", "is", "are", "do", "does", "?"}
        tokens = [t for t in query.lower().split() if t not in stop_words]
        return " ".join(tokens)

    def _synthesise(self, query: str, evidence: list[SubQuestionResult]) -> str:
        """
        Produce the final answer from validated evidence.
        In demo mode, assembles a templated response from chunk texts.
        In live mode, calls the synthesis LLM with a structured prompt.
        """
        if self.config.demo_mode:
            return self._demo_synthesise(query, evidence)
        return self._live_synthesise(query, evidence)

    def _demo_synthesise(self, query: str, evidence: list[SubQuestionResult]) -> str:
        """Assemble answer from retrieved chunk texts without an LLM call."""
        parts = []
        citation_index = 1
        for ev in evidence:
            if ev.answerable and ev.chunks:
                top_chunk = ev.chunks[0]
                parts.append(f"{top_chunk.text} [{citation_index}]")
                citation_index += 1
            else:
                parts.append(
                    f"(No reliable information found for: '{ev.sub_question}')"
                )
        return " ".join(parts)

    def _live_synthesise(self, query: str, evidence: list[SubQuestionResult]) -> str:
        """LLM-based synthesis. Requires OPENAI_API_KEY."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            evidence_block = ""
            for i, ev in enumerate(evidence, 1):
                if ev.answerable and ev.chunks:
                    evidence_block += (
                        f"Evidence [{i}] (confidence: {ev.confidence}):\n"
                        f"{ev.chunks[0].text}\n\n"
                    )
                else:
                    evidence_block += (
                        f"Evidence [{i}]: No reliable information found for "
                        f"'{ev.sub_question}'\n\n"
                    )
            prompt = (
                f"Answer the user query using ONLY the evidence below.\n"
                f"Cite evidence inline using [1], [2] notation.\n"
                f"If evidence for a sub-question is missing, state that explicitly.\n\n"
                f"--- RETRIEVED EVIDENCE START ---\n{evidence_block}"
                f"--- RETRIEVED EVIDENCE END ---\n\n"
                f"User query: {query}"
            )
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Synthesis failed: {exc}]"


# ---------------------------------------------------------------------------
# NaiveRAGPipeline — for side-by-side comparison
# ---------------------------------------------------------------------------

class NaiveRAGPipeline:
    """
    Single-pass retrieval: embed query → fetch top-k → call LLM once.
    No decomposition, no validation, no retry loop.
    Included for direct comparison with AgenticRAGPipeline.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.retriever = ChunkRetriever(config=self.config)

    def run(self, query: str) -> RAGResult:
        chunks = self.retriever.retrieve(query)
        answer = self._generate(query, chunks)
        return RAGResult(
            query=query,
            answer=answer,
            sub_questions=[query],
            retrieval_calls=self.retriever.call_count,
            chunks_used=len(chunks),
            pipeline="naive",
        )

    def _generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if self.config.demo_mode:
            if chunks:
                return f"{chunks[0].text} [1]"
            return "No relevant information found."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            context = "\n\n".join(c.text for c in chunks)
            prompt = (
                f"Answer the user query using only the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Query: {query}"
            )
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return f"[Generation failed: {exc}]"
