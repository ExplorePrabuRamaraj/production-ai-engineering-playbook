"""
W3D2 — Context Compression Core Logic
======================================
Reusable, testable compression functions.
Three strategies are implemented behind a unified interface:
  - extractive: TF-IDF sentence scoring (no API call, CPU-only)
  - abstractive: LLM-based summarisation (requires API key)
  - hybrid: extractive first, abstractive if still over budget
"""

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float   # 0.0–1.0; lower = more compressed
    strategy_used: str


# ---------------------------------------------------------------------------
# Token estimation
# Approximates token count without requiring tiktoken as a hard dependency.
# Rule of thumb: 1 token ≈ 4 characters for English text (OpenAI guidance).
# In production, replace with: import tiktoken; enc.encode(text)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count using the 4-chars-per-token heuristic."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Sentence splitting
# Handles common abbreviations to avoid splitting "Dr. Smith" into two sentences.
# ---------------------------------------------------------------------------

_ABBREVIATIONS = r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\."

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences while preserving abbreviations.
    Returns list of non-empty sentence strings.
    """
    # Temporarily mask abbreviation periods so they don't trigger splits
    masked = re.sub(_ABBREVIATIONS, lambda m: m.group().replace(".", "<DOT>"), text)
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", masked)
    # Restore masked periods and strip whitespace
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (extractive compression)
# Pure Python + math — no scikit-learn required for the core PoC.
# ---------------------------------------------------------------------------

def _term_frequencies(text: str) -> dict[str, float]:
    """Compute normalised term frequency for a piece of text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {}
    freq: dict[str, float] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    total = len(words)
    return {w: c / total for w, c in freq.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two term-frequency dictionaries."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = sum(v ** 2 for v in vec_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in vec_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extractive_compress(text: str, query: str, token_budget: int) -> CompressionResult:
    """
    Score each sentence by TF-IDF cosine similarity to the query.
    Greedily retain the highest-scoring sentences until the token budget
    is reached. Reassemble in original order to preserve narrative flow.

    Args:
        text: Input text to compress.
        query: The current user query — compression is query-aware.
        token_budget: Maximum tokens allowed in the output.

    Returns:
        CompressionResult with the compressed text and metrics.
    """
    original_tokens = estimate_tokens(text)
    sentences = split_sentences(text)

    if not sentences:
        return CompressionResult(
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            strategy_used="extractive",
        )

    query_tf = _term_frequencies(query)

    # Score each sentence against the query
    scored = []
    for idx, sentence in enumerate(sentences):
        score = _cosine_similarity(_term_frequencies(sentence), query_tf)
        scored.append((score, idx, sentence))

    # Sort descending by relevance score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedily select sentences within budget
    selected_indices: set[int] = set()
    used_tokens = 0
    for score, idx, sentence in scored:
        sent_tokens = estimate_tokens(sentence)
        if used_tokens + sent_tokens <= token_budget:
            selected_indices.add(idx)
            used_tokens += sent_tokens
        if used_tokens >= token_budget:
            break

    # Reassemble in original document order (not score order)
    retained = [sentences[i] for i in sorted(selected_indices)]
    compressed_text = " ".join(retained)
    compressed_tokens = estimate_tokens(compressed_text)

    return CompressionResult(
        compressed_text=compressed_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(compressed_tokens / original_tokens, 3) if original_tokens else 1.0,
        strategy_used="extractive",
    )


# ---------------------------------------------------------------------------
# Abstractive compression (LLM summarisation)
# Requires an OpenAI client. For demo mode, returns a static summary.
# ---------------------------------------------------------------------------

def abstractive_compress(
    text: str,
    query: str,
    token_budget: int,
    openai_client=None,
    model: str = "gpt-4o-mini",
) -> CompressionResult:
    """
    Use a language model to produce a condensed, query-conditioned summary.
    Falls back to extractive compression if no client is provided (demo mode).

    Args:
        text: Input text to compress.
        query: Current user query — summary is conditioned on this.
        token_budget: Target maximum tokens for the output.
        openai_client: An initialised OpenAI client (or None for demo mode).
        model: Model to use for summarisation.

    Returns:
        CompressionResult with the compressed text and metrics.
    """
    original_tokens = estimate_tokens(text)

    # Fall back to extractive in demo/offline mode
    if openai_client is None:
        result = extractive_compress(text, query, token_budget)
        result.strategy_used = "abstractive-demo-fallback"
        return result

    system_prompt = (
        "You are a precise summariser. Preserve all named entities, "
        "numerical values, dates, and decisions. Omit pleasantries, "
        "repetitions, and off-topic content. Respond only with the summary."
    )
    user_prompt = (
        f"Summarise the following text to answer this query: {query}\n\n"
        f"Text:\n{text}\n\n"
        f"Respond in at most {token_budget} tokens."
    )

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=token_budget,
    )

    compressed_text = response.choices[0].message.content.strip()
    compressed_tokens = estimate_tokens(compressed_text)

    return CompressionResult(
        compressed_text=compressed_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(compressed_tokens / original_tokens, 3) if original_tokens else 1.0,
        strategy_used="abstractive",
    )


# ---------------------------------------------------------------------------
# Unified compression interface
# ---------------------------------------------------------------------------

def compress_context(
    text: str,
    query: str,
    token_budget: int,
    strategy: str = "extractive",
    openai_client=None,
    model: str = "gpt-4o-mini",
    min_segment_tokens: int = 50,
) -> CompressionResult:
    """
    Compress a text segment using the specified strategy.
    Bypasses compression when the segment is already within budget
    or below the minimum segment size threshold.

    Args:
        text: Input segment to compress.
        query: Current user query for relevance scoring.
        token_budget: Token ceiling for the compressed output.
        strategy: "extractive", "abstractive", or "hybrid".
        openai_client: OpenAI client (required for abstractive; None = demo).
        model: LLM model name for abstractive compression.
        min_segment_tokens: Segments smaller than this bypass compression.

    Returns:
        CompressionResult — even if no compression was applied.
    """
    original_tokens = estimate_tokens(text)

    # Bypass: segment already within budget or too small to bother
    if original_tokens <= token_budget or original_tokens < min_segment_tokens:
        return CompressionResult(
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            strategy_used="bypass",
        )

    if strategy == "abstractive":
        return abstractive_compress(text, query, token_budget, openai_client, model)

    if strategy == "hybrid":
        # First pass: extractive (free, fast)
        result = extractive_compress(text, query, token_budget)
        # Second pass: if still over budget, apply abstractive
        if result.compressed_tokens > token_budget and openai_client:
            result = abstractive_compress(
                result.compressed_text, query, token_budget, openai_client, model
            )
            result.strategy_used = "hybrid"
        return result

    # Default: extractive
    return extractive_compress(text, query, token_budget)
