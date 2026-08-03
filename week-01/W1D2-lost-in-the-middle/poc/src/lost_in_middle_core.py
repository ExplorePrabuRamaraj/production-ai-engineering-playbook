"""
Core logic for Lost-in-the-Middle context position analysis.

Simulates how transformer attention distributes non-uniformly across
context positions (U-shaped curve) and provides three document ordering
strategies to maximise the effective utilisation of retrieved content.

Reference: Liu et al. (2023). arXiv:2307.03172
"""
import math
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """A retrieved document with its ground-truth relevance score and context position."""
    id: str
    text: str
    relevance_score: float  # Similarity score in [0.0, 1.0] from the vector store
    position: int = 0       # Assigned position in the assembled context window


def u_shaped_attention_weight(position: int, total_docs: int) -> float:
    """
    Model the U-shaped attention pattern observed in transformer LLMs.

    Based on Liu et al. (2023): models attend more strongly to tokens at
    the start and end of the context window than to tokens in the middle.

    The curve uses a scaled cosine: weight = alpha + (1-alpha)|cos(pi * i/(N-1))|
    which gives weight = 1.0 at the edges and ~0.4 at the middle.

    Args:
        position:   0-based index of the document in the context window
        total_docs: total number of documents in the context

    Returns:
        Normalised attention weight in [0.4, 1.0]
    """
    if total_docs <= 1:
        return 1.0
    relative_pos = position / (total_docs - 1)
    weight = 0.4 + 0.6 * abs(math.cos(math.pi * relative_pos))
    return round(weight, 4)


def _assign_positions(docs: List[Document]) -> List[Document]:
    """Return a new list of Documents with 0-based position indices assigned."""
    return [
        Document(id=d.id, text=d.text, relevance_score=d.relevance_score, position=i)
        for i, d in enumerate(docs)
    ]


def naive_ordering(docs: List[Document]) -> List[Document]:
    """Return documents in their original retrieval order — the baseline to beat."""
    return _assign_positions(list(docs))


def relevance_sorted_ordering(docs: List[Document]) -> List[Document]:
    """
    Sort documents by relevance score descending (most relevant at position 0).

    Better than naive but still places 2nd-best at position 1, 3rd at position 2,
    etc. — leaving high-relevance documents in the middle for larger K.
    """
    sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
    return _assign_positions(sorted_docs)


def lost_in_middle_aware_ordering(docs: List[Document]) -> List[Document]:
    """
    Place the highest-relevance documents at context boundaries (positions 0
    and N-1) where transformer attention is highest.

    Algorithm:
      1. Sort documents by relevance score descending.
      2. Even-ranked documents (0, 2, 4...) fill from the start of the context.
      3. Odd-ranked documents (1, 3, 5...) fill from the end backwards.

    Result: top-2 documents occupy positions 0 and N-1; low-relevance documents
    accumulate in the middle dead zone where their impact is smallest.
    """
    sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
    left: List[Document] = []
    right: List[Document] = []
    for i, doc in enumerate(sorted_docs):
        if i % 2 == 0:
            left.append(doc)
        else:
            right.append(doc)
    return _assign_positions(left + right[::-1])


def compute_effective_scores(docs: List[Document]) -> List[dict]:
    """
    Compute effective retrieval score = relevance_score x attention_weight.

    This estimates how much of each document's content the LLM will actually
    utilise, given its position in the context window.

    Args:
        docs: Documents with position indices already assigned

    Returns:
        List of dicts with id, position, relevance_score, attention_weight,
        and effective_score for each document
    """
    n = len(docs)
    results = []
    for doc in docs:
        attention = u_shaped_attention_weight(doc.position, n)
        effective = round(doc.relevance_score * attention, 4)
        results.append({
            "id": doc.id,
            "position": doc.position,
            "relevance_score": doc.relevance_score,
            "attention_weight": attention,
            "effective_score": effective,
        })
    return results


def summarise_effectiveness(scores: List[dict]) -> dict:
    """Return mean, min, and max effective retrieval score across all documents."""
    values = [r["effective_score"] for r in scores]
    return {
        "mean_effective_score": round(sum(values) / len(values), 4),
        "min_effective_score": round(min(values), 4),
        "max_effective_score": round(max(values), 4),
    }
