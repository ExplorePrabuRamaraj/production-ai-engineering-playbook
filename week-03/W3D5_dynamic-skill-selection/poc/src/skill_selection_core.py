"""
W3D5 — Dynamic Skill Selection — Core Module
=============================================
Provides SkillRegistry, EmbeddingRouter, and SkillInjector.
All three are designed to be imported and used independently.

In demo mode, embeddings are pre-computed mock vectors so the module
runs entirely offline with no API key required.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Top-level import so unit tests can patch "skill_selection_core.OpenAI".
# Wrapped in try/except so the module loads in demo mode without the package.
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """A registered agent capability with its metadata and embedding."""
    name: str
    description: str
    schema: dict                          # OpenAI function-calling schema
    required_roles: Set[str]              # Empty set = available to all roles
    embedding: Optional[List[float]] = None
    last_called_turn: int = 0             # For eviction tracking
    times_called: int = 0


@dataclass
class SelectionResult:
    """Output of a single skill selection step."""
    selected_skills: List[Skill]
    query_embedding: List[float]
    scores: Dict[str, float]              # skill_name -> cosine similarity score
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python — no numpy required for demo)
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors of equal length."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Mock embeddings for demo mode
# Demo vectors are hand-crafted unit vectors in a 6-dimensional space.
# Real embeddings (text-embedding-3-small) are 1536-dimensional.
# ---------------------------------------------------------------------------

# Each dimension loosely represents: [billing, network, account, security, it, general]
_MOCK_SKILL_EMBEDDINGS: Dict[str, List[float]] = {
    "get_invoice":           [0.90, 0.05, 0.20, 0.10, 0.05, 0.10],
    "process_refund":        [0.85, 0.05, 0.25, 0.15, 0.05, 0.10],
    "check_network_speed":   [0.10, 0.92, 0.10, 0.05, 0.20, 0.05],
    "run_ping_diagnostic":   [0.05, 0.95, 0.05, 0.05, 0.15, 0.05],
    "reset_password":        [0.10, 0.10, 0.50, 0.80, 0.70, 0.10],
    "provision_access":      [0.10, 0.10, 0.60, 0.75, 0.65, 0.10],
    "create_it_ticket":      [0.10, 0.25, 0.30, 0.20, 0.90, 0.20],
    "general_response":      [0.20, 0.20, 0.20, 0.20, 0.20, 0.95],
}

_MOCK_QUERY_EMBEDDINGS: Dict[str, List[float]] = {
    "billing":   [0.88, 0.05, 0.15, 0.10, 0.05, 0.10],
    "network":   [0.08, 0.91, 0.08, 0.05, 0.18, 0.05],
    "password":  [0.08, 0.08, 0.45, 0.85, 0.68, 0.08],
    "it_ticket": [0.08, 0.22, 0.28, 0.18, 0.92, 0.18],
    "unknown":   [0.15, 0.15, 0.15, 0.15, 0.15, 0.90],
}

def _classify_demo_query(query: str) -> List[float]:
    """Return a mock embedding vector based on keyword signals in the query."""
    q = query.lower()
    if any(w in q for w in ["invoice", "charge", "bill", "payment", "refund"]):
        return _MOCK_QUERY_EMBEDDINGS["billing"]
    if any(w in q for w in ["slow", "speed", "ping", "latency", "network", "internet"]):
        return _MOCK_QUERY_EMBEDDINGS["network"]
    if any(w in q for w in ["password", "login", "access", "locked", "reset"]):
        return _MOCK_QUERY_EMBEDDINGS["password"]
    if any(w in q for w in ["ticket", "support", "broken", "not working", "help"]):
        return _MOCK_QUERY_EMBEDDINGS["it_ticket"]
    return _MOCK_QUERY_EMBEDDINGS["unknown"]


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """
    Central store for all registered agent capabilities.

    In production, call embed_all() once at startup to compute real embeddings.
    In demo mode, mock embeddings are assigned from the hardcoded dictionary above.
    """

    def __init__(self, demo_mode: bool = True):
        self._skills: Dict[str, Skill] = {}
        self._demo_mode = demo_mode
        self._current_turn: int = 0

    def register(
        self,
        name: str,
        description: str,
        schema: dict,
        required_roles: Optional[Set[str]] = None,
    ) -> None:
        """Register a new skill. Assigns a mock embedding in demo mode."""
        embedding = _MOCK_SKILL_EMBEDDINGS.get(name) if self._demo_mode else None
        self._skills[name] = Skill(
            name=name,
            description=description,
            schema=schema,
            required_roles=required_roles or set(),
            embedding=embedding,
        )

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def log_call(self, skill_name: str) -> None:
        """Record that a skill was actually invoked this turn."""
        if skill_name in self._skills:
            self._skills[skill_name].last_called_turn = self._current_turn
            self._skills[skill_name].times_called += 1

    def advance_turn(self) -> None:
        self._current_turn += 1

    def evict_stale(self, eviction_threshold: int = 50) -> List[str]:
        """
        Mark skills as inactive if unused for eviction_threshold turns.
        Returns the names of evicted skills.
        Evicted skills are removed from the registry — re-register to restore.
        """
        evicted = []
        for name, skill in list(self._skills.items()):
            turns_since_call = self._current_turn - skill.last_called_turn
            if turns_since_call > eviction_threshold and skill.times_called > 0:
                del self._skills[name]
                evicted.append(name)
        return evicted

    def __len__(self) -> int:
        return len(self._skills)


# ---------------------------------------------------------------------------
# Embedding Router
# ---------------------------------------------------------------------------

class EmbeddingRouter:
    """
    Routes a user query to the most relevant registered skills using
    cosine similarity between the query embedding and skill embeddings.

    In demo mode, uses _classify_demo_query() for query embedding.
    In live mode, calls the OpenAI Embeddings API.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        top_k: int = 5,
        similarity_threshold: float = 0.35,
        fallback_skills: Optional[List[str]] = None,
        demo_mode: bool = True,
        api_key: str = "",
        embedding_model: str = "text-embedding-3-small",
    ):
        self._registry = registry
        self._top_k = top_k
        self._threshold = similarity_threshold
        self._fallback_names = fallback_skills or ["general_response"]
        self._demo_mode = demo_mode
        self._api_key = api_key
        self._embedding_model = embedding_model

    def _embed_query(self, query: str) -> List[float]:
        """Return an embedding vector for the query string."""
        if self._demo_mode:
            return _classify_demo_query(query)
        # Live mode: call OpenAI Embeddings API (OpenAI imported at module top)
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install -r requirements.txt")
        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(
            model=self._embedding_model,
            input=query,
        )
        return response.data[0].embedding

    def select(
        self,
        query: str,
        user_roles: Optional[Set[str]] = None,
    ) -> SelectionResult:
        """
        Select the top-k skills most relevant to the query, filtered by user roles.

        Permission filtering happens AFTER similarity scoring to avoid leaking
        information about which tools exist for which roles via timing side-channels.
        """
        user_roles = user_roles or set()
        query_vec = self._embed_query(query)

        # Score every registered skill
        scores: Dict[str, float] = {}
        for skill in self._registry.all_skills():
            if skill.embedding is None:
                continue
            scores[skill.name] = _cosine_similarity(query_vec, skill.embedding)

        # Sort by score descending, apply threshold, then filter by role
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates: List[Tuple[str, float]] = []
        for name, score in ranked:
            if score < self._threshold:
                break  # Remaining scores are below threshold
            skill = self._registry.get(name)
            if skill is None:
                continue
            # Empty required_roles means accessible to all
            if skill.required_roles and not skill.required_roles.intersection(user_roles):
                continue
            candidates.append((name, score))
            if len(candidates) >= self._top_k:
                break

        used_fallback = False
        if not candidates:
            # Activate fallback skill set when no relevant skills found
            used_fallback = True
            for fname in self._fallback_names:
                skill = self._registry.get(fname)
                if skill:
                    candidates.append((fname, 0.0))

        selected_skills = [
            self._registry.get(name)
            for name, _ in candidates
            if self._registry.get(name) is not None
        ]

        return SelectionResult(
            selected_skills=selected_skills,
            query_embedding=query_vec,
            scores=scores,
            used_fallback=used_fallback,
        )


# ---------------------------------------------------------------------------
# Skill Injector
# ---------------------------------------------------------------------------

class SkillInjector:
    """
    Serialises a list of selected Skill objects into the OpenAI
    function-calling format ready for prompt injection.
    """

    @staticmethod
    def build_tool_block(skills: List[Skill]) -> List[dict]:
        """Return a list of tool dicts in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": skill.schema,
                },
            }
            for skill in skills
        ]
