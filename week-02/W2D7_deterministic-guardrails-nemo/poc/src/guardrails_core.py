"""
W2D7 — Deterministic Guardrails (NeMo) — Core Logic
=====================================================
Reusable functions and data structures that implement:
  - Input rail evaluation (pattern-based blocking)
  - Output rail evaluation (vocabulary + required-token checks)
  - A lightweight canonical flow state tracker

This module has no side effects and no external API calls.
All functions are pure and fully testable offline.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RailResponse:
    """Result returned by any rail that evaluates a message or response."""
    blocked: bool               # True = this rail stops processing
    modified: bool              # True = output was altered (disclaimer injected)
    reply: Optional[str]        # Canned response text when blocked=True
    reason: str                 # Audit-log reason code (e.g. "competitor_mention")
    rail_name: str              # Which rail fired
    original: str = ""          # The original text evaluated
    final: str = ""             # The final text after modification (if modified=True)


@dataclass
class GuardrailsResult:
    """Final result returned to the application after all rail evaluation."""
    response: str
    blocked: bool
    modified: bool
    rails_evaluated: List[str] = field(default_factory=list)
    rails_fired: List[str] = field(default_factory=list)
    latency_ms: int = 0


@dataclass
class FlowState:
    """
    Tracks position in a canonical multi-step dialogue flow.
    In production NeMo Guardrails, this is managed by the Colang runtime.
    Here we implement a simplified version to demonstrate the concept.
    """
    flow_name: str
    required_steps: List[str]
    completed_steps: List[str] = field(default_factory=list)

    def next_pending_step(self) -> Optional[str]:
        """Return the next step that has not yet been completed, or None."""
        for step in self.required_steps:
            if step not in self.completed_steps:
                return step
        return None

    def is_complete(self) -> bool:
        """True when all required steps have been completed."""
        return all(step in self.completed_steps for step in self.required_steps)

    def complete_step(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)


# ---------------------------------------------------------------------------
# Unicode normalisation — apply before pattern matching to defeat homoglyph attacks
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Normalise Unicode to NFKC form and lowercase.
    This defeats homoglyph substitution attacks (e.g. 'ign0re' → 'ignore').
    Applied to all inputs before pattern evaluation.
    """
    return unicodedata.normalize("NFKC", text).lower()


# ---------------------------------------------------------------------------
# Input rail evaluation
# ---------------------------------------------------------------------------

# Scripted canned responses keyed by reason code
_INPUT_CANNED_RESPONSES = {
    "jailbreak_framing":
        "I can only respond to genuine questions about our products and services.",
    "competitor_mention":
        "I can only discuss our own products and services. "
        "For competitor comparisons, please visit an independent review site.",
    "default_block":
        "I'm not able to help with that request. "
        "Please rephrase or contact support for assistance.",
}


def evaluate_input_rails(
    message: str,
    blocked_patterns: List[str],
) -> RailResponse:
    """
    Evaluate all input rail patterns against the normalised user message.
    Returns the first blocking RailResponse, or a pass result if none match.

    Each pattern in blocked_patterns is a case-insensitive regex.
    Patterns are evaluated in list order — first match wins (short-circuit).

    Args:
        message: Raw user message text.
        blocked_patterns: List of regex pattern strings to match against.

    Returns:
        RailResponse with blocked=True if a pattern matched, blocked=False otherwise.
    """
    normalised = normalise_text(message)

    for pattern in blocked_patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        if compiled.search(normalised):
            # Classify the reason from the pattern — use a heuristic
            if any(kw in pattern for kw in ["ignore", "hypothetical", "you are now"]):
                reason = "jailbreak_framing"
            elif "competitor" in pattern or "rival" in pattern:
                reason = "competitor_mention"
            else:
                reason = "default_block"

            canned = _INPUT_CANNED_RESPONSES.get(reason, _INPUT_CANNED_RESPONSES["default_block"])
            return RailResponse(
                blocked=True,
                modified=False,
                reply=canned,
                reason=reason,
                rail_name="input_pattern_rail",
                original=message,
                final=canned,
            )

    # No pattern matched — pass
    return RailResponse(
        blocked=False,
        modified=False,
        reply=None,
        reason="pass",
        rail_name="input_pattern_rail",
        original=message,
        final=message,
    )


# ---------------------------------------------------------------------------
# Output rail evaluation
# ---------------------------------------------------------------------------

# Investment vocabulary that triggers the disclaimer requirement
_INVESTMENT_VOCABULARY = [
    "investment", "portfolio", "returns", "dividend",
    "equity", "bond", "index fund", "etf", "stock",
]


def _response_contains_investment_vocabulary(response: str) -> bool:
    """True if the response contains any investment-domain vocabulary."""
    lower = response.lower()
    return any(term in lower for term in _INVESTMENT_VOCABULARY)


def evaluate_output_rails(
    response: str,
    required_tokens: List[str],
    blocked_vocab: List[str],
) -> RailResponse:
    """
    Evaluate output rails against the LLM response text.

    Two checks are performed in sequence:
      1. Blocked vocabulary check — if any blocked phrase is present, block the response.
      2. Required token check — if investment vocabulary is present but required disclaimer
         token is absent, inject the disclaimer.

    Args:
        response: Raw LLM response text.
        required_tokens: Tokens that must be present when investment vocabulary is detected.
        blocked_vocab: Multi-word phrases that must not appear in any response.

    Returns:
        RailResponse with blocked=True, modified=True, or neither (pass).
    """
    lower_response = response.lower()

    # Check 1: blocked vocabulary (hard block — return canned response)
    for phrase in blocked_vocab:
        if phrase.lower() in lower_response:
            return RailResponse(
                blocked=True,
                modified=False,
                reply=(
                    "I can provide general investment information, but I cannot make "
                    "personalised investment recommendations. Please consult a licensed advisor."
                ),
                reason="blocked_recommendation_language",
                rail_name="output_vocabulary_rail",
                original=response,
                final=(
                    "I can provide general investment information, but I cannot make "
                    "personalised investment recommendations. Please consult a licensed advisor."
                ),
            )

    # Check 2: required token injection (soft modification)
    if _response_contains_investment_vocabulary(response):
        for token in required_tokens:
            if token not in response:
                modified_response = response.rstrip() + f"\n\n{token}: This is general information only and does not constitute investment advice. Please consult a licensed financial adviser before making investment decisions."
                return RailResponse(
                    blocked=False,
                    modified=True,
                    reply=None,
                    reason="disclaimer_injected",
                    rail_name="output_disclaimer_rail",
                    original=response,
                    final=modified_response,
                )

    # All output rails passed
    return RailResponse(
        blocked=False,
        modified=False,
        reply=None,
        reason="pass",
        rail_name="output_rail",
        original=response,
        final=response,
    )


# ---------------------------------------------------------------------------
# Canonical flow helpers
# ---------------------------------------------------------------------------

INVESTMENT_FLOW = FlowState(
    flow_name="investment_advice_flow",
    required_steps=["disclosure_presented", "user_acknowledged"],
)

_DISCLOSURE_PROMPT = (
    "Before I provide investment information, please confirm you understand: "
    "this is general information only and not personalised financial advice. "
    "Type 'I understand' to continue."
)

_ACKNOWLEDGMENT_KEYWORDS = ["i understand", "i agree", "understood", "yes", "ok", "okay"]


def get_flow_next_turn(flow: FlowState, user_message: str) -> Tuple[Optional[str], FlowState]:
    """
    Advance the canonical flow state based on the user's latest message.

    Returns a tuple of:
      - bot_turn: The scripted bot message for this step (None if flow is complete).
      - updated_flow: The flow state after processing this turn.

    This demonstrates NeMo Guardrails' canonical flow concept without requiring
    the full Colang runtime.
    """
    if flow.is_complete():
        return None, flow

    next_step = flow.next_pending_step()

    if next_step == "disclosure_presented":
        flow.complete_step("disclosure_presented")
        return _DISCLOSURE_PROMPT, flow

    if next_step == "user_acknowledged":
        if any(kw in user_message.lower() for kw in _ACKNOWLEDGMENT_KEYWORDS):
            flow.complete_step("user_acknowledged")
            return None, flow  # Flow complete — proceed to LLM call
        else:
            # User did not acknowledge — re-present the disclosure
            return (
                "Please type 'I understand' to confirm before I continue. "
                "This is required before discussing investment topics.",
                flow,
            )

    return None, flow
