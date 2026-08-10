"""
W2D1 — Type-Safe Schemas with Pydantic AI — Core Logic
=======================================================
Defines the output schemas and extraction helpers used by main.py.

Key design decisions:
- extra="forbid" on all models: rejects unexpected fields (partial prompt-injection defence)
- Enum for categorical fields: the allowed values appear in the JSON Schema injected into
  the prompt, dramatically reducing the model's tendency to invent new category names
- Validators are pure functions with no I/O: safe to run synchronously inside Pydantic
- Error messages are written as model instructions, not internal codes, because the
  message is fed verbatim to the LLM on retry
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Enums — categorical fields with a fixed, small set of valid values
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Output schemas — each is a locked form the LLM must fill in exactly
# ---------------------------------------------------------------------------

class ReviewAnalysis(BaseModel):
    """Structured extraction from a product review.

    Used to demonstrate field-level validation and enum constraints.
    The schema is injected into the LLM prompt so the model knows
    exactly what fields and value types are required.
    """

    # extra="forbid" rejects any fields the model adds beyond the schema.
    # This prevents schema drift from silently introducing unintended fields
    # and provides a partial defence against prompt injection via output.
    model_config = ConfigDict(extra="forbid")

    sentiment: Sentiment
    confidence: float          # Expected range: 0.0 – 1.0
    key_topics: list[str]      # At least one topic required
    summary: str               # Max 150 characters

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_range(cls, v: float) -> float:
        # Error message is an instruction, not a code — it is fed to the LLM on retry
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be a decimal between 0.0 and 1.0")
        return round(v, 3)

    @field_validator("key_topics")
    @classmethod
    def key_topics_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("key_topics must contain at least one topic string")
        return [topic.strip() for topic in v if topic.strip()]

    @field_validator("summary")
    @classmethod
    def summary_must_fit_display(cls, v: str) -> str:
        if len(v) > 150:
            raise ValueError(
                f"summary must be 150 characters or fewer; received {len(v)} characters"
            )
        return v.strip()


class SupportTicketTriage(BaseModel):
    """Structured triage record extracted from a support ticket.

    Demonstrates enum constraints for routing fields where a wrong value
    causes a routing miss rather than a visible error.
    """

    model_config = ConfigDict(extra="forbid")

    urgency: UrgencyLevel
    department: str            # Free-form but required — no default
    refund_involved: bool
    one_line_summary: str      # Max 100 characters

    @field_validator("one_line_summary")
    @classmethod
    def summary_length_check(cls, v: str) -> str:
        if len(v) > 100:
            raise ValueError(
                f"one_line_summary must be 100 characters or fewer; received {len(v)}"
            )
        return v.strip()


# ---------------------------------------------------------------------------
# Demo fixtures — pre-computed outputs that mirror live mode results
# ---------------------------------------------------------------------------

DEMO_REVIEW_OUTPUT = ReviewAnalysis(
    sentiment=Sentiment.POSITIVE,
    confidence=0.92,
    key_topics=["battery life", "build quality", "value for money"],
    summary="Reviewer is highly satisfied with battery endurance and build, considers it good value.",
)

DEMO_TICKET_OUTPUT = SupportTicketTriage(
    urgency=UrgencyLevel.HIGH,
    department="billing",
    refund_involved=True,
    one_line_summary="Customer charged twice for the same order, requesting immediate refund.",
)
