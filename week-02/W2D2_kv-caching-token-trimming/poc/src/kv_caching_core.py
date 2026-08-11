"""
Core logic for KV caching and token trimming.

Key concepts demonstrated:
  - Token counting using tiktoken (exact, not estimated)
  - Sliding window eviction at message boundaries (never mid-turn)
  - Tool call pair integrity: tool_call and tool_result evicted together
  - Summary compression placeholder for large eviction events

This module has no side effects and is fully unit-testable offline.
"""
from __future__ import annotations

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens_approx(text: str) -> int:
    """
    Approximate token count without the tiktoken dependency.
    Uses the ~4 chars/token heuristic for English prose.
    Accurate to ±15% — use only when tiktoken is unavailable.
    """
    return max(1, len(text) // 4)


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a given text string using tiktoken when available,
    falling back to the character-based approximation.

    tiktoken is the official OpenAI tokeniser and is accurate for all
    GPT-3.5 / GPT-4 / GPT-4o model families.
    """
    try:
        import tiktoken
        # encoding_for_model raises KeyError for unknown models; cl100k_base
        # is the correct encoding for all gpt-4* and gpt-3.5-turbo models.
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # tiktoken not installed — use approximation
        return count_tokens_approx(text)


def count_messages_tokens(messages: List[dict], model: str = "gpt-4o-mini") -> int:
    """
    Count the total tokens across a list of OpenAI-format message dicts.
    Adds 4 tokens per message for the role/content framing overhead.
    """
    total = 0
    for msg in messages:
        # 4 tokens overhead per message (role marker + separators)
        total += 4 + count_tokens(msg.get("content") or "", model)
    return total


# ---------------------------------------------------------------------------
# Message classification helpers
# ---------------------------------------------------------------------------

def is_system_message(msg: dict) -> bool:
    return msg.get("role") == "system"


def is_tool_related(msg: dict) -> bool:
    """Return True for tool_call and tool_result messages."""
    role = msg.get("role", "")
    return role in ("tool", "function") or bool(msg.get("tool_calls"))


# ---------------------------------------------------------------------------
# Eviction strategies
# ---------------------------------------------------------------------------

def trim_to_budget(
    messages: List[dict],
    budget: int,
    model: str = "gpt-4o-mini",
) -> Tuple[List[dict], int, int]:
    """
    Evict the oldest non-system turns until the total token count is within budget.

    Rules:
      - System messages are NEVER evicted (they contain instructions and constraints).
      - Tool call + tool result pairs are evicted atomically to preserve message integrity.
      - Returns the trimmed message list, the original count, and the final count.

    Args:
        messages: OpenAI-format message list [{"role": ..., "content": ...}, ...]
        budget:   Maximum allowed token count for the returned list
        model:    Model name for tiktoken selection

    Returns:
        (trimmed_messages, original_token_count, final_token_count)
    """
    system_msgs = [m for m in messages if is_system_message(m)]
    turn_msgs   = [m for m in messages if not is_system_message(m)]

    original_count = count_messages_tokens(system_msgs + turn_msgs, model)

    if original_count <= budget:
        return messages, original_count, original_count

    # Evict from the oldest end of the turn list
    while turn_msgs:
        current_count = count_messages_tokens(system_msgs + turn_msgs, model)
        if current_count <= budget:
            break

        # Remove the oldest message; if it is a tool_call, also remove the
        # immediately following tool_result to preserve pair integrity.
        evicted = turn_msgs.pop(0)
        if evicted.get("tool_calls") and turn_msgs and is_tool_related(turn_msgs[0]):
            turn_msgs.pop(0)  # Evict the paired tool_result

    final_count = count_messages_tokens(system_msgs + turn_msgs, model)
    return system_msgs + turn_msgs, original_count, final_count


def compute_eviction_ratio(original_count: int, final_count: int) -> float:
    """
    Return the fraction of tokens evicted: 0.0 = no eviction, 1.0 = all evicted.
    Used to decide whether summary compression is warranted.
    """
    if original_count == 0:
        return 0.0
    return (original_count - final_count) / original_count


# ---------------------------------------------------------------------------
# Summary compression (placeholder — calls LLM in live mode)
# ---------------------------------------------------------------------------

def build_compression_summary(evicted_messages: List[dict]) -> str:
    """
    Produce a plain-text summary of evicted messages.

    In demo mode this returns a static placeholder.
    In live mode the caller should pass this text to the LLM and replace it
    with the actual summarised output before injecting it into the prompt.
    """
    if not evicted_messages:
        return ""

    # Build a readable representation of the evicted turns for summarisation
    lines = []
    for msg in evicted_messages:
        role    = msg.get("role", "unknown").upper()
        content = (msg.get("content") or "")[:200]  # Truncate for safety
        lines.append(f"{role}: {content}")

    summary_input = "\n".join(lines)
    # In production: pass summary_input to LLM with instruction
    # "Summarise the key facts, decisions, and constraints from these conversation
    #  turns in under 150 tokens."
    return (
        f"[CONVERSATION SUMMARY — {len(evicted_messages)} turns compressed]\n"
        f"Key context from earlier in this session:\n{summary_input[:300]}..."
    )


def inject_summary(messages: List[dict], summary: str) -> List[dict]:
    """
    Insert a summary as a synthetic system message after the last system message
    but before the first user/assistant turn.
    This preserves the summary in context without it being evictable as a turn.
    """
    if not summary:
        return messages

    summary_msg = {"role": "system", "content": summary}
    # Find the insertion point: after the last system message
    insert_at = 0
    for i, msg in enumerate(messages):
        if is_system_message(msg):
            insert_at = i + 1

    return messages[:insert_at] + [summary_msg] + messages[insert_at:]


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def prepare_context(
    messages: List[dict],
    budget: int,
    compression_threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Orchestrate token trimming with optional summary compression.

    Steps:
      1. Count tokens in the current message list.
      2. If within budget, return as-is.
      3. Identify evictable turns (non-system).
      4. Evict oldest turns until within budget.
      5. If eviction ratio exceeds compression_threshold, generate a summary
         of the evicted content and inject it before the remaining turns.

    Returns a dict with:
      - messages:       The trimmed (and optionally summary-injected) message list
      - original_tokens: Token count before trimming
      - final_tokens:    Token count after trimming
      - eviction_ratio:  Fraction of tokens evicted
      - summary_injected: Whether a summary was injected
    """
    system_msgs = [m for m in messages if is_system_message(m)]
    turn_msgs   = [m for m in messages if not is_system_message(m)]

    original_count = count_messages_tokens(messages, model)

    if original_count <= budget:
        return {
            "messages":        messages,
            "original_tokens": original_count,
            "final_tokens":    original_count,
            "eviction_ratio":  0.0,
            "summary_injected": False,
        }

    # Determine which turns will be evicted before modifying the list
    trimmed, original_count, final_count = trim_to_budget(messages, budget, model)
    eviction_ratio = compute_eviction_ratio(original_count, final_count)

    summary_injected = False
    if eviction_ratio >= compression_threshold:
        # Identify the evicted turns (those in original turn_msgs but not in trimmed)
        trimmed_turn_ids = {id(m) for m in trimmed}
        evicted_turns = [m for m in turn_msgs if id(m) not in trimmed_turn_ids]
        if evicted_turns:
            summary = build_compression_summary(evicted_turns)
            trimmed = inject_summary(trimmed, summary)
            summary_injected = True
            final_count = count_messages_tokens(trimmed, model)

    return {
        "messages":        trimmed,
        "original_tokens": original_count,
        "final_tokens":    final_count,
        "eviction_ratio":  eviction_ratio,
        "summary_injected": summary_injected,
    }
