"""
W2D2 — KV Caching & Token Trimming — Unit Tests
=================================================
Run: pytest tests/ -v

All external API calls are mocked so tests pass completely offline.
Tests are grouped into four classes per the studio code-generation rules.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — allow importing from src/ without installation
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_caching_core import (
    count_tokens,
    count_tokens_approx,
    count_messages_tokens,
    trim_to_budget,
    compute_eviction_ratio,
    prepare_context,
    inject_summary,
    build_compression_summary,
    is_system_message,
    is_tool_related,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_conversation():
    """A small conversation that fits within any reasonable budget."""
    return [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
        {"role": "user",      "content": "What about 3 + 3?"},
        {"role": "assistant", "content": "3 + 3 equals 6."},
    ]


@pytest.fixture
def long_conversation():
    """A conversation that exceeds a tight token budget."""
    turns = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(20):
        turns.append({"role": "user",      "content": f"Question number {i}: What is {i} times {i}?"})
        turns.append({"role": "assistant", "content": f"Answer {i}: {i} times {i} is {i*i}."})
    return turns


@pytest.fixture
def tool_conversation():
    """A conversation containing a tool_call / tool_result pair."""
    return [
        {"role": "system",    "content": "You are a tool-using assistant."},
        {"role": "user",      "content": "What is the weather in London?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "t1", "function": {"name": "get_weather"}}]},
        {"role": "tool",      "content": '{"temp": 15, "condition": "cloudy"}'},
        {"role": "assistant", "content": "It is 15°C and cloudy in London."},
        {"role": "user",      "content": "Thanks!"},
    ]


@pytest.fixture
def expected_output_keys():
    return {"messages", "original_tokens", "final_tokens", "eviction_ratio", "summary_injected"}


# ---------------------------------------------------------------------------
# Class 1: TestDemoMode — offline demo mode, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Tests for the offline demo mode — must pass without any API key."""

    def test_demo_returns_expected_schema(self, short_conversation):
        """run_demo output must contain all required result keys."""
        from main import run_demo
        input_data = {
            "system_prompt": "You are a helpful assistant.",
            "conversation": short_conversation[1:],  # exclude system msg
        }
        result = run_demo(input_data)
        required = {"original_tokens", "final_tokens", "eviction_ratio",
                    "summary_injected", "messages_before", "messages_after", "model"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_demo_model_is_demo(self, short_conversation):
        """Demo mode must not claim to use a real model."""
        from main import run_demo
        result = run_demo({"system_prompt": "You are helpful.", "conversation": []})
        assert result["model"] == "demo"

    def test_demo_final_tokens_within_budget(self, long_conversation):
        """After demo trimming, final token count must be within the given budget."""
        from main import run_demo
        # Budget of 600 is tight enough to force eviction of most turns in
        # long_conversation (41 messages, ~1400+ tokens) while still being above
        # the irreducible minimum (system message + 1–2 turns ≈ 30–50 tokens).
        budget = 600
        input_data = {
            "system_prompt": "You are a helpful assistant.",
            "conversation": long_conversation[1:],
        }
        result = run_demo(input_data, budget=budget)
        assert result["final_tokens"] <= budget, (
            f"final_tokens={result['final_tokens']} exceeds budget of {budget}"
        )

    def test_demo_messages_after_lte_messages_before(self, long_conversation):
        """Trimming must never increase the number of messages."""
        from main import run_demo
        result = run_demo({
            "system_prompt": "System.",
            "conversation": long_conversation[1:],
        }, budget=150)
        assert result["messages_after"] <= result["messages_before"]


# ---------------------------------------------------------------------------
# Class 2: TestCoreConcept — pure function behaviour, no I/O
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for core trimming and counting logic — pure functions, offline."""

    def test_count_tokens_approx_nonempty(self):
        """Approximate token count must be positive for non-empty strings."""
        assert count_tokens_approx("Hello, world!") > 0

    def test_count_tokens_approx_empty(self):
        """Approximate token count returns at least 1 (minimum floor)."""
        assert count_tokens_approx("") == 1

    def test_count_tokens_returns_int(self):
        """count_tokens must always return an int."""
        result = count_tokens("some text here", model="gpt-4o-mini")
        assert isinstance(result, int)

    def test_count_messages_tokens_sum(self, short_conversation):
        """Total tokens across messages must be greater than any single message."""
        total = count_messages_tokens(short_conversation)
        single = count_tokens(short_conversation[1]["content"])
        assert total > single

    def test_system_message_never_evicted(self, long_conversation):
        """System messages must survive trimming regardless of budget pressure."""
        trimmed, _, _ = trim_to_budget(long_conversation, budget=50)
        system_msgs = [m for m in trimmed if m["role"] == "system"]
        assert len(system_msgs) >= 1, "System message was evicted — must never happen"

    def test_trim_to_budget_result_within_budget(self, long_conversation):
        """After trimming, token count must be <= the specified budget."""
        budget = 100
        trimmed, _, final_count = trim_to_budget(long_conversation, budget=budget)
        assert final_count <= budget, (
            f"final_count={final_count} exceeds budget={budget}"
        )

    def test_trim_does_not_exceed_original(self, short_conversation):
        """Trimming a conversation already within budget must not increase token count."""
        original_count = count_messages_tokens(short_conversation)
        _, original, final = trim_to_budget(short_conversation, budget=10000)
        assert final <= original

    def test_compute_eviction_ratio_full(self):
        """Full eviction returns ratio of 1.0."""
        ratio = compute_eviction_ratio(original_count=100, final_count=0)
        assert ratio == 1.0

    def test_compute_eviction_ratio_none(self):
        """No eviction returns ratio of 0.0."""
        ratio = compute_eviction_ratio(original_count=100, final_count=100)
        assert ratio == 0.0

    def test_compute_eviction_ratio_zero_original(self):
        """Zero original count must not raise ZeroDivisionError."""
        ratio = compute_eviction_ratio(original_count=0, final_count=0)
        assert ratio == 0.0

    def test_inject_summary_adds_system_message(self, short_conversation):
        """inject_summary must add a system message containing the summary text."""
        result = inject_summary(short_conversation, summary="Key facts: Paris is the capital.")
        system_msgs = [m for m in result if m["role"] == "system"]
        assert any("Paris" in m["content"] for m in system_msgs)

    def test_build_compression_summary_empty(self):
        """Empty evicted list must return empty string."""
        assert build_compression_summary([]) == ""

    def test_build_compression_summary_nonempty(self, short_conversation):
        """Non-empty evicted list must return a non-empty summary string."""
        result = build_compression_summary(short_conversation)
        assert len(result) > 0

    def test_is_system_message_true(self):
        assert is_system_message({"role": "system", "content": "instructions"}) is True

    def test_is_system_message_false(self):
        assert is_system_message({"role": "user", "content": "hello"}) is False

    def test_tool_pair_evicted_atomically(self, tool_conversation):
        """
        If the oldest turn is a tool_call, its paired tool_result must also be evicted.
        This prevents orphaned tool_results from appearing in the context.
        """
        # Use a very tight budget to force eviction of the tool pair
        budget = 30  # Forces eviction of most of the conversation
        trimmed, _, _ = trim_to_budget(tool_conversation, budget=budget)

        tool_call_msgs  = [m for m in trimmed if m.get("tool_calls")]
        tool_result_msgs = [m for m in trimmed if m.get("role") == "tool"]

        # After trimming, tool_call and tool_result counts must be equal
        assert len(tool_call_msgs) == len(tool_result_msgs), (
            "Orphaned tool_call or tool_result detected after trimming"
        )

    @pytest.mark.parametrize("budget,expected_within", [
        (50,   True),
        (100,  True),
        (500,  True),
        (10,   True),   # Very tight — may leave only system message but must not exceed
        (5000, True),   # Generous — no eviction needed
    ])
    def test_trim_respects_various_budgets(self, long_conversation, budget, expected_within):
        """trim_to_budget must respect the budget for a range of budget values."""
        _, _, final_count = trim_to_budget(long_conversation, budget=budget)
        assert (final_count <= budget) == expected_within

    def test_prepare_context_returns_all_keys(self, expected_output_keys, short_conversation):
        """prepare_context must return all required keys in its result dict."""
        result = prepare_context(short_conversation, budget=10000)
        assert expected_output_keys.issubset(result.keys())

    def test_prepare_context_no_eviction_when_within_budget(self, short_conversation):
        """No eviction should occur when the conversation fits within the budget."""
        result = prepare_context(short_conversation, budget=10000)
        assert result["eviction_ratio"] == 0.0
        assert not result["summary_injected"]

    def test_prepare_context_summary_injected_on_large_eviction(self, long_conversation):
        """Summary must be injected when eviction ratio exceeds compression_threshold."""
        result = prepare_context(long_conversation, budget=80, compression_threshold=0.1)
        # With budget=80 and compression_threshold=0.1, almost any eviction triggers compression
        if result["eviction_ratio"] >= 0.1:
            assert result["summary_injected"] is True


# ---------------------------------------------------------------------------
# Class 3: TestLiveMode — live mode with all API calls mocked
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for live mode with mocked OpenAI API calls."""

    @patch("main.OpenAI")
    def test_live_mode_calls_openai(self, mock_openai_class):
        """Live mode must make exactly one API call to OpenAI."""
        mock_client   = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content  = "The answer is 42."
        mock_response.usage.total_tokens          = 55
        mock_response.model                       = "gpt-4o-mini"
        mock_client.chat.completions.create.return_value = mock_response

        import main
        result = main.run_live({
            "system_prompt": "You are helpful.",
            "conversation":  [{"role": "user", "content": "What is 6 times 7?"}],
        })

        mock_client.chat.completions.create.assert_called_once()
        assert result["llm_response"] == "The answer is 42."
        assert result["tokens_used"]  == 55

    @patch("main.OpenAI")
    def test_live_mode_trims_before_api_call(self, mock_openai_class):
        """Live mode must trim messages before submitting to the API."""
        mock_client   = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content  = "Trimmed."
        mock_response.usage.total_tokens          = 20
        mock_response.model                       = "gpt-4o-mini"
        mock_client.chat.completions.create.return_value = mock_response

        import main
        # Build a long conversation that must be trimmed.
        # Each user message is ~20 tokens; 10 messages = ~200 tokens.
        # Budget=120 is below total but above the irreducible minimum
        # (system "System." = ~6 tokens + a few turns ≈ 50–80 tokens).
        conversation = [
            {"role": "user", "content": f"Long message number {i} " + "x" * 50}
            for i in range(10)
        ]
        result = main.run_live(
            {"system_prompt": "System.", "conversation": conversation},
            budget=120,
        )
        assert result["final_tokens"] <= 120

    @patch("main.OpenAI")
    def test_live_mode_propagates_api_error(self, mock_openai_class):
        """Live mode must propagate API errors to the caller."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

        import main
        with pytest.raises(Exception, match="Rate limit exceeded"):
            main.run_live({"system_prompt": "System.", "conversation": []})


# ---------------------------------------------------------------------------
# Class 4: TestSampleFiles — validates sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Tests that verify sample input/output files are valid JSON with correct schema."""

    def test_sample_input_is_valid_json(self):
        """sample_input.json must be parseable."""
        path = Path(__file__).parent.parent / "sample_input.json"
        assert path.exists(), "sample_input.json is missing"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_sample_input_has_required_keys(self):
        """sample_input.json must contain 'system_prompt' and 'conversation'."""
        path = Path(__file__).parent.parent / "sample_input.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "system_prompt"  in data, "Missing 'system_prompt' key"
        assert "conversation"   in data, "Missing 'conversation' key"
        assert isinstance(data["conversation"], list)

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable."""
        path = Path(__file__).parent.parent / "sample_output.json"
        assert path.exists(), "sample_output.json is missing"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_sample_output_has_required_keys(self):
        """sample_output.json must contain the expected result fields."""
        path = Path(__file__).parent.parent / "sample_output.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"original_tokens", "final_tokens", "eviction_ratio", "model"}
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"

    def test_sample_input_conversation_has_messages(self):
        """sample_input.json conversation must contain at least one message."""
        path = Path(__file__).parent.parent / "sample_input.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["conversation"]) > 0, "conversation must have at least one message"
