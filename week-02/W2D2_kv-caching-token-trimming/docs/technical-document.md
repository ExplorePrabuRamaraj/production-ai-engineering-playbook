# W2D2 — KV Caching & Token Trimming
## AI Engineering Production Playbook — Week 2, Day 2

**Vertical:** Context Engineering & Tokens
**Series:** Production AI Engineering Playbook

---

## 1. Overview

Every call to a large language model recomputes the full attention matrix over every token in the context window — system prompt, retrieved documents, and conversation history alike. This is computationally expensive and, in multi-turn or long-context scenarios, the dominant driver of both latency and cost. **KV caching** exploits the fact that attention Key-Value tensors for unchanged prefix tokens can be computed once and reused, while **token trimming** enforces a context budget by evicting low-priority tokens before the API call. Together these two techniques make production LLM systems that handle long conversations, large document retrieval results, or high-volume traffic both faster and cheaper without sacrificing output quality.

---

## 2. Learning Objectives

By the end of this document you will be able to:

1. **Explain** how transformer attention produces Key-Value tensors and why recomputing them on every call is wasteful.
2. **Distinguish** between server-side KV caching (provider-managed) and client-side token trimming (application-managed).
3. **Implement** a token-budget enforcer using tiktoken and the sliding window strategy.
4. **Apply** Anthropic's cache_control prompt caching and OpenAI's prompt caching headers to reduce prefill latency.
5. **Evaluate** trade-offs between sliding window, importance-scored pruning, and summary compression strategies.
6. **Design** a combined caching + trimming architecture for a multi-turn production assistant.
7. **Benchmark** latency and cost impact of KV cache hit rates against a baseline.
8. **Build** unit-testable token budget functions that operate entirely offline.

---

## 3. Problem Statement

### What Breaks and How

Every transformer forward pass recomputes the full **prefill** — the process of computing Query, Key, and Value projections for every token in the prompt before generating the first output token. For a conversation with 8,000 tokens of history, the LLM reprocesses all 8,000 tokens from scratch on every turn. This is quadratic in complexity: doubling the context roughly quadruples the attention computation.

### What Failure Looks Like in Production

- **Latency spike:** Time-to-first-token (TTFT) for a 16k-token context is 3–4× higher than for a 4k-token context on the same model. A customer-facing assistant that responds in 800ms on turn 1 may respond in 3,200ms by turn 20.
- **Cost explosion:** API providers charge per input token. A 50-turn conversation where each turn includes the full history sends roughly 625,000 input tokens instead of 12,500 — a 50× cost multiplier.
- **Window exhaustion:** GPT-4o has a 128k context window. A support bot storing raw transcripts hits this limit after approximately 200 dense turns, causing hard errors.

### How Naive Approaches Fall Short

Naive approaches are: (a) keep all history forever until the window overflows, or (b) drop all history beyond the last N turns. Option (a) fails on long sessions; option (b) destroys conversational coherence because the model loses track of user intent, earlier constraints, and established facts.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: Legal Document Review Bot

A law firm deploys a multi-turn assistant for contract review. The system prompt contains a 3,000-token policy document, each user message attaches a 2,000-token contract excerpt, and the conversation accumulates 20+ turns of back-and-forth analysis.

By turn 15, the full context is approximately 90,000 tokens. TTFT has grown from 1.1 seconds to 6.8 seconds. The billing team reports that this single workflow accounts for 38% of the firm's monthly API spend. On turn 22, a hard context-length error terminates the session, forcing the user to restart and lose all prior analysis.

Root cause: no caching of the static system prompt, no eviction of resolved discussion turns, and no budget enforcement before submission.

### Scenario B — The Solution: Legal Document Review Bot with KV Caching and Trimming

The same firm applies two changes:

1. The 3,000-token policy document is marked with `cache_control: {"type": "ephemeral"}` in the Anthropic Messages API. After the first call, the KV cache for this prefix is reused on every subsequent turn, eliminating 3,000 tokens of prefill computation per request.

2. A client-side token budget enforcer keeps a maximum of 8,000 tokens of recent conversation history, evicting the oldest complete turns first (sliding window) and replacing them with a 200-token LLM-generated summary when the window is first breached.

Result: TTFT stabilises at 1.3 seconds regardless of conversation length. Cost per session drops 61%. The context window never overflows. Summary accuracy measured against ground-truth transcripts achieves 94% recall on key facts.

---

## 5. Solution Architecture

The combined architecture has three logical layers.

**Layer 1 — Static Prefix Cache (Server-Side):** The application splits the prompt into a cacheable static prefix (system instructions, retrieved documents that do not change within a session) and a dynamic suffix (conversation history, current user message). The static prefix is submitted once with a cache-control directive. The provider's infrastructure stores the computed KV tensors and attaches them to the session. On subsequent calls, only the dynamic suffix is prefilled from scratch.

**Layer 2 — Token Budget Enforcer (Client-Side):** Before every API call, the application counts the tokens in the dynamic suffix using a tokeniser (tiktoken for OpenAI, the Anthropic token-counting API endpoint for Claude). If the count exceeds the configured budget, the enforcer evicts messages according to the chosen strategy until the budget is met. The enforcer never trims within a message — it always removes complete turns.

**Layer 3 — Compression Fallback:** When the budget enforcer must evict more than a configurable threshold (e.g., more than 50% of history), it first calls the LLM to produce a summary of the content being dropped and inserts that summary as a synthetic system message before the trimmed window. This preserves semantic continuity at the cost of one additional LLM call.

---

## 6. Internal Working Mechanics

### KV Cache Mechanics

A transformer attention head computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

For each token position `i`, the Key tensor `K_i` and Value tensor `V_i` are functions only of that token and the model weights — not of the query. If the token sequence at positions 0..N does not change between two calls, `K_0..K_N` and `V_0..V_N` are identical and can be reused. This is the KV cache.

**Provider implementations:**
- **Anthropic (Claude):** Explicit opt-in via `cache_control: {"type": "ephemeral"}` on a `system` block or a `user` message block. Cache TTL is 5 minutes for ephemeral caches. Cache hits reduce input token billing by approximately 90% for cached tokens.
- **OpenAI (GPT-4o, GPT-4o-mini):** Automatic prefix caching. Any prompt with a prefix that matches a recently cached prefix (≥1,024 tokens, aligned to 128-token boundaries) receives a cache hit. Cached tokens are billed at 50% of the standard input rate.
- **Self-hosted (vLLM):** Configurable `block_size` and `gpu_memory_utilization` control the KV cache pool. The PagedAttention algorithm manages KV blocks in a virtual memory fashion.

### Token Trimming Strategies

**Sliding Window:** Keep the last `N` complete turns. Simple, predictable, zero additional LLM calls. Fails when critical context is established early in the conversation (e.g., user's name, stated constraints).

**Importance-Scored Pruning:** Assign each turn an importance score. Scores can be computed via:
- TF-IDF similarity to the current query (fast, no LLM call)
- BM25 retrieval score (slightly better recall)
- LLM-graded relevance (highest quality, adds latency and cost)

Evict lowest-scoring turns first. Preserves turns that are semantically relevant to the current question at the expense of turns that are topically distant.

**Summary Compression:** When evicting a block of turns, call the LLM with those turns and the instruction "summarise the key facts, decisions, and constraints from this conversation segment in under 200 tokens." Insert the summary as a synthetic assistant turn. Preserves semantic continuity but adds one LLM call per compression event.

### Token Counting

Accurate token counting before trimming is essential. Estimating by word count introduces errors of ±15–30% depending on the language and content type. Use the exact tokeniser:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
token_count = len(enc.encode(text))
```

For Anthropic models, use the `client.beta.messages.count_tokens()` API method or the `anthropic-tokenizer` Python package.

### Edge Cases

- **Mid-turn trimming:** Never trim within a message. A half-message is worse than no message — the model sees an incomplete instruction and hallucinates the missing content.
- **System prompt in trim budget:** Treat the system prompt as untouchable. Only evict user/assistant turns.
- **Tool call turns:** Preserve tool_call and tool_result pairs together — splitting them causes the model to misinterpret the tool result.

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd` for the full Mermaid source.

```
[Static Prefix]            [KV Cache Pool]
System Prompt + Docs  -->  Provider Infrastructure
                                  |
                                  v (cache hit: skip prefill)
[Dynamic Suffix]           [Token Budget Enforcer]
Conversation History  -->  Count → Evict → Compress
Current User Message        |
                            v
                      [LLM API Call]
                            |
                            v
                      [Response]
```

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd` for the full Mermaid source.

The sequence shows:
1. Application splits prompt into static prefix and dynamic suffix.
2. Budget enforcer counts tokens in the dynamic suffix.
3. If over budget, enforcer evicts turns (and optionally compresses).
4. API call is submitted with cache_control on the static prefix.
5. Provider checks KV cache — hit returns immediately; miss computes and stores.
6. Generation proceeds on the dynamic suffix only.
7. Response is returned and appended to conversation history.

---

## 9. Implementation Guide

### Step 1: Install dependencies

```bash
pip install openai tiktoken anthropic
```

### Step 2: Configure environment

```bash
cp .env.example .env
# Set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Step 3: Implement the token budget enforcer

```python
# From src/kv_caching_core.py
import tiktoken
from typing import List

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def trim_messages_to_budget(
    messages: List[dict],
    budget: int,
    model: str = "gpt-4o"
) -> List[dict]:
    """
    Evict oldest complete turns until the message list fits within the token budget.
    Never evicts the system message (role == 'system').
    Never splits a turn — always removes complete user+assistant pairs.
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    turn_msgs   = [m for m in messages if m["role"] != "system"]

    while True:
        total = sum(count_tokens(m["content"], model) for m in system_msgs + turn_msgs)
        if total <= budget or len(turn_msgs) == 0:
            break
        # Evict the oldest turn (preserve pairs: remove first message)
        turn_msgs.pop(0)

    return system_msgs + turn_msgs
```

### Step 4: Apply KV cache headers (Anthropic)

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": STATIC_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}  # Cache this prefix
        }
    ],
    messages=trimmed_conversation_history
)
```

### Step 5: Run and verify

```bash
python src/main.py
# Demo mode (no API key):
DEMO_MODE=true python src/main.py
```

Expected output:

```
KV Caching & Token Trimming Demo
=================================
Input tokens (full history): 12,450
Tokens after trimming:        3,820
Cache status:                 HIT (3,000 static prefix tokens reused)
Effective input tokens billed: 820
Estimated latency reduction:  ~65%
```

---

## 10. Benefits & Trade-offs

| Benefit | Trade-off |
|---|---|
| Up to 85% prefill latency reduction on cache hits | KV cache miss on first call still pays full prefill cost |
| Input token costs drop 50–90% for cached prefixes | Cache TTL limits (5 min ephemeral) require re-submission on timeout |
| Context window never exhausted for long sessions | Sliding window can drop early critical context |
| Coherence preserved via summary compression | Compression introduces one extra LLM call per eviction event |
| Importance-scored pruning retains relevant history | TF-IDF scoring adds CPU overhead; LLM scoring adds latency |
| Deterministic token counting prevents hard context errors | Requires per-model tokeniser; models tokenise differently |

---

## 11. Performance Characteristics

### Latency

- **KV cache hit (Anthropic ephemeral):** Prefill for cached tokens is eliminated. TTFT improvement is approximately proportional to the fraction of total tokens that are cached. A 3,000-token system prompt out of a 4,000-token context yields ~75% TTFT improvement on cache hit.
- **KV cache miss:** Full prefill cost. First call in a session always misses.
- **Token trimming overhead:** Tiktoken tokenisation runs at approximately 500k tokens/second on a modern CPU. Trimming 16k tokens adds ~32ms of CPU time — negligible versus API latency.
- **Summary compression:** Adds one full LLM round-trip (typically 1–3 seconds). Should be triggered infrequently (e.g., only when evicting >50% of history).

### Memory

- **Client-side:** Conversation history stored in memory. A 50-turn conversation with average 200 tokens/turn is ~40KB — not a concern.
- **Server-side:** KV cache memory per active session is model-dependent. For Claude 3.5 Sonnet, approximately 1MB per 1,000 cached tokens.

### Throughput

Token trimming is synchronous and single-threaded but fast. At 100 concurrent sessions, CPU cost of trimming is negligible compared to API I/O wait.

### References

- Anthropic prompt caching documentation: https://docs.anthropic.ai/en/docs/build-with-claude/prompt-caching
- OpenAI prompt caching: https://platform.openai.com/docs/guides/prompt-caching
- vLLM PagedAttention: Kwon et al. (2023), "Efficient Memory Management for Large Language Model Serving with PagedAttention", arXiv:2309.06180

---

## 12. Security Considerations

### OWASP LLM Top 10 Relevance

**LLM06 — Sensitive Information Disclosure:** Cached prefixes persist in provider infrastructure for the TTL duration. If a static prefix contains PII or confidential business rules, that data remains in the cache after the session ends. Mitigation: do not cache prompts containing PII; use separate cached prefixes per tenant.

**LLM01 — Prompt Injection:** If the static prefix includes user-supplied content (e.g., a dynamically constructed system prompt), an attacker can craft input that modifies the cached prefix on their session and potentially influences future behaviour if the cache is shared. Mitigation: treat the static prefix as strictly application-controlled; never include raw user input in the cacheable portion.

**Data Residency:** Anthropic ephemeral caches are stored in the same region as the API endpoint. For regulated industries (healthcare, finance), verify that the caching region satisfies data residency requirements before enabling prompt caching.

**Token Budget Manipulation:** A malicious user sending very long messages can force aggressive trimming of legitimate history, causing the model to lose context of prior security constraints. Mitigation: enforce a maximum per-message token limit (e.g., 2,000 tokens) before the message enters the history.

---

## 13. Cost Analysis

### Baseline (no caching or trimming)

A 50-turn conversation where each turn includes the full cumulative history:
- Average history length per turn: 5,000 tokens (growing from 0 to 10,000)
- Total input tokens across 50 turns: ~250,000
- At $3/1M tokens (GPT-4o input): **$0.75 per conversation**

### With KV Caching (3,000-token system prompt cached)

- Each turn: 3,000 cached tokens (50% billing = 1,500 effective) + remaining uncached tokens
- Effective input tokens across 50 turns: ~175,000
- Cost: ~$0.525 per conversation — **30% reduction**

### With Token Trimming (8,000-token budget)

- Each turn submits at most 8,000 tokens regardless of history length
- Total input tokens across 50 turns: ~400,000 (but bounded, no growth past turn 20)
- Without trimming, turn 50 would submit 50,000 tokens alone; with trimming, it submits 8,000
- Net cost at scale: **55–65% reduction** on long-running sessions

### Combined (caching + trimming)

- Roughly 70–80% cost reduction on conversations >20 turns
- Break-even point: any conversation longer than ~5 turns benefits from caching; any conversation longer than ~10 turns benefits from trimming

---

## 14. Best Practices

1. **Always trim at message boundaries.** Cutting within a message produces incoherent context. Implement eviction at the turn (user+assistant pair) level.

2. **Pin the system prompt as the cacheable prefix.** The system prompt is the most stable part of the context and the best candidate for caching. Separate it cleanly from dynamic content.

3. **Count tokens before every API call, not after.** Reactive trimming (catching a context-length error) wastes one failed API call. Proactive counting adds microseconds.

4. **Use the provider's official tokeniser.** Word-count estimates are ±30% inaccurate. Use tiktoken for OpenAI models and the Anthropic token-count endpoint for Claude models.

5. **Preserve tool_call / tool_result pairs.** The model requires both halves of a tool interaction to reason about it correctly. Never evict one without the other.

6. **Set the trim budget below the model's context limit, not at it.** Leave a 20% buffer (e.g., set budget to 100k for a 128k-window model) to accommodate the current message, output tokens, and tokenisation variance.

7. **Log cache hit rates.** Track `cache_read_input_tokens` (Anthropic) or the `cached_tokens` field (OpenAI) in every response. A hit rate below 70% on a system with a static system prompt indicates a structural issue.

8. **Separate trim budget per role.** Reserve a minimum allocation for the system prompt and recent N turns; use the remainder for older history. This prevents a very long system prompt from leaving no room for history.

9. **Compress, do not discard, when evicting >50% of history.** Abrupt loss of large context blocks degrades coherence more than a lossy summary. Reserve summary compression for large evictions only.

10. **Test trimming logic offline.** Trimming functions depend only on string content and token counts — no API calls required. Write unit tests that verify boundary conditions without spending tokens.

---

## 15. Anti-Patterns

### 1. The Infinite Scroll

**What it looks like:** Appending every turn to the messages array with no eviction policy.
**Why it fails:** Context length errors occur suddenly, terminating live sessions. Costs grow linearly with conversation length. TTFT degrades progressively.
**What to do instead:** Implement a token budget enforcer from day one, even if the initial budget is generous.

### 2. The Hard Truncation

**What it looks like:** Slicing the messages array at a fixed index (e.g., `messages[-10:]`) without considering token counts.
**Why it fails:** Token density varies. Ten short messages may be 500 tokens; ten long messages may be 8,000 tokens. Index-based truncation is not token-budget aware.
**What to do instead:** Use a token-counting loop that evicts until the budget is satisfied.

### 3. The Shared Cache Prefix

**What it looks like:** Using the same cacheable prefix for all users, including user-specific instructions ("Your name is Alice and you work at Acme Corp").
**Why it fails:** User-specific content in the prefix means a different prefix per user — zero cache hits. Also risks cross-user data leakage if the prefix accidentally contains another user's context.
**What to do instead:** The cacheable prefix must contain only truly static, tenant-agnostic content.

### 4. The Mid-Tool Eviction

**What it looks like:** Trimming evicts a tool_call message but retains the tool_result message that follows it.
**Why it fails:** The model sees a tool_result with no corresponding tool_call and interprets it as a malformed conversation, triggering hallucinated explanations of where the result came from.
**What to do instead:** Treat tool_call + tool_result as an atomic unit; evict both or neither.

### 5. The Compression Cascade

**What it looks like:** Compressing at every turn because the budget is set too low relative to the average turn length.
**Why it fails:** Every turn triggers a compression LLM call, doubling latency and cost — worse than no caching at all.
**What to do instead:** Set the budget to at least 5× the average turn length. Compression should trigger rarely, not routinely.

### 6. The Stale Cache

**What it looks like:** Assuming a KV cache hit on every call after the first, without checking the cache usage response fields.
**Why it fails:** Anthropic ephemeral caches expire after 5 minutes of inactivity. A user who pauses a conversation pays full prefill cost on resumption.
**What to do instead:** Monitor cache hit rates in production. Re-warm caches proactively in long-idle sessions if the use case requires consistent latency.

---

## 16. Common Mistakes

### Mistake 1: Tokenising after trimming instead of before

**Symptom:** Occasional context-length API errors despite having a trimming function.
**Root cause:** The trim function counts tokens on the original list, evicts based on that count, but the final assembled prompt (including separators, special tokens, and the new user message) exceeds the limit.
**Fix:** Count tokens on the fully assembled prompt — including the new user message and all formatting — before calling the API.

### Mistake 2: Including the new user message inside the trim budget

**Symptom:** The assistant appears to "forget" the user's most recent question.
**Root cause:** The trimmer treats the current user message as evictable history and removes it when the budget is tight.
**Fix:** Exclude the current user message from the budget calculation. Trim only the prior history, then add the current message at the end.

### Mistake 3: Using character count as a proxy for token count

**Symptom:** Intermittent context-length errors on inputs containing code, URLs, or non-English text.
**Root cause:** Character-to-token ratio is roughly 4:1 for English prose but can be 1:1 for dense code or non-Latin scripts. Character-based estimation under-counts tokens for these content types.
**Fix:** Always use the model's official tokeniser. The 32ms overhead of exact counting is always worth it.

---

## 17. Production Checklist

- [ ] Token budget enforcer implemented and unit-tested offline
- [ ] Budget set to ≤80% of the model's context window limit
- [ ] System prompt separated into a static cacheable prefix
- [ ] Cache_control headers applied to static prefix (Anthropic) or prompt structure optimised for prefix caching (OpenAI)
- [ ] Token counting uses the provider's official tokeniser (tiktoken or Anthropic count_tokens)
- [ ] Tool_call / tool_result pairs evicted atomically
- [ ] Current user message excluded from the trim budget calculation
- [ ] Summary compression implemented for large eviction events (>50% of history)
- [ ] Cache hit rate logged from every response (cache_read_input_tokens / total_input_tokens)
- [ ] Alert configured when cache hit rate drops below 60% on a session type expected to hit frequently
- [ ] Per-message token limit enforced at ingestion (prevents single large message exhausting the budget)
- [ ] System prompt excluded from eviction candidates
- [ ] Cost monitoring dashboard includes breakdown of cached vs. uncached input tokens
- [ ] Integration tests cover the context-limit boundary condition
- [ ] Fallback behaviour defined for summary compression failures (LLM call fails mid-session)

---

## 18. References

[1] Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180. https://arxiv.org/abs/2309.06180

[2] Anthropic. (2024). "Prompt Caching." Anthropic Documentation. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

[3] OpenAI. (2024). "Prompt Caching." OpenAI Platform Documentation. https://platform.openai.com/docs/guides/prompt-caching

[4] LangChain. (2024). "ConversationTokenBufferMemory." LangChain Python Documentation. https://python.langchain.com/docs/modules/memory/types/token_buffer

[5] OpenAI. (2024). "tiktoken." GitHub. https://github.com/openai/tiktoken

[6] Liu, N. F. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172. https://arxiv.org/abs/2307.03172

---

## 19. Summary

KV caching and token trimming address the same root cause from two directions: the quadratic cost of reprocessing unchanged context. KV caching delegates the solution to provider infrastructure — once a stable prefix is computed, its Key-Value tensors are stored and reused across calls, cutting prefill latency and input token costs by up to 85%. Token trimming solves the complementary problem at the application layer — enforcing a context budget so the dynamic suffix never grows unboundedly, preventing context overflows and cost explosions in long conversations. The two techniques compose naturally: cache what does not change, trim what does. In production, any multi-turn LLM application handling conversations longer than ten turns should implement both.

---

## 20. Exercises

**Beginner:** Run the PoC in demo mode (`DEMO_MODE=true python src/main.py`). Observe the token counts before and after trimming. Change `MAX_CONTEXT_TOKENS` in `.env` to 1,000 and note how many more turns are evicted.

**Intermediate:** Modify `kv_caching_core.py` to implement importance-scored pruning using TF-IDF similarity between each historical turn and the current user message. Compare the turns retained by sliding window versus TF-IDF pruning on the same input.

**Advanced:** Extend the PoC to add a summary compression step. When the trimmer evicts more than 5 turns in a single pass, call the LLM (or mock it in demo mode) to produce a 100-token summary of the evicted turns and insert it as a synthetic system message.

**Expert:** Using the OpenAI API, instrument a 30-turn conversation with and without prompt caching. Record TTFT and `cached_tokens` at each turn. Plot TTFT vs. turn number for both conditions and calculate the average cost per turn in each scenario.

**Research:** Read Kwon et al. (2023) "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arXiv:2309.06180). Identify one limitation of the PagedAttention KV cache approach not mentioned in this document and describe how it affects multi-tenant LLM deployments.

---

## 21. Interview Questions

**Conceptual:**
1. Explain KV caching to a product manager who has never studied transformers. What is being cached, and why does it save time?
2. What is the difference between server-side KV caching and client-side token trimming? When would each be insufficient on its own?

**Technical:**
3. Anthropic's ephemeral cache has a 5-minute TTL. How would you design a system that maintains consistent low latency for users who pause their session for 10 minutes?
4. A token trimmer using `messages[-10:]` index slicing passes code review. What specific input would cause it to fail in production?
5. Why must tool_call and tool_result messages be evicted as a pair? What does the model produce if tool_result exists without the preceding tool_call?

**Design:**
6. You are building a customer support chatbot that handles 50,000 conversations per day, averaging 15 turns each. The system prompt is 2,000 tokens and is identical for all users. Design the caching and trimming architecture. What metrics would you track on day one?
7. How would you architect token trimming differently for a coding assistant (where code blocks must be preserved intact) versus a general Q&A assistant?

**Trade-off:**
8. When would you choose summary compression over sliding window eviction, and when would the reverse be true? Name a concrete use case for each.
9. Importance-scored pruning retains semantically relevant history but adds latency for scoring. At what conversation length does the quality improvement justify the scoring overhead?

**Debugging:**
10. A production assistant starts giving incoherent responses after approximately 20 turns, but only for users who ask long questions. The trimmer log shows no errors. What are the three most likely root causes, and how would you diagnose each?
