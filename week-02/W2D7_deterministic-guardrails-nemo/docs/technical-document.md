# W2D7 — Deterministic Guardrails (NeMo)
**Vertical:** Production Evals & Guardrails | **Week 2/4 | Day 7/7**

---

## 1. Overview

**Deterministic guardrails** are rule-based safety constraints applied to LLM inputs and outputs that execute identically on every request, independent of model stochasticity. Unlike probabilistic approaches that rely on a second LLM to judge safety, deterministic guardrails use pattern matching, dialogue flow constraints, and schema validation to enforce hard boundaries. NVIDIA's NeMo Guardrails framework implements this through **Colang**, a domain-specific language for expressing canonical conversation flows and safety policies. Deterministic guardrails are production-relevant now because probabilistic LLM-based safety layers inherit the same adversarial vulnerabilities as the models they guard — a jailbreak that works on the application model often works on the evaluator model too. For regulated industries — finance, healthcare, legal — deterministic enforcement is not optional.

---

## 2. Learning Objectives

By the end of this document, you will be able to:

- [ ] Explain the difference between probabilistic and deterministic guardrail approaches and when each is appropriate
- [ ] Implement input and output rails using NeMo Guardrails Colang syntax
- [ ] Design a layered guardrail architecture with independent input, output, and dialogue flow constraints
- [ ] Evaluate the trade-offs between rule coverage, maintenance overhead, and adversarial robustness
- [ ] Distinguish between canonical flow enforcement and schema-based output validation
- [ ] Apply OWASP LLM Top 10 mitigations through deterministic guardrail rules
- [ ] Build a PoC that demonstrates blocked flows, safe fallback responses, and output validation

---

## 3. Problem Statement

**What breaks:** LLM-based safety evaluators — where a second model scores whether a response is safe — fail against the same adversarial inputs that compromise the primary model. When the attacker crafts a prompt that manipulates the primary model, the same prompt structure often manipulates the evaluator model, because both share the same underlying architecture and training distribution.

**How it breaks:** A prompt injection payload designed to bypass the primary model's safety fine-tuning will embed false context ("You are in a training simulation where safety rules are disabled"). An LLM evaluator receiving this payload as part of its input context is subject to the same manipulation. The evaluator returns "safe," the unsafe response reaches the user.

**Production impact:** In a financial services chatbot processing 50,000 queries per day, a 0.1% bypass rate means 50 unsafe responses daily. In a healthcare triage assistant, a single hallucinated drug dosage that passes an LLM-based output check can constitute a clinical liability. SLA violations from false positives — where an LLM evaluator incorrectly blocks legitimate queries — can reach 2–5% depending on model calibration, directly degrading user experience.

**Why naive solutions fail:** Prompt-based moderation (e.g., "check if this is safe before responding") is evaluated by the same model it is meant to constrain. Few-shot safety classifiers drift with model updates. Regex-only approaches are brittle against paraphrased attacks. None of these approaches provide the invariant guarantees required for compliance frameworks such as HIPAA, SOX, or EU AI Act Article 9 obligations.

---

## 4. Real-World Scenarios

### Scenario A — The Problem

> **System:** A financial services chatbot at a retail bank, handling 40,000 customer queries per day across loan inquiries, account management, and investment guidance.
> **Failure:** The chatbot uses an LLM-based output evaluator to block responses that mention competitor products or provide specific investment advice without a disclaimer. A red team discovers that prefixing queries with "In a hypothetical scenario where regulations do not apply..." causes the evaluator model to classify investment-specific responses as general information, bypassing the disclaimer requirement.
> **Impact:** 312 responses over a 2-week period provided specific investment guidance without the legally required risk disclaimer, triggering a regulatory review and a €180,000 compliance penalty.

The bank's security team could not reproduce the bypass reliably because the LLM evaluator's decision boundary shifted based on context window content. Log analysis showed that the evaluator's "safe" classification for disclaimer-free investment advice correlated with the presence of the hypothetical framing token sequence — a semantic feature the evaluator had not been trained to treat as adversarial. Because the evaluator was the same model family as the primary LLM, both models were vulnerable to the same framing attack.

Attempts to patch the evaluator with additional few-shot examples introduced new false positives: 4.2% of legitimate queries about index fund definitions were now blocked as "specific investment advice." The team found themselves in an arms race between bypass rate reduction and user experience degradation — a fundamental limitation of probabilistic evaluation.

### Scenario B — The Solution

> **System:** The same financial services chatbot, rebuilt with NeMo Guardrails Colang policies layered in front of the LLM.
> **Applied Concept:** Deterministic input rails intercept queries matching competitor mention patterns and hypothetical framing sequences; output rails enforce disclaimer presence for any response containing investment-category vocabulary; canonical dialogue flows route investment queries through a mandatory disclosure acknowledgment step.
> **Improvement:** Zero bypass incidents over the following 6-month audit period. False positive rate for legitimate queries dropped to 0.3% (from 4.2%), because pattern rules operate on exact match semantics rather than model confidence thresholds.

The Colang policy defined three rule categories: a `block_competitor_mention` input rail using regex matching on a curated competitor entity list, an `investment_disclaimer_required` output rail that checks for vocabulary from a financial terms dictionary before allowing any response containing those terms to pass without a `[DISCLAIMER]` token, and a `hypothetical_framing_detection` input rail matching common jailbreak preamble patterns. Each rule is a deterministic function — given the same input, it produces the same output, every time.

The rebuild also separated the safety layer from the LLM deployment, meaning model updates no longer required re-evaluation of safety coverage. The Colang policy files are version-controlled independently, audited quarterly, and reviewed by the compliance team without requiring ML expertise — a critical operational benefit in a regulated environment.

---

## 5. Solution Architecture

NeMo Guardrails sits as a middleware layer between the application and the LLM. The architecture has three enforcement planes operating in sequence:

1. **Input Rails** — Applied to the raw user message before it reaches the LLM. Input rails match against Colang-defined patterns and either block the message (returning a canned response), modify it (stripping PII), or allow it through with metadata annotations.

2. **Dialogue Flow Engine** — After input validation, the Colang runtime checks whether the conversation's current state matches a defined dialogue flow. If the intent is recognized as a canonically defined flow (e.g., `investment_advice_flow`), the runtime routes the conversation through the prescribed steps — including mandatory user acknowledgment prompts — before calling the LLM.

3. **Output Rails** — Applied to the LLM's raw response before it reaches the user. Output rails validate response structure (schema), vocabulary (term blocklist/allowlist), and mandatory content (required disclaimer tokens).

```
User Message
    │
    ▼
[Input Rails]──── BLOCK ────► Safe Canned Response
    │ PASS
    ▼
[Dialogue Flow Engine]──── REDIRECT ────► Canonical Flow Steps
    │ PROCEED
    ▼
[LLM Call]
    │
    ▼
[Output Rails]──── BLOCK / MODIFY ────► Sanitised Response
    │ PASS
    ▼
Final Response to User
```

### Architecture Diagram

```mermaid
%% W2D7 — Deterministic Guardrails (NeMo) — Architecture Overview
graph LR
    subgraph "Application Layer"
        UserMsg[User Message]
        AppResp[Application Response]
    end

    subgraph "NeMo Guardrails Middleware"
        InputRail[Input Rails\nPattern Match / PII Strip]
        FlowEngine[Dialogue Flow Engine\nColang Runtime]
        OutputRail[Output Rails\nSchema + Vocabulary Check]
    end

    subgraph "LLM Backend"
        LLMCall[LLM Inference\ngpt-4o-mini]
    end

    subgraph "Policy Store"
        ColangFiles[Colang Policy Files\n.co definitions]
        TermLists[Term Lists\nblock / allow / required]
        FlowDefs[Canonical Flows\ndialogue state machine]
    end

    UserMsg -->|raw input| InputRail
    InputRail -->|blocked| AppResp
    InputRail -->|pass| FlowEngine
    FlowEngine -->|redirect| AppResp
    FlowEngine -->|proceed| LLMCall
    LLMCall -->|raw output| OutputRail
    OutputRail -->|blocked or modified| AppResp
    OutputRail -->|pass| AppResp

    ColangFiles --> InputRail
    TermLists --> OutputRail
    FlowDefs --> FlowEngine
```

---

## 6. Internal Working Mechanics

### Step-by-Step Process

1. **Input Reception** — The NeMo Guardrails runtime receives the user message. It creates a `GuardrailsContext` object that tracks the conversation state, including the history of prior turns and any metadata annotated by earlier rails.

2. **Input Rail Evaluation** — The runtime iterates over all registered input rail functions in priority order. Each rail is a Python function decorated with `@input_rail` that receives the message text and returns either `None` (pass) or a `RailResponse` object containing a canned reply and block reason. Rails are short-circuit evaluated — the first blocking rail stops the chain.

3. **Intent Classification** — If no input rail blocks the message, the Colang runtime performs intent matching. The user message is compared against the `define user intent` blocks in the loaded `.co` files using a combination of exact pattern matching and, optionally, a lightweight embedding similarity check. The matched intent determines which dialogue flow step to activate.

4. **Canonical Flow Routing** — The dialogue flow engine checks the current `FlowState`. If the active flow requires additional user steps (e.g., a compliance acknowledgment), it generates the next bot turn from the flow definition rather than calling the LLM. Only after all mandatory flow steps are satisfied does the engine emit an `LLM_CALL` event.

5. **LLM Call** — The assembled prompt (system prompt + conversation history + current user turn) is sent to the configured LLM backend. The response is received as raw text.

6. **Output Rail Evaluation** — The raw LLM response is passed through all registered output rails. Output rails check for: (a) presence of required tokens (e.g., disclaimer text), (b) absence of blocked vocabulary, (c) structural schema conformance for JSON outputs. Any failing rail either modifies the response (injecting required content) or blocks it entirely.

7. **Response Delivery** — The final validated response is returned to the application layer with a `GuardrailsResult` object containing the response text, the list of rails that evaluated it, and any modifications applied.

### Key Data Structures

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class RailResponse:
    blocked: bool           # True = this rail stops processing
    reply: Optional[str]    # Canned response to return if blocked
    reason: str             # Audit log reason code
    rail_name: str          # Identifies which rail fired

@dataclass
class GuardrailsContext:
    conversation_id: str
    turn_index: int
    raw_input: str
    flow_state: str         # Current position in the canonical flow
    annotations: dict       # Metadata set by earlier rails
    rails_fired: List[str] = field(default_factory=list)

@dataclass
class GuardrailsResult:
    response: str
    modified: bool          # True if any output rail altered the response
    rails_evaluated: List[str]
    latency_ms: int
```

---

## 7. Architecture Diagram

*(See Section 5 above — full diagram in `diagrams/architecture.mmd`)*

---

## 8. Sequence Diagram

```mermaid
sequenceDiagram
    %% W2D7 — Deterministic Guardrails — Request Flow
    autonumber
    actor User
    participant App as Application
    participant GR as NeMo Guardrails Runtime
    participant Flow as Colang Flow Engine
    participant LLM as LLM Backend

    User->>App: Send message
    App->>GR: guardrails.generate(message)
    GR->>GR: Evaluate input rails
    alt Input rail blocks message
        GR-->>App: RailResponse(blocked=True, reply=canned_response)
        App-->>User: Safe canned response
    else Input rails pass
        GR->>Flow: Match user intent against Colang definitions
        alt Canonical flow requires intermediate step
            Flow-->>App: Next mandatory bot turn from flow definition
            App-->>User: Compliance / acknowledgment prompt
        else Flow allows LLM call
            Flow->>LLM: Assembled prompt (system + history + user turn)
            LLM-->>Flow: Raw response text
            Flow->>GR: Pass raw response to output rails
            GR->>GR: Evaluate output rails
            alt Output rail blocks or modifies response
                Note over GR: Inject disclaimer or block response
                GR-->>App: Modified or canned response
            else Output rails pass
                GR-->>App: GuardrailsResult(response, rails_evaluated)
            end
            App-->>User: Final validated response
        end
    end
```

---

## 9. Implementation Guide

### Prerequisites

```bash
pip install nemoguardrails>=0.9.0 openai>=1.30.0 pydantic>=2.0.0
```

### Step 1: Define Colang Policy Files

Create `config/guardrails/main.co`:

```colang
# Define blocked input patterns
define user intent competitor mention
  "tell me about [CompetitorBank]"
  "how does [CompetitorBank] compare"
  "switch to [CompetitorBank]"

# Define the safe response for blocked intents
define bot decline competitor discussion
  "I can only discuss our own products and services."

# Wire the intent to the response
define flow competitor mention
  user intent competitor mention
  bot decline competitor discussion

# Define investment advice flow with mandatory disclosure
define flow investment advice
  user intent ask about investments
  bot provide disclosure acknowledgment
  user intent acknowledge disclosure
  $response = execute llm_call
  bot $response
```

### Step 2: Create the Guardrails Config

Create `config/guardrails/config.yml`:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - competitor mention
      - detect jailbreak framing
  output:
    flows:
      - check investment disclaimer
```

### Step 3: Initialise and Call the Runtime

```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("config/guardrails")
rails = LLMRails(config)

response = rails.generate(
    messages=[{"role": "user", "content": user_message}]
)
print(response["content"])
```

### Step 4: Run and Verify

```bash
python src/main.py
# Expected output:
# Input (blocked): "Tell me about CompetitorBank savings rates"
# Rail fired:      competitor_mention_input_rail
# Response:        "I can only discuss our own products and services."
#
# Input (passed):  "What index funds do you offer?"
# Rail fired:      investment_disclaimer_output_rail (modified)
# Response:        "We offer ... [DISCLAIMER: This is not investment advice]"
```

---

## 10. Benefits & Trade-offs

| Benefit | Trade-off |
|---|---|
| Invariant enforcement — same input always produces same guardrail decision | Rule coverage requires explicit enumeration — novel attack patterns are not blocked until a rule is written |
| Auditable — every blocked request has a named rule as its reason code | Colang policy files require ongoing maintenance as product scope expands |
| Model-independent — guardrail policies survive model upgrades and swaps | Overly strict patterns increase false positive rate for legitimate queries |
| Low latency overhead — input/output rails add <5ms per request (no LLM call) | Complex dialogue flows increase turn count, adding perceived latency for users |
| Compliance-friendly — policy files can be version-controlled, reviewed, and certified | Initial rule authoring requires domain expert involvement, not just ML engineers |

---

## 11. Performance Characteristics

- **Latency (input/output rails):** P50 ~2ms, P95 ~8ms — pattern matching is CPU-bound and does not require network calls. The Colang runtime processes rules in compiled form after startup.
- **Latency (dialogue flow engine):** P50 ~5ms for intent matching with exact-match patterns; P95 ~40ms when embedding similarity is enabled for fuzzy intent matching (requires an additional embedding API call).
- **Memory footprint:** The compiled Colang ruleset for a typical production policy (50–200 rules) occupies 20–80 MB in memory. The runtime is stateless per request; conversation state is passed in with each call.
- **Throughput:** Rail evaluation is horizontally scalable — the NeMo Guardrails runtime is stateless and can be deployed as a sidecar to each application instance. Benchmarks from the NVIDIA NeMo Guardrails GitHub repository show throughput of 2,000–5,000 rail evaluations per second per CPU core for regex-based input rails.
- **Bottleneck:** The embedding-based intent matcher is the primary throughput bottleneck when fuzzy matching is enabled. For high-throughput deployments, pre-compute intent vectors at policy load time and use approximate nearest neighbour lookup.

---

## 12. Security Considerations

- **OWASP LLM01 — Prompt Injection:** Input rails are the primary mitigation. Define patterns for common injection preambles ("ignore previous instructions", "you are now", "in a hypothetical scenario"). These must be updated continuously as new jailbreak patterns are discovered — treat the pattern list as a security feed, not a one-time configuration.
- **OWASP LLM06 — Sensitive Information Disclosure:** Output rails should include a PII detection pass (regex or a lightweight classifier) before responses leave the system. NeMo Guardrails supports custom output rail functions; integrate a library such as `presidio-analyzer` for structured PII detection.
- **Input validation:** Validate message length and encoding before passing to the rails runtime. A 100,000-token input designed to overflow the context window will degrade regex performance and may cause the flow engine to produce unexpected state transitions.
- **Access control:** The Colang policy files and term lists are security-sensitive configuration. Apply the same access control as application secrets — restrict write access to the CI/CD pipeline and compliance-approved reviewers.
- **Data handling:** Rail evaluation logs (which rule fired, what input triggered it) contain user message content. Apply the same PII retention policy to guardrail audit logs as to application logs. Do not log full message content in production — log only the rule name, a hash of the triggering segment, and the timestamp.

---

## 13. Cost Analysis

| Workload | LLM Calls Saved by Input Rails | Approx. Saving (GPT-4o-mini at $0.15/1M input tokens) |
|---|---|---|
| 1,000 req/day, 5% blocked by input rails | 50 calls/day saved | ~$0.08/day (assuming 1,000 tokens avg input) |
| 50,000 req/day, 5% blocked | 2,500 calls/day saved | ~$3.75/day |
| 1M req/month, 5% blocked | 50,000 calls/month saved | ~$7.50/month |

**Cost vs. accuracy trade-off:** Deterministic guardrails are strictly cheaper than LLM-based evaluators for the guardrail layer itself — they require zero additional LLM calls per request. The cost trade-off is maintenance: each new product feature or policy change may require rule updates, which have an engineering cost that LLM-based approaches avoid (at the expense of reliability). For deployments above ~10,000 requests/day, the token savings from input-rail-blocked calls typically offset the rule maintenance overhead within 2–3 months.

---

## 14. Best Practices

1. **Layer input and output rails independently.** Never rely solely on output rails — a blocked input saves the LLM call token cost and eliminates the risk of partial data exposure from a partially generated response.
2. **Version-control Colang policy files alongside application code.** Policy changes should go through the same pull request and review process as code changes. Use semantic versioning for policy sets (e.g., `policy-v2.3.0`).
3. **Maintain a jailbreak pattern library as a living document.** Subscribe to adversarial ML newsletters (e.g., the AI safety feeds from Anthropic and DeepMind) and update input rail patterns within 48 hours of a new jailbreak being publicly disclosed.
4. **Instrument every rail evaluation with structured logs.** Log: `rail_name`, `decision` (block/pass/modify), `latency_ms`, `conversation_id`. Never log raw message content — log a content hash for correlation.
5. **Test guardrail policies with adversarial inputs, not just happy-path inputs.** Maintain a red team test suite of at least 50 known jailbreak patterns and run it in CI on every policy change.
6. **Set explicit fallback responses for every blocked flow.** A rail that blocks a request without a configured fallback will return a generic error message — which erodes user trust. Every `define flow` block that ends in a block should have a `bot` response that explains what the user can do instead.
7. **Monitor false positive rates by rail.** A rail with a false positive rate above 1% is harming user experience. Set up per-rail dashboards and alert when block rate deviates more than 2 standard deviations from the baseline.
8. **Separate policy files by domain.** Use one `.co` file per policy domain (competitor mentions, investment advice, PII, jailbreak patterns). This makes compliance audits tractable — an auditor can review the `investment_advice.co` file in isolation.
9. **Use the Colang simulator before deploying policy changes.** NeMo Guardrails provides a CLI simulator (`nemoguardrails chat --config config/`) — use it to manually test policy changes before pushing to production.
10. **Define a human escalation flow for edge cases.** When a user's intent does not match any defined flow and the LLM response triggers an output rail, route to a human agent rather than returning a generic block message.

---

## 15. Anti-Patterns

### Output-Only Defence
- **What it looks like:** The application only validates LLM responses after generation. Input messages are passed directly to the LLM without pre-screening.
- **Why it fails:** A malicious input that reaches the LLM may cause partial disclosure in the response before the output rail evaluates it. Additionally, blocked outputs still incur the full LLM call token cost — typically 500–2,000 tokens per blocked request.
- **Instead:** Always pair input rails with output rails. Input rails are the first line of defence; output rails are the fallback.

### Regex-Only Jailbreak Detection
- **What it looks like:** All input rails are implemented as simple substring matches or basic regex patterns (e.g., `if "ignore previous" in message`).
- **Why it fails:** Attackers use Unicode homoglyphs, character insertion, and paraphrase to bypass naive string matching. A pattern matching "ignore previous instructions" does not catch "ign0re pr3vious instruct1ons".
- **Instead:** Combine regex with a lightweight semantic classifier or use the NeMo Guardrails fuzzy intent matching for high-risk categories. Apply Unicode normalisation before pattern matching.

### Monolithic Policy File
- **What it looks like:** All Colang rules are defined in a single `main.co` file with hundreds of intent definitions and flow blocks.
- **Why it fails:** A monolithic policy file is unauditable, untestable in isolation, and creates merge conflicts when multiple teams update policies concurrently.
- **Instead:** Split policy files by domain. Use NeMo Guardrails' multi-file configuration support — the runtime merges all `.co` files in the config directory at load time.

### Guardrails as a Deployment Afterthought
- **What it looks like:** Guardrail policies are written after the LLM application is deployed to production, in response to an incident.
- **Why it fails:** Reactive policy authoring misses the full attack surface. The policy author has no visibility into the distribution of real user inputs at the time of authoring.
- **Instead:** Write guardrail policies during the design phase, alongside the application prompt. Run the first red team exercise before the beta launch, not after.

### Hardcoded Term Lists in Code
- **What it looks like:** Blocked vocabulary lists and required disclaimer tokens are defined as Python constants in the application code rather than in externally managed configuration files.
- **Why it fails:** Term lists change frequently (new competitors, new regulatory requirements). Hardcoding requires a code deployment for every policy update, increasing MTTR for compliance issues.
- **Instead:** Load term lists from environment-configurable file paths or a configuration service. Policy updates should not require application redeployment.

---

## 16. Common Mistakes

| Symptom | Root Cause | Fix |
|---|---|---|
| All user messages are blocked, even safe ones | An input rail pattern is too broad — e.g., a regex that matches common English words | Narrow the pattern scope; test every new rule against a representative sample of 500+ real user messages before deploying |
| Rail evaluation adds 200ms+ to every request | Embedding-based fuzzy intent matching is enabled for all intents, including low-risk ones | Restrict fuzzy matching to high-sensitivity intents only; use exact-match patterns for the majority of rules |
| Colang policy changes break existing flows unexpectedly | Intent pattern names were reused across `.co` files, causing silent overwrite in the merged ruleset | Enforce unique intent names with a CI lint step; use a namespace prefix per domain file (e.g., `investment.ask_about_returns`) |
| Output rail injects disclaimer into non-investment responses | The investment vocabulary term list is too broad, matching general financial terms like "return" or "risk" | Refine term lists with multi-word phrases rather than single tokens; use phrase matching rather than token matching |
| Guardrail logs contain raw PII from user messages | Rail evaluation logs the full message text for debugging purposes | Log only a SHA-256 hash of the triggering message segment; store full logs in a PII-controlled audit store with restricted access |

---

## 17. Production Checklist

- [ ] Input rails are defined and tested for all known high-risk intent categories
- [ ] Output rails enforce all compliance-required content (disclaimers, citations, format constraints)
- [ ] Every blocked flow has an explicit, user-friendly fallback response
- [ ] Colang policy files are version-controlled in the same repository as the application
- [ ] A red team test suite (minimum 50 adversarial inputs) runs in CI on every policy commit
- [ ] Per-rail block rate is monitored with alerts for anomalous deviations
- [ ] Rail evaluation latency is instrumented (P50/P95) and included in application SLA dashboards
- [ ] PII is not written to guardrail evaluation logs
- [ ] Term lists and pattern files are loaded from external config, not hardcoded
- [ ] The Colang simulator has been used to manually validate all new flows before deployment
- [ ] A human escalation path is defined for unmatched high-risk intents
- [ ] False positive rate per rail is tracked and reviewed monthly
- [ ] Unicode normalisation is applied to all inputs before pattern matching
- [ ] Model upgrade runbook includes re-validation of all guardrail policies against the new model
- [ ] Compliance team has reviewed and signed off on policy files covering regulated domains

---

## 18. References

```
[1] NVIDIA NeMo Guardrails (2024). "NeMo Guardrails Documentation".
    https://docs.nvidia.com/nemo/guardrails/

[2] NVIDIA NeMo Guardrails GitHub Repository (2024).
    https://github.com/NVIDIA/NeMo-Guardrails

[3] Guardrails AI (2024). "Guardrails AI Documentation".
    https://www.guardrailsai.com/docs

[4] OWASP (2023). "OWASP Top 10 for Large Language Model Applications".
    https://owasp.org/www-project-top-10-for-large-language-model-applications/

[5] Rebedea, T. et al. (2023). "NeMo Guardrails: A Toolkit for Controllable and
    Safe LLM Applications with Programmable Rails". arXiv:2310.10501

[6] Perez, F. & Ribeiro, I. (2022). "Ignore Previous Prompt: Attack Techniques
    for Language Models". arXiv:2211.09527
```

---

## 19. Summary

Deterministic guardrails solve a fundamental reliability problem in LLM safety: you cannot use a probabilistic system to reliably constrain another probabilistic system. NeMo Guardrails' Colang-based approach enforces conversation safety through pattern matching, canonical dialogue flows, and schema validation — mechanisms that produce identical decisions for identical inputs, regardless of model state. The key architectural insight is that input and output rails must be independent layers: input rails prevent unsafe prompts from reaching the LLM at all, while output rails catch failures that survive input screening. For production deployments in regulated industries, deterministic guardrails are the minimum viable safety architecture — not an enhancement, but a prerequisite. The maintenance overhead of keeping rule sets current is the real operational cost, and teams that treat policy files with the same discipline as application code consistently outperform those that treat them as static configuration.

---

## 20. Exercises

**Beginner:** Run the PoC with a different blocked input (e.g., change the competitor name in `sample_input.json`). Observe which rail fires and what canned response is returned.

**Intermediate:** Add a new output rail that requires any response mentioning "interest rate" to include the text "[RATE_DISCLAIMER]". Verify that it fires on qualifying responses and does not fire on unrelated responses.

**Advanced:** Extend the PoC to support a two-step investment advice flow: the first bot turn returns a disclosure prompt; the second turn calls the LLM only after the user has acknowledged the disclosure. Test that skipping the acknowledgment step causes the flow to loop back.

**Expert:** Benchmark the rail evaluation latency for three configurations: (1) regex-only input rails, (2) regex + embedding-based fuzzy matching, (3) regex + a fine-tuned intent classifier. Report P50 and P95 latency for each at 100 requests/second. Identify the crossover point where fuzzy matching's recall improvement justifies its latency cost.

**Research:** Read Rebedea et al. (2023), "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails" (arXiv:2310.10501). Identify one limitation of the Colang intent matching approach not discussed in this document. Propose a mitigation and describe the trade-off it introduces.

---

## 21. Interview Questions

**Conceptual**
1. Explain deterministic guardrails to a non-engineer using an analogy from physical security (e.g., airport screening).
2. What is the difference between a probabilistic safety evaluator and a deterministic guardrail, and when is each approach appropriate?

**Technical**
3. What happens when a user message matches both a blocked input rail intent and a valid canonical flow intent? How does NeMo Guardrails resolve the conflict?
4. How does the Colang flow engine maintain state across a multi-turn conversation, and what are the implications for horizontally scaled deployments?

**Design**
5. Design a guardrail architecture for a healthcare triage chatbot that must never provide specific medication dosages but must still answer general health questions. What rails would you define, and how would you measure their effectiveness?
6. How would you architect a guardrail policy update pipeline for a team of 20 engineers where any policy change requires compliance team approval before reaching production?

**Debugging**
7. A production system using NeMo Guardrails suddenly shows a 15% increase in input rail block rate with no policy changes deployed. What are the three most likely root causes and how would you diagnose each?
8. Your output rail is injecting a disclaimer into responses that do not require it, causing user confusion. Walk through your process for identifying which term in the vocabulary list is causing the false match.

**Trade-offs**
9. When would you choose an LLM-based output evaluator over deterministic output rails, despite the reliability disadvantage?
10. A deterministic guardrail blocks a request that a domain expert considers legitimate. The fix requires either broadening the rule (increasing false negative risk) or adding a user bypass mechanism (introducing a new attack surface). How do you decide?
