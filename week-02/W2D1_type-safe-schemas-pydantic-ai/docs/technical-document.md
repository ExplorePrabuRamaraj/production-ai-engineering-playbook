# W2D1 — Type-Safe Schemas with Pydantic AI
## AI Engineering Production Playbook — Week 2, Day 1

**Vertical:** Prompt Engineering & Schemas
**Series Position:** Week 2 of 4 | Day 1 of 7
**Prerequisites:** W1D1 (DSPy & Programmatic Prompts), W1D7 (LLM-as-a-Judge Evals)

---

## 1. Overview

Large language models return plain text. When your application expects structured data — a JSON object, a typed record, a validated schema — you are responsible for bridging that gap. **Type-safe schema enforcement** is the practice of defining a strict data contract at the LLM output boundary and rejecting or retrying any response that violates it. Pydantic AI is a Python framework that makes this boundary explicit, composable, and production-ready. Unlike ad-hoc JSON parsing, it integrates schema validation, automatic retry on failure, and agent tool-call type enforcement into a single coherent abstraction. This technique is production-relevant now because LLM-powered pipelines are increasingly multi-step: a malformed output at step 2 corrupts every downstream step silently, making root-cause analysis disproportionately expensive.

---

## 2. Learning Objectives

By the end of this document you will be able to:

1. **Explain** why unvalidated LLM output is a reliability risk in production pipelines
2. **Distinguish** between optimistic JSON parsing and schema-enforced structured output
3. **Implement** a Pydantic AI agent that returns a validated typed response
4. **Apply** field-level validators to enforce domain-specific constraints on LLM output
5. **Design** a retry strategy for partial schema failures without restarting full pipelines
6. **Evaluate** the latency and cost trade-offs of schema validation with automatic retry
7. **Build** unit tests that verify schema enforcement behaviour offline
8. **Benchmark** malformed-output rates before and after schema enforcement

---

## 3. Problem Statement

Every LLM call returns a string. When application code expects structured data, the most common approach is to prompt the model to "respond in JSON" and then call `json.loads()` on the result. This approach has three distinct failure modes in production:

**Structural failure:** The model includes prose before or after the JSON block, or uses single quotes instead of double quotes, causing `json.loads()` to raise an exception. The exception propagates upstream as an unhandled error.

**Schema drift failure:** The model returns valid JSON but uses a different key name than expected (`user_name` instead of `username`) or omits an optional field that downstream code accesses unconditionally. The failure is a `KeyError` or `AttributeError` that appears far from the LLM call in the stack trace.

**Type failure:** The model returns a number as a string (`"42"` instead of `42`), or a boolean as a string (`"true"` instead of `true`). Downstream arithmetic operations silently produce wrong results or raise `TypeError`.

In high-volume systems processing 50,000 LLM responses per day, a 2% malformed-output rate produces 1,000 failed records daily. Without schema enforcement, these failures are invisible until a downstream SLA violation surfaces them — often hours later.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: Insurance Claim Triage Pipeline

An insurance company runs an LLM pipeline that extracts structured fields from unstructured claim narratives: `claim_id`, `incident_date`, `damage_category` (enum), `estimated_value` (float), and `priority_flag` (boolean). The pipeline uses `json.loads()` with no schema validation.

Over three months of production operation, the team observes:
- 3.1% of responses omit `incident_date` because the model paraphrases it as `date_of_incident`
- 1.4% of responses return `estimated_value` as a formatted string (`"$4,200.00"`) rather than a float
- 0.6% of responses include markdown code fences around the JSON, breaking parsing entirely

The combined 5.1% failure rate causes 255 claims per day to enter a manual review queue that was budgeted for 50 entries. The backlog grows faster than the team can clear it, resulting in regulatory SLA violations within 60 days.

**Root cause:** No schema enforcement at the LLM boundary. Failures are discovered downstream, not at the source.

### Scenario B — The Solution: Schema-Enforced Extraction with Pydantic AI

The same insurance pipeline is refactored to use a Pydantic AI agent with a `ClaimExtraction` model:

```python
from pydantic import BaseModel, field_validator
from datetime import date
from enum import Enum

class DamageCategory(str, Enum):
    PROPERTY = "property"
    VEHICLE = "vehicle"
    LIABILITY = "liability"
    OTHER = "other"

class ClaimExtraction(BaseModel):
    claim_id: str
    incident_date: date
    damage_category: DamageCategory
    estimated_value: float
    priority_flag: bool

    @field_validator("estimated_value")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("estimated_value must be non-negative")
        return v
```

Pydantic AI instructs the model to produce output matching this schema and automatically retries with a correction hint when validation fails. Results after refactoring:

- Malformed-output rate drops from 5.1% to 0.3% (responses that exceed the retry limit)
- Manual review queue averages 15 entries per day (down from 255)
- Root-cause visibility: every retry is logged with the specific validation error, enabling targeted prompt improvement

---

## 5. Solution Architecture

Pydantic AI sits between your application and the LLM provider. It wraps the model call in a structured output loop: build a prompt that communicates the schema to the model, call the model, attempt to parse and validate the response against the Pydantic model, and if validation fails, inject the validation error back into the conversation and retry.

The key architectural decisions are:

**Schema communication:** Pydantic AI serialises the Pydantic model's JSON schema and injects it into the system prompt. Modern frontier models (GPT-4o, Claude 3.5, Gemini 1.5) understand JSON Schema well enough that this alone reduces malformed-output rates significantly.

**Retry strategy:** On validation failure, the framework appends the validation error message to the conversation as a user turn and calls the model again. This is more token-efficient than resetting the conversation: the model has context about what it returned and what was wrong.

**Tool call typing:** When agents invoke tools, argument types are validated by Pydantic before execution. An agent cannot call a tool with `user_id="abc"` if the tool signature declares `user_id: int`.

**Provider independence:** Pydantic AI abstracts over OpenAI, Anthropic, Google, and local Ollama models. Schema enforcement works identically across providers.

See the architecture diagram below for the component relationships.

---

## 6. Internal Working Mechanics

### 6.1 Schema Serialisation

When you define a `BaseModel` subclass and pass it as the `result_type` to a Pydantic AI agent, the framework calls `model.model_json_schema()` to produce a JSON Schema dict. This schema is formatted as a string and appended to the system prompt in a structured block:

```
Respond with a JSON object that matches this schema:
{"type": "object", "properties": {"claim_id": {"type": "string"}, ...}, "required": [...]}
```

For models that support native structured output (OpenAI's `response_format={"type": "json_schema", ...}`), Pydantic AI uses the provider's native mechanism instead, which constrains token sampling at the decoding level and eliminates most structural failures entirely.

### 6.2 Validation Loop

The validation loop executes as follows:

1. Build messages list (system prompt with schema + user message)
2. Call the model API
3. Extract the text content from the response
4. Attempt `json.loads()` on the content
5. If JSON parsing fails: append error to messages, go to step 2
6. If JSON parsing succeeds: attempt `ResultModel.model_validate(parsed_dict)`
7. If Pydantic validation fails: append the `ValidationError` message to messages, go to step 2
8. If validation succeeds: return the validated `ResultModel` instance
9. If retries exhausted: raise `UnexpectedModelBehavior`

The default retry limit is 1. You can increase it via `retries=N` on the agent constructor. Each retry consumes additional tokens, so setting `retries=3` can up to quadruple token usage for a malformed response.

### 6.3 Field Validators

Pydantic validators run after the base type check. They receive the already-coerced value and can raise `ValueError` to trigger a validation error. Pydantic AI propagates the `ValueError` message verbatim into the retry prompt, giving the model precise feedback:

```
ValidationError: 1 validation error for ClaimExtraction
estimated_value
  Value error, estimated_value must be non-negative [type=value_error]
```

This specificity is important: a vague retry prompt ("the response was invalid") produces worse corrections than a targeted one ("estimated_value must be non-negative").

### 6.4 Agent Tool Call Validation

When a Pydantic AI agent calls a tool (a Python function decorated with `@agent.tool`), the framework validates the arguments against the function's type annotations before calling the function. If the LLM generates `{"user_id": "abc"}` for a tool that expects `user_id: int`, the framework raises a `ModelRetry` exception internally and appends a correction hint to the conversation.

### 6.5 Streaming Structured Output

For streaming responses, Pydantic AI accumulates the streamed tokens into a buffer and only attempts validation once the stream completes. Partial validation is not currently supported — the full response must arrive before schema checking begins.

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd`.

```
Application → Pydantic AI Agent → [Schema Serialiser] → [Prompt Builder]
                                                              ↓
                                                         LLM Provider
                                                              ↓
                                                    [JSON Parser + Validator]
                                                         ↓           ↓
                                                    Valid Result   Retry Loop
```

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd`.

---

## 9. Implementation Guide

### Step 1: Install dependencies

```bash
pip install pydantic-ai>=0.0.13 openai>=1.30.0 pydantic>=2.0.0
```

### Step 2: Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Step 3: Define the output schema

```python
from pydantic import BaseModel, field_validator
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ReviewAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float          # 0.0 – 1.0
    key_topics: list[str]      # extracted topics, min 1
    summary: str               # max 100 chars

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return round(v, 3)

    @field_validator("summary")
    @classmethod
    def summary_not_too_long(cls, v: str) -> str:
        if len(v) > 100:
            raise ValueError(f"summary exceeds 100 chars: {len(v)}")
        return v
```

### Step 4: Create the agent

```python
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    result_type=ReviewAnalysis,
    system_prompt=(
        "You are a product review analyser. "
        "Extract sentiment, confidence, key topics, and a brief summary."
    ),
    retries=2,           # allow up to 2 retry attempts on validation failure
)
```

### Step 5: Run the agent

```python
import asyncio

async def analyse(review_text: str) -> ReviewAnalysis:
    result = await agent.run(review_text)
    return result.data   # result.data is a validated ReviewAnalysis instance

# Synchronous wrapper for scripts
result = asyncio.run(analyse("The battery lasts 3 days. Very happy with this purchase."))
print(result.sentiment)      # Sentiment.POSITIVE
print(result.confidence)     # 0.95
print(result.key_topics)     # ["battery life", "satisfaction"]
```

### Step 6: Run the PoC

```bash
# Demo mode (no API key)
DEMO_MODE=true python src/main.py

# Live mode
python src/main.py
```

---

## 10. Benefits and Trade-offs

| Benefit | Trade-off |
|---|---|
| Malformed-output rate drops from ~5% to ~0.3% | Each retry consumes additional tokens (up to 3× for complex schemas) |
| Schema failures surface at the LLM boundary, not deep in the pipeline | Pydantic AI adds ~5–15 ms overhead per call for schema serialisation and validation |
| Field-level validation errors produce targeted, actionable retry prompts | Overly strict validators increase retry frequency and inflate cost |
| Typed tool arguments eliminate a full class of agent runtime errors | Schema complexity must be managed — deeply nested models confuse smaller models |
| Provider-agnostic: same schema code works across OpenAI, Anthropic, Ollama | Native structured output (OpenAI json_schema mode) is provider-specific |
| Validation errors are logged with precise field names for prompt improvement | Teams must monitor retry rates as a new production metric |

---

## 11. Performance Characteristics

**Latency overhead (schema validation path, no retry):**
- Schema serialisation and prompt injection: ~2–5 ms
- Pydantic model validation: ~1–3 ms for models with up to 20 fields
- Total overhead vs. raw string call: 3–8 ms (P50), negligible at P95

**Latency impact of retries:**
- Each retry adds one additional full LLM round-trip
- For GPT-4o-mini: ~400–800 ms additional per retry (P50)
- For Claude 3.5 Haiku: ~300–600 ms additional per retry (P50)
- At `retries=2`, worst-case latency is 3× the base call latency

**Token cost of retries:**
- Retry messages include the full conversation history plus the validation error
- Each retry adds approximately 50–200 tokens (error message + schema reminder)
- For a complex schema with 10 fields, worst-case retry overhead: ~300 extra tokens per call

**Retry frequency benchmarks (from Pydantic AI documentation examples):**
- GPT-4o with native JSON schema mode: <0.5% retry rate
- GPT-4o-mini with prompt-injected schema: ~1–3% retry rate
- Claude 3.5 Sonnet with prompt-injected schema: ~0.5–1% retry rate
- Smaller local models (Llama 3.1 8B): ~10–20% retry rate

**Throughput:** Schema validation is CPU-bound and adds negligible throughput constraint. The bottleneck remains the LLM API rate limit.

---

## 12. Security Considerations

**OWASP LLM Top 10 — LLM01: Prompt Injection**
Schema enforcement provides a partial defence against prompt injection in structured output scenarios. If a user input contains text like `}, "priority_flag": true, "injected_field": "`, Pydantic validation will reject the response because `injected_field` is not in the schema (with `model_config = ConfigDict(extra="forbid")`). Always set `extra="forbid"` on output models to prevent schema injection via unexpected fields.

```python
from pydantic import BaseModel, ConfigDict

class SafeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_one: str
    field_two: int
```

**OWASP LLM Top 10 — LLM02: Insecure Output Handling**
Validated output is not automatically safe for downstream use. A `summary: str` field that passes Pydantic validation may still contain HTML injection (`<script>`) or SQL fragments. Apply domain-specific sanitisation after schema validation, not instead of it.

**Input validation before prompt construction:**
Sanitise user-supplied strings before including them in the prompt. Strip or escape characters that could alter the schema instruction block (e.g., `---`, `###`, triple backticks).

**Secret exposure in error messages:**
Pydantic validation errors include field values in some configurations. Ensure that error messages logged in production do not contain PII or sensitive field values from the LLM response.

---

## 13. Cost Analysis

**Baseline (no schema enforcement, 50,000 calls/day, GPT-4o-mini):**
- Average tokens per call: 500 input + 200 output = 700 tokens
- Daily token volume: 35,000,000 tokens
- Daily cost at $0.15/1M input + $0.60/1M output: ~$6.45/day

**With schema enforcement (1.5% retry rate, 1 retry each):**
- Additional tokens per retried call: ~250 tokens
- Additional calls per day: 750 retries × 250 tokens = 187,500 tokens
- Additional daily cost: ~$0.03/day (< 0.5% cost increase)

**Cost of not using schema enforcement:**
- 5.1% malformed rate × 50,000 calls = 2,550 failed records/day
- Manual review cost: $15/record × 2,550 = $38,250/day in labour (insurance example)
- Schema enforcement cost: $0.03/day
- **ROI: schema enforcement pays for itself within minutes of production traffic**

---

## 14. Best Practices

1. **Set `extra="forbid"` on all output models.** This rejects responses with unexpected fields, which is both a correctness guarantee and a prompt-injection defence.

2. **Keep output models shallow.** Models with more than 3 levels of nesting confuse smaller models significantly. Flatten nested structures into a single level where possible, or extract sub-objects as separate agent calls.

3. **Use enums for categorical fields.** An `Enum` field generates a tight constraint in the injected schema (`"enum": ["positive", "negative", "neutral"]`) that dramatically reduces hallucination in that field.

4. **Write validator error messages as model instructions.** The error message is fed verbatim to the model on retry. Write it as an instruction: "confidence must be between 0.0 and 1.0" not "invalid value".

5. **Monitor retry rate as a production metric.** A rising retry rate indicates schema drift (the model changed its output format) or a prompt regression. Alert when retry rate exceeds 5%.

6. **Use native structured output for OpenAI models when available.** The `response_format={"type": "json_schema", ...}` API constrains token sampling and reduces malformed-output rates to near-zero at no extra cost.

7. **Set `retries=2` as the default and `retries=0` for latency-critical paths.** For interactive user-facing calls, a retry doubles latency. Use fallback defaults for latency-critical paths instead.

8. **Test validators with adversarial LLM responses.** In unit tests, feed your validators malformed inputs that a real model could plausibly produce (wrong types, out-of-range numbers, extra fields) to verify the retry prompt would be meaningful.

9. **Log the full validation error and the offending response.** This data drives targeted prompt improvements. A field that retries 3× every day indicates a prompt instruction that the model consistently misunderstands.

10. **Version your output schemas.** When a `BaseModel` changes (fields added, types changed), old cached or batched responses may no longer validate. Use schema versioning if you process historical data alongside live data.

---

## 15. Anti-Patterns

### Anti-Pattern 1: The Optimistic Parser
**What it looks like:** `result = json.loads(llm_response.content)` followed by direct field access with no validation.
**Why it fails:** Any of the three failure modes (structural, schema drift, type) propagates as an unhandled exception or silent wrong value into downstream logic.
**Fix:** Always validate against a Pydantic model before accessing fields.

### Anti-Pattern 2: The God Schema
**What it looks like:** A single `BaseModel` with 30+ fields covering every possible output variant, with most fields `Optional`.
**Why it fails:** Larger schemas produce longer system prompts, increasing token cost and confusing models into omitting required fields. Optional fields are frequently left null when the model should have populated them.
**Fix:** Split large schemas into focused sub-schemas. Use separate agent calls for logically distinct extraction tasks.

### Anti-Pattern 3: The Silent Fallback
**What it looks like:** Catching `ValidationError` and returning a default object silently when retries are exhausted.
**Why it fails:** The failure is invisible. Downstream code processes default values as if they were real extractions, producing incorrect business decisions.
**Fix:** On exhausted retries, log the failure with the original response and validation error, then either raise or route to a manual review queue.

### Anti-Pattern 4: The Validator as Business Logic
**What it looks like:** A validator that calls an external database to check if a `claim_id` exists, or calls another API to validate a postal code.
**Why it fails:** Validators run synchronously in the Pydantic validation step. External I/O in validators blocks the event loop and adds latency that compounds with retry overhead.
**Fix:** Keep validators pure and fast. Perform external validation in a separate step after schema validation succeeds.

### Anti-Pattern 5: Prompt-Only Schema Enforcement
**What it looks like:** Including the schema in the prompt as English prose ("please return a JSON object with the following fields: ...") instead of injecting the actual JSON Schema.
**Why it fails:** English schema descriptions are ambiguous. Models interpret "a list of strings" differently from `"type": "array", "items": {"type": "string"}`. Malformed-output rates are 3–10× higher with prose descriptions.
**Fix:** Use a framework (Pydantic AI, instructor) that injects the machine-readable JSON Schema, or use the provider's native structured output API.

### Anti-Pattern 6: Ignoring Retry Rate
**What it looks like:** Schema enforcement is deployed and the retry rate is never monitored.
**Why it fails:** A 15% retry rate (common with smaller models and complex schemas) means 15% of calls are paying double token cost. This goes undetected until the invoice arrives.
**Fix:** Emit a counter metric on every retry. Set an alert threshold at 5%.

---

## 16. Common Mistakes

**Mistake 1: Pydantic v1 model syntax with Pydantic v2 runtime**
- Symptom: `@validator` decorator raises `PydanticUserError` or silently does nothing
- Root cause: Pydantic v2 deprecated `@validator` in favour of `@field_validator`. The v1 decorator is available in v2 via a compatibility shim but behaves differently.
- Fix: Replace `@validator("field")` with `@field_validator("field")` and add `@classmethod`. Run `python -m pydantic.v1 --upgrade` to catch remaining v1 patterns.

**Mistake 2: Using `Optional` fields without `None` defaults**
- Symptom: `ValidationError: field required` for a field you thought was optional
- Root cause: In Pydantic v2, `Optional[str]` means `str | None` (allows `None`) but still requires the field to be present in the input. To make a field truly optional, use `Optional[str] = None`.
- Fix: Always pair `Optional[T]` with `= None` for fields that may be absent from LLM output.

**Mistake 3: Synchronous agent calls blocking the event loop**
- Symptom: FastAPI endpoint hangs or times out when calling the Pydantic AI agent
- Root cause: Calling `agent.run_sync()` inside an async FastAPI route blocks the event loop during the LLM call.
- Fix: Use `await agent.run()` inside async routes. Reserve `agent.run_sync()` for CLI scripts and test fixtures.

**Mistake 4: Not accounting for retry tokens in cost estimates**
- Symptom: Actual token bill is 20–40% higher than projected
- Root cause: Cost estimates counted one call per request but did not account for retry volume
- Fix: Instrument the retry counter and include retry token consumption in cost projections. Assume a 3–5% retry rate for production estimates with GPT-4o-mini.

**Mistake 5: Treating schema validation as security sanitisation**
- Symptom: Validated output is passed directly to a SQL query or HTML renderer
- Root cause: Pydantic validates structure and types, not content safety. A `name: str` field that validates successfully may contain `Robert'); DROP TABLE users;--`.
- Fix: Schema validation and content sanitisation are separate concerns. Apply both.

---

## 17. Production Checklist

- [ ] Output model has `model_config = ConfigDict(extra="forbid")` to reject unexpected fields
- [ ] All categorical fields use `Enum` types rather than unconstrained `str`
- [ ] Field validators are pure functions with no external I/O
- [ ] Validator error messages are written as model-directed instructions
- [ ] Retry limit is set explicitly (`retries=N`) and documented
- [ ] Retry rate is emitted as a counter metric to your observability platform
- [ ] Alert configured when retry rate exceeds 5% over a 10-minute window
- [ ] Exhausted-retry path raises or routes to a fallback queue — never silently returns defaults
- [ ] Full validation errors are logged with the offending LLM response for prompt debugging
- [ ] Native structured output mode enabled where the provider supports it
- [ ] Output schemas are versioned alongside the application code
- [ ] Unit tests cover at least 3 adversarial LLM response patterns per schema
- [ ] Cost projections include retry overhead at the observed retry rate
- [ ] PII fields are not included in validation error log entries

---

## 18. References

[1] Pydantic AI (2024). "Structured Results". Official Documentation. https://ai.pydantic.dev/results/

[2] Pydantic AI (2024). "Agents". Official Documentation. https://ai.pydantic.dev/agents/

[3] Liu, J. et al. (2023). "instructor: Structured LLM Outputs". GitHub. https://github.com/jxnl/instructor

[4] OpenAI (2024). "Structured Outputs". OpenAI Platform Documentation. https://platform.openai.com/docs/guides/structured-outputs

[5] OWASP (2025). "OWASP Top 10 for Large Language Model Applications". https://owasp.org/www-project-top-10-for-large-language-model-applications/

[6] Pydantic (2024). "Pydantic V2 Migration Guide". https://docs.pydantic.dev/latest/migration/

[7] Willard, B. T., & Louf, R. (2023). "Efficient Guided Generation for Large Language Models". arXiv:2307.09702

---

## 19. Summary

Type-safe schemas with Pydantic AI address a fundamental reliability gap in LLM-powered pipelines: the mismatch between a model's text output and the structured data your application code expects. By enforcing a contract at the model boundary — validating structure, types, and domain constraints before the response reaches application logic — you convert a class of silent data corruption into visible, retryable, loggable errors. The cost of enforcement is minimal (3–8 ms overhead, <0.5% token overhead at typical retry rates). The cost of not enforcing is substantial: malformed-output rates of 3–5% translate directly into operational failures, manual review queues, and SLA violations. Production systems should treat schema enforcement as non-negotiable infrastructure, not an optimisation.

---

## 20. Exercises

**Beginner:** Run the PoC in demo mode (`DEMO_MODE=true python src/main.py`) and examine the sample output. Identify which fields would have failed a raw `json.loads()` call.

**Intermediate:** Add a new field `word_count: int` to the `ReviewAnalysis` model in `src/pydantic_schemas_core.py`. Add a validator that rejects values below 1. Run the tests to verify the validator triggers correctly.

**Advanced:** Extend the PoC to handle a batch of 10 reviews from `sample_input.json`. Track the retry count per item and print a summary of retry rates at the end. Simulate a 10% malformed-output rate by monkey-patching the demo output generator.

**Expert:** Benchmark the malformed-output rate across three models (GPT-4o, GPT-4o-mini, a local Ollama model) on a fixed set of 100 review texts. Use `retries=0` to count raw failure rates. Compare costs and failure rates in a summary table.

**Research:** Read Willard & Louf (2023), arXiv:2307.09702, on constrained decoding for structured generation. Identify one limitation of grammar-constrained generation that Pydantic AI's retry approach does not share, and one limitation of the retry approach that constrained decoding solves.

---

## 21. Interview Questions

**Conceptual:**
1. Explain the difference between Pydantic schema validation and OpenAI's native `json_schema` structured output mode to a non-engineer. When would you choose each?
2. Why does setting `extra="forbid"` on an output model provide a partial defence against prompt injection?

**Technical:**
3. What happens to a Pydantic AI agent call when `retries=1` and the model returns structurally invalid JSON twice in a row?
4. A `field_validator` raises `ValueError` for a field value. Describe the exact sequence of events that follows inside the Pydantic AI runtime.
5. A field is declared as `Optional[date] = None`. The LLM returns `"incident_date": "January 5th 2024"`. What does Pydantic do with this value?

**Design:**
6. You are designing a schema for extracting 40 fields from a legal document. How do you structure the Pydantic models to minimise retry rates with GPT-4o-mini?
7. How would you architect a schema validation system that routes to different fallback handlers depending on which field failed validation?

**Trade-off:**
8. When would you choose `retries=0` and a fallback default over `retries=2` with a strict validator? What production metrics would inform this decision?
9. A team argues that using `instructor` instead of Pydantic AI gives them more control over retry prompts. What is the strongest argument for each approach?

**Debugging:**
10. A production pipeline shows a 12% retry rate on a Monday after a weekend with no code changes. What are the three most likely root causes and how would you investigate each?
