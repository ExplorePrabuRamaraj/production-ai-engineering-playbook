# Type-Safe Schemas with Pydantic AI in Simple Words — Real-World QA Scenarios

No ML background required — if you have ever filled out a form that rejected your input, you already understand the core idea.

---

## Core Idea

When you ask an AI model a question and expect a structured answer — like a filled-in form — the model returns plain text. It is the equivalent of emailing someone a blank form and hoping they fill it in correctly. Sometimes they do. Sometimes they write the date in the wrong format, skip a required field, or add extra notes in a box that only accepts numbers. Your application then crashes trying to read their response as if it were a perfectly completed form.

Type-safe schemas are the equivalent of replacing that email with a locked digital form. The AI can only submit a response that matches the expected structure. If it tries to put text in a number field, the system immediately tells it "that is wrong, try again" — and logs exactly what went wrong so you can fix the underlying instruction later.

Pydantic AI is the Python library that builds and enforces those locked digital forms at the boundary between your application and the AI model. It turns "trust the AI to format correctly" into "verify that the AI formatted correctly, retry if not, and raise an alert if it still cannot."

| Concept | Real-World Analogy |
|---|---|
| Pydantic BaseModel | A form template with required fields and accepted value types |
| Field validator | A form field that rejects invalid formats (e.g., phone must be 10 digits) |
| Schema injection | Attaching the blank form to the AI's instructions before it answers |
| Validation error | The "this field is required" or "invalid date format" message on a rejected form |
| Automatic retry | The form bouncing back to the sender with the specific error highlighted |
| extra="forbid" | A form that rejects submissions with extra fields not on the original template |

---

## Scenario 1: Customer Support — Ticket Classification

### Problem Statement

A retail company uses an AI to read incoming support tickets and fill in a classification card: urgency level (low/medium/high), department to route to, and whether a refund is involved. The AI's text response is parsed by the system to create database records. When the AI writes "High Priority" instead of "high", the database insert fails. When it skips the department field because it is "obvious from context", the routing system sends the ticket to the default inbox — which no one monitors.

### Solution

The team defines a Pydantic model for the classification card. The AI can only submit responses that match exactly: `urgency` must be one of three allowed values, `department` is required with no default, and `refund_involved` must be a true/false value, not the word "yes".

**Layman version:** Before, the AI was filling in a paper form and a human was re-typing it into the computer. Now the AI fills in a digital form that only accepts the right answers in the right boxes. If it writes "High Priority" instead of "high", the form bounces back immediately with the message "urgency must be one of: low, medium, high" and the AI corrects it — all in under a second, with no human involved.

### Outcome

- Ticket routing errors drop from 8% to 0.4% within the first week of deployment
- The manual re-routing queue clears from 200 tickets/day to under 10
- Every retry is logged, revealing that 90% of retries were caused by one phrase in the system prompt that the AI misread as permission to use free-form urgency labels

### Benefits

- **Immediate error visibility:** Failures surface at the AI boundary, not three steps later when the database insert crashes
- **Self-correcting pipeline:** The AI fixes its own formatting errors without human intervention
- **Continuous prompt improvement:** Retry logs show exactly which instructions produce bad output, turning every failure into a data point for improvement

### Best Practices

- Use an `Enum` type for the `urgency` field so the AI sees the exact allowed values in its instructions
- Set `retries=2` for this use case — urgency classification rarely needs more than one correction
- Log every retry with the original response so the support team can review patterns weekly

---

## Scenario 2: Healthcare — Patient Intake Form Extraction

### Problem Statement

A telehealth platform receives patient intake notes written in plain text and needs to extract structured fields: patient age, chief complaint, allergies (as a list), and whether the case is flagged for urgent review. The extraction feeds a triage queue. When the AI returns age as "forty-two" instead of `42`, the triage sorting logic throws a type error. When allergies come back as a single comma-separated string instead of a list, the medication safety check misses individual entries.

### Solution

A `PatientIntake` Pydantic model enforces that `age` is an integer, `allergies` is a list of strings (never a single string), and `urgent_review` is a boolean. A validator confirms age is within a plausible range (0–130). The AI is instructed via the schema and retries automatically if the structure does not match.

**Layman version:** Think of the AI as a nurse filling in a paper intake form and handing it to a pharmacist. If the nurse writes allergies as one long sentence — "penicillin, aspirin, latex" — and the pharmacist's system expects each allergy on its own line, the check fails silently. The schema enforcement is like a form that physically has three separate allergy boxes, forcing the nurse to split the entries before submission. The pharmacist's system never sees the comma-separated version.

### Outcome

- Medication safety check false-negative rate drops from 3.2% (caused by unparsed allergy strings) to 0%
- Triage queue type errors fall from 120/day to 2/day (edge cases involving non-standard age representations)
- Average extraction latency increases by 380 ms due to retry overhead on 1.8% of cases — accepted as a worthwhile trade-off given the safety implications

### Benefits

- **Safety-critical correctness:** A missed allergy caused by a parsing failure has direct patient harm potential; schema enforcement eliminates the entire structural failure mode
- **Auditable failures:** Every validation failure is logged with the exact field and value, creating an audit trail for compliance review
- **Type coercion transparency:** When the AI returns `"42"` (string) for age, Pydantic coerces it to `42` (integer) automatically and logs the coercion — no silent wrong-type processing

### Best Practices

- Add a `model_config = ConfigDict(extra="forbid")` to reject responses with unexpected medical fields that were not requested
- Keep the allergies field as `list[str]` with a minimum length validator of 0 — explicitly allow empty lists rather than `None` to avoid downstream null checks
- Route exhausted-retry cases to a human reviewer queue, never silently to a default record

---

## Scenario 3: Finance — Earnings Call Summarisation

### Problem Statement

An investment research firm uses an AI to summarise quarterly earnings calls into a structured report card: revenue figure (float), guidance sentiment (positive/neutral/negative), key risks (list of strings, max 5), and a one-sentence executive summary. Analysts downstream use the revenue figure in spreadsheet models. When the AI returns revenue as "$4.2B" (a string with currency symbol and abbreviation), the spreadsheet formula fails. When key risks contains 12 entries, the summary card overflows its display template.

### Solution

A `EarningsReport` Pydantic model enforces `revenue_usd_billions: float` (the AI is instructed to return a plain decimal number), `guidance_sentiment` as an `Enum`, `key_risks: list[str]` with a max-length validator capped at 5 entries, and `executive_summary: str` with a character-length validator capped at 150 characters.

**Layman version:** Imagine an analyst filling in a standard research form. The revenue box says "enter numbers only, in billions" but the analyst writes "$4.2B" out of habit. In the old system, an intern had to strip the dollar sign and "B" before entering it into the model. With schema enforcement, the form rejects "$4.2B" the moment it is submitted and displays the message: "revenue_usd_billions must be a decimal number (e.g., 4.2)". The analyst — in this case the AI — corrects it before the form leaves their desk.

### Outcome

- Spreadsheet import errors drop from 11% to 0.2% across 500 weekly earnings reports
- The data cleaning step that previously took an analyst 45 minutes per week is eliminated entirely
- The max-5 risk constraint forces the AI to prioritise, producing more focused risk lists that senior analysts rate as higher quality than the unconstrained versions

### Benefits

- **Downstream model integrity:** Financial models built on extracted data produce correct outputs only when input types are exact; schema enforcement guarantees this at the source
- **Quality improvement as a side effect:** Constraints that prevent over-generation (max 5 risks, max 150 char summary) produce more focused outputs that analysts prefer
- **Zero-maintenance data cleaning:** The manual post-processing step that cleaned malformed AI outputs is removed from the workflow entirely

### Best Practices

- Instruct the AI in the system prompt that `revenue_usd_billions` should be "a plain decimal number such as 4.2, not a formatted string such as $4.2B" — the schema enforces the type but the prompt reduces retry frequency
- Use `round(v, 2)` in the revenue validator to normalise floating-point precision before the value enters spreadsheet models
- Version the `EarningsReport` schema and store the version alongside each record to handle schema changes without corrupting historical data

---

## Scenario 4: IT Helpdesk — Automated Incident Triage

### Problem Statement

An IT operations team uses an AI to parse incoming incident tickets and populate a structured triage record: affected system (string), severity (P1/P2/P3/P4), estimated user impact count (integer), requires_escalation flag (boolean), and suggested_team (one of five fixed team names). The triage record feeds an automated paging system. When `requires_escalation` comes back as the string "yes" instead of `true`, the on-call engineer is not paged. When `suggested_team` comes back as "Network Infrastructure" instead of the valid value "networking", the routing rule does not match and the ticket sits unassigned.

### Solution

A `IncidentTriage` Pydantic model with `severity` as a `SeverityLevel` enum (P1–P4), `suggested_team` as a `TeamName` enum, `requires_escalation` as `bool`, and `user_impact_count` as `int` with a non-negative validator. The enum constraints mean the AI sees the exact valid values in its schema instructions and the framework rejects any response that uses a non-member value.

**Layman version:** Picture a fire station dispatch form. There are exactly four severity boxes — P1, P2, P3, P4 — and exactly five team checkboxes. A dispatcher cannot write "urgent" in the severity field or "Network Infrastructure" in the team field because those boxes do not exist on the form. The AI faces the same constraint: the schema physically prevents it from submitting a value that is not on the pre-approved list. When it tries "yes" for the escalation field, the form says "escalation must be true or false" and it tries again.

### Outcome

- On-call paging miss rate drops from 4.7% to 0.1% (residual cases involve ambiguous incident descriptions that exhaust retries)
- Unassigned ticket rate drops from 6.2% to 0.3%
- Mean time to acknowledge (MTTA) for P1 incidents improves from 8.3 minutes to 4.1 minutes because routing is now instant and correct

### Benefits

- **Operational safety:** Missed pages for P1 incidents have direct business impact; schema enforcement converts a reliability gap into a near-zero failure mode
- **Enum constraints as documentation:** The `SeverityLevel` and `TeamName` enums serve as the authoritative source of valid values for both the AI and the routing rules — one definition, two uses
- **Retry as diagnostic signal:** The 0.1% residual failure rate identifies the specific ticket types (ambiguous multi-system incidents) that need improved prompt handling

### Best Practices

- Define `SeverityLevel` and `TeamName` enums in a shared module imported by both the AI extraction layer and the routing rules — this ensures the two systems can never diverge
- Set `retries=1` for P1 incidents to minimise paging latency; accept the 0.1% miss rate as preferable to the 2× latency of a second retry
- Emit a separate alert when `requires_escalation=True` AND `severity=P1` both appear in a triage record, as a belt-and-suspenders safeguard independent of the paging system

---

## Summary

| Dimension | Without Type-Safe Schemas | With Type-Safe Schemas |
|---|---|---|
| Failure discovery | Deep in downstream logic, often hours later | At the AI boundary, immediately on each call |
| Failure visibility | Silent wrong values or unhandled exceptions | Logged validation errors with exact field and value |
| Recovery | Manual data cleaning or pipeline restart | Automatic retry with targeted correction hint |
| Malformed-output rate | 2–10% depending on model and schema complexity | 0.1–0.5% after retry |
| Prompt improvement | Guesswork — hard to know which instruction caused the error | Systematic — retry logs show exactly which field fails repeatedly |
| Downstream code safety | Requires defensive null checks and type guards everywhere | Output is guaranteed to match the declared type on every field |
| Cost of failure | High — operational errors, manual review, SLA violations | Low — minor token overhead for retries |
