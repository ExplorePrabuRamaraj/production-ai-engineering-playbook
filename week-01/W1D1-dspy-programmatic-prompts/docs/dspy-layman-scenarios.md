# DSPy in Simple Words — Real-World QA Scenarios

> A layman-friendly walkthrough of DSPy concepts using four everyday business problems.
> No ML background needed — if you've ever written a form, trained a new employee, or asked someone to "show their work", you already understand DSPy.

---

## The Core Idea (Before the Scenarios)

Think of a traditional prompt like a sticky note you hand to a new employee:

> *"Hey, when a customer asks something, figure it out and reply nicely."*

It works — until the employee changes, the context shifts, or the sticky note gets crumpled. You have no way to know if the instruction is still being followed correctly.

**DSPy replaces the sticky note with a structured process:**

| DSPy Concept | Layman Analogy |
|---|---|
| **Signature** | A standardised form — specific fields to fill in, nothing ambiguous |
| **ChainOfThought** | "Show your working" — reason first, answer second |
| **BootstrapFewShot** | Training with real past examples, not just instructions |
| **teleprompter.compile()** | A manager reviews 50 past cases and picks the best examples to guide future work |

---

## Scenario 1 — Customer Support Bot

### Problem Statement

An e-commerce company receives 3,000 customer queries per day — order status, return requests, product complaints. They built a support bot using a hand-written prompt:

> *"You are a helpful support agent. Read the customer's message and reply with a resolution."*

It worked well for six months. Then the AI vendor updated their model. Overnight, 20% of responses stopped including the resolution steps — just sympathy, no action. Nobody noticed for two weeks. 400 customers were left unresolved.

### Solution — With Layman Understanding

Instead of a free-form instruction, DSPy defines a **Signature** — a form the AI must fill in completely:

```
Input  : customer_query
Outputs: issue_category, resolution_steps, estimated_resolution_time, tone
```

**Layman version:** Imagine replacing the sticky note with a structured ticket form. Every response must have a category (e.g., "Refund Request"), step-by-step resolution instructions, a time estimate, and a tone check. The AI cannot submit a "half-filled form."

**BootstrapFewShot** is then run against 100 past tickets where human agents gave excellent resolutions. It picks the 4 best examples and automatically includes them as guidance in every future prompt — without anyone writing example text manually.

### Outcome

- Response format is consistent regardless of model updates
- Resolution steps are always present — no more half-answers
- When the vendor updates their model, recompilation takes 15 minutes and restores full accuracy

### Benefits

- **Reliability** — format is enforced by the program, not by trusting the model's interpretation
- **Speed** — BootstrapFewShot finds better examples than a human would pick manually
- **Resilience** — model swaps no longer require rewriting the prompt from scratch

### Best Practices

- Define one Signature per task type (refund, delivery, complaint) — don't bundle all query types into one signature
- Include a `tone` output field to catch aggressive or off-brand responses before they reach the customer
- Use real resolved tickets as your training set, not synthetic ones

---

## Scenario 2 — Medical Symptom Triage

### Problem Statement

A clinic built a tool to help triage nurses pre-classify patient symptom descriptions into urgency levels before the doctor's review. The original prompt was:

> *"Read the symptoms and classify urgency as low, medium, high, or emergency."*

The model often returned answers like "The patient seems to be experiencing moderate discomfort, suggesting a medium-high urgency situation." The nurse had to read the full sentence to extract the classification — and sometimes the classification was buried or missing entirely.

Worse, the model would jump to a classification without explaining why, making it impossible for a nurse to quickly check if the reasoning was sound.

### Solution — With Layman Understanding

DSPy introduces two changes:

**1. Signature** — forces the output to be structured:
```
Input  : symptom_description
Outputs: clinical_reasoning, urgency_level (low / medium / high / emergency), recommended_action
```

**2. ChainOfThought** — forces the model to reason before classifying:

**Layman version:** This is like a doctor's checklist. Before ticking "high urgency", the doctor must write down which symptoms led to that decision. ChainOfThought enforces the same rule on the AI — it must fill in `clinical_reasoning` before it is allowed to fill in `urgency_level`. If the reasoning doesn't support the classification, the inconsistency becomes visible immediately.

### Outcome

- Every triage response includes explicit reasoning the nurse can scan in 5 seconds
- Urgency level is always one of four defined values — no ambiguous middle-ground sentences
- Nurses caught two incorrect high→medium downgrades in the first week because the reasoning field made the error visible

### Benefits

- **Auditability** — reasoning is logged alongside every classification; you can review why the model decided what it decided
- **Safety** — a wrong classification is now easier to catch because the reasoning is explicit, not hidden
- **Consistency** — `urgency_level` is always one of four exact values, making downstream alerting trivial to build

### Best Practices

- Always use `ChainOfThought` (not plain `Predict`) when the output has safety implications — the reasoning trail is non-negotiable
- Constrain output field values using `dspy.Assert` to reject any `urgency_level` response that isn't one of the four allowed strings
- Keep the training set current — retriage old cases monthly and add new examples that reflect evolving symptom patterns

---

## Scenario 3 — Product Review Analyser

### Problem Statement

A retail brand publishes 500+ customer reviews per day across product lines. The marketing team wanted a tool to extract structured insights: sentiment, specific complaints, and improvement suggestions — to feed into a weekly product quality report.

They wrote a prompt asking for a JSON response. It worked for two months. Then a model update changed how the JSON was structured — `"issues"` became `"problems"`, `"suggestions"` became `"improvements"`. The parsing pipeline broke silently. The weekly report showed zero complaints for three weeks because the parser couldn't find the old key names.

### Solution — With Layman Understanding

DSPy Signature declares the exact output fields as typed Python attributes — not prose instructions:

```
Input  : review_text, product_name
Outputs: sentiment (positive / neutral / negative),
         key_issues (comma-separated list),
         improvement_suggestions (comma-separated list),
         summary (one sentence)
```

**Layman version:** Instead of asking the AI to "return a JSON with these keys" (which it can misinterpret), you define Python variables with those exact names. The field name `key_issues` is now a contract — not a suggestion in a paragraph. If the model doesn't return it, the program raises an error immediately rather than silently producing a malformed response.

**BootstrapFewShot** is compiled against 50 manually reviewed products. It learns which past review analyses were rated as most accurate and uses those as reference examples in every future call.

### Outcome

- Zero silent parsing failures — field names are enforced at the program level
- The weekly report now processes 500 reviews in under 4 minutes with consistent structure
- When the model was updated again, recompilation took 10 minutes with no manual prompt changes

### Benefits

- **Data quality** — downstream systems (dashboards, reports) receive consistently structured data every time
- **Debuggability** — when a review produces an unexpected output, the exact field that failed is immediately visible
- **Scalability** — the same Signature works across all product categories without per-category prompt customisation

### Best Practices

- Use comma-separated strings for list outputs rather than JSON arrays — easier for the model to generate reliably and simpler to parse
- Set `max_bootstrapped_demos=4` to start; only increase it if accuracy on held-out reviews is below your quality threshold
- Validate `sentiment` values at the application boundary — even with a Signature, add a check that the value is one of the three allowed strings before it enters the database

---

## Scenario 4 — Job Description to Resume Matching

### Problem Statement

An HR technology company built a resume screening tool. Recruiters would upload a job description and a batch of resumes; the tool would score each resume from 0–10 and explain the match.

The original prompt produced scores, but recruiters complained that two resumes could both score 7/10 for completely different reasons — one missing a critical required skill, the other missing a nice-to-have. The score alone wasn't enough, and the explanation was inconsistent — sometimes a paragraph, sometimes a single sentence, sometimes missing entirely.

When pushed to hire volume (500 resumes in a week), the inconsistency made the tool less trusted than a manual review.

### Solution — With Layman Understanding

**Signature** defines four precise output fields:

```
Inputs : resume_text, job_description
Outputs: match_score (integer 0–10),
         matched_strengths (comma-separated),
         skill_gaps (comma-separated),
         hiring_recommendation (strong yes / yes / maybe / no)
```

**ChainOfThought** is layered on top — the model must list `matched_strengths` and `skill_gaps` before it is allowed to produce a `match_score`. This mirrors how a good recruiter actually thinks: gather evidence first, score second.

**Layman version:** It's the difference between a recruiter saying "I'd give this a 7" and a recruiter saying "They have Python and system design — those match. They're missing Kubernetes and team lead experience — those are required. Given that, I'd say 6/10." The second recruiter's score is auditable. DSPy enforces the second approach structurally.

**BootstrapFewShot** is compiled against 80 past resume-JD pairs where senior recruiters had already made hiring decisions. The compiler finds the 4 examples where the model's reasoning most closely matched the human recruiter's rationale and pins those as reference cases.

### Outcome

- Recruiters can scan `skill_gaps` in 3 seconds and immediately understand why a score is what it is
- `hiring_recommendation` gives a plain-English signal that non-technical hiring managers can act on directly
- False positives (high score, wrong candidate) dropped by 30% in the first month because the reasoning chain exposed flawed score justifications before they reached recruiters

### Benefits

- **Trust** — recruiters trust a scored recommendation they can verify over a black-box number
- **Fairness** — explicit `skill_gaps` and `matched_strengths` make bias easier to spot and audit
- **Consistency** — the same scoring rubric applies to every resume, eliminating per-recruiter variance in the screening stage

### Best Practices

- Keep `match_score` as an integer 0–10 in the Signature description — avoid ranges like "0.0–1.0" which invite floating-point formatting variance
- Include both required and preferred skills in the `job_description` input field with clear labels — the model uses these to differentiate gaps from nice-to-haves
- Recompile quarterly using the most recent batch of successful hires as positive training examples — hiring patterns shift with team needs

---

## Summary — What DSPy Gives You Across All Four Scenarios

| Without DSPy | With DSPy |
|---|---|
| Format breaks silently on model update | Format enforced by the program — update triggers recompilation, not debugging |
| Reasoning is hidden inside the response | ChainOfThought surfaces reasoning as a named, inspectable field |
| Training examples hand-picked by engineers | BootstrapFewShot selects optimal examples automatically from real data |
| One model change = days of prompt rewriting | One model change = 10–15 minutes of recompilation |
| Output schema lives in natural language prose | Output schema lives in typed Python — testable, versionable, refactorable |

The pattern is the same in every scenario: **define the contract, enforce the reasoning, compile from real data.**
