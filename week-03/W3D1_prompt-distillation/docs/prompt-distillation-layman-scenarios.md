# Prompt Distillation in Simple Words — Real-World QA Scenarios

You already know how to write instructions — but can those instructions automatically get shorter and cheaper the more you use them?

---

## Core Idea

Imagine you are training a new employee. On day one, you hand them a 20-page manual: every policy, every edge case, every worked example. It works — they follow the manual and do a good job. But after a month you notice that only 5 of those 20 pages actually come up in daily work. The other 15 pages cover rare situations that almost never happen.

Prompt Distillation is the process of finding those 5 essential pages and creating a condensed 2-page quick-reference card that gets the same results as the original manual — for the work that actually happens every day.

In AI terms: you start with a large, carefully written prompt (the 20-page manual). You collect real examples of the task being done well. You then run a systematic search to find the shortest version of that prompt that still gets the job done at the same quality level. The result is a "distilled" prompt — smaller, faster, and cheaper to run — without meaningful loss in accuracy.

The key insight is that prompts accumulate waste over time. Engineers add instructions reactively — one failure leads to one new rule. After months of iteration, a prompt is full of defensive clauses for scenarios that rarely occur. Distillation cuts through that accumulated waste by asking: "What does this prompt actually need to say, given the real inputs it receives?"

| Concept | Analogy |
|---|---|
| Teacher prompt | The original 20-page employee manual |
| Student prompt | The 2-page quick-reference card |
| Training dataset | A month of real daily work examples |
| Metric function | Your manager scoring the work product |
| Optimization loop | HR testing 30 versions of the quick-reference card |
| Held-out eval set | A surprise quiz using tasks the employee hasn't seen |
| Compiled state file | The laminated card that goes on every desk |
| Re-distillation | Updating the card when the job role changes |

---

## Scenario 1: Customer Support — Intent Classification

### Problem Statement

A software company runs a customer support chatbot that reads incoming tickets and classifies them into one of 8 categories: Billing, Bug Report, Feature Request, Account Access, Refund, General Inquiry, Complaint, and Cancellation. The system prompt was written 18 months ago. It has grown to 1,600 tokens: a description of each category, 8 worked examples (one per category), and 9 "do not classify as X if..." clauses added after edge-case mistakes.

The chatbot handles 30,000 tickets per day. The prompt alone costs $7.20 per day — $2,628 per year — before a single word of customer input is counted. An audit reveals that 6 of the 9 defensive clauses were written for incidents that have not recurred, and 3 of the 8 worked examples cover categories that together represent less than 5% of actual tickets. Nobody knows which parts are safe to remove.

### Solution

**Layman version:** The team treated the prompt like a recipe that had too many ingredients. Instead of guessing which ingredients to cut, they cooked 400 dishes using the original recipe, kept the ones that turned out well, and then ran a cooking competition: 25 teams each tried to make the same dish using fewer ingredients. At the end, they picked the team that used the fewest ingredients while still making a dish that tasted just as good.

Technically: the team exported 400 real production tickets with their correct classifications as labelled examples. They reserved 80 tickets as a held-out test. They set a token budget of 500 tokens (one-third of the original). DSPy's MIPROv2 optimizer generated 25 candidate student prompts — each with different instruction phrasing and a different subset of worked examples. Each candidate was scored on the 80 held-out tickets using exact-match accuracy. The winning candidate was 520 tokens with 96.1% accuracy vs. the original's 96.8%.

### Outcome

- Daily prompt token cost dropped from $7.20 to $2.34 — a 67.5% reduction
- Latency per request dropped by ~110ms due to shorter prefill
- The distilled prompt is now version-controlled with its accuracy score embedded in the filename

### Benefits

- **Cost efficiency:** Token savings compound at scale — 30,000 calls/day means even small per-call savings add up to thousands of dollars annually
- **Maintainability:** The distilled prompt is smaller and easier for engineers to read, understand, and audit
- **Evidence-based quality:** Every deployed prompt now has a documented accuracy score on a real held-out dataset, replacing subjective engineering confidence

### Best Practices

- Always retain the original (teacher) prompt and its held-out score as a rollback baseline
- Run a 5% canary before full cutover — live accuracy may differ from held-out accuracy if the ticket distribution shifts seasonally
- Schedule re-distillation after any major product change that shifts the types of incoming tickets

---

## Scenario 2: Healthcare — Clinical Note Summarisation

### Problem Statement

A health-tech company builds software for outpatient clinics. One feature uses an LLM to summarise a clinician's free-text consultation notes into a structured 4-field summary: Chief Complaint, Findings, Assessment, and Plan. The prompt is 1,900 tokens: a medical terminology guide, detailed format instructions, 4 worked examples (one per field structure), and 7 special-case rules added after clinicians complained about summaries that were too long, too short, or wrongly formatted.

The feature runs across 200 clinics with a combined 15,000 summaries per day. The prompt costs $4.28 per day in token charges. More critically, each summary adds 380ms of latency (prefill-heavy). Clinicians using the feature on mobile devices during patient transitions find the delay disruptive.

### Solution

**Layman version:** Imagine a cooking school where the instructor has a 10-step recipe for making a perfect omelette. Some of those steps exist only because a student once cracked the shell into the bowl by mistake. The instructor collects 300 omelettes made by good students and asks: "What is the shortest set of instructions that reliably produces a good omelette from these students?" The answer turns out to be 4 clear steps — the other 6 were defensive rules for mistakes good students never make.

Technically: the team collected 300 real consultation notes with clinician-approved gold-standard summaries (used with appropriate de-identification). They held out 60 for evaluation. The metric function used an LLM-as-judge approach: GPT-4o scored each generated summary on a 0–2 rubric per field (0 = missing or wrong, 1 = partially correct, 2 = fully correct), for a maximum score of 8 per summary. MIPROv2 ran 20 candidate student prompts at a 600-token budget. The winning student prompt scored 7.4 / 8.0 average vs. the teacher's 7.6 / 8.0 — a 2.6% delta within the clinical team's stated tolerance of ≤ 5%.

### Outcome

- Prompt token count reduced from 1,900 to 580 — 69.5% reduction
- Latency per summary dropped from 380ms to 145ms — a 62% latency improvement on mobile
- Clinician satisfaction scores for the feature increased by 18 points (NPS) after the latency improvement

### Benefits

- **User experience:** Latency reduction directly translates to usability for time-constrained clinicians
- **Compliance readiness:** Smaller prompts with explicitly audited few-shot examples are easier to review for regulatory compliance
- **Metric rigour:** LLM-as-judge with a per-field rubric catches partial errors that exact-match would miss

### Best Practices

- For healthcare use cases, have clinicians review the few-shot examples selected by the optimizer before deployment — the optimizer cannot assess clinical accuracy
- Use de-identified examples in both training and the compiled state file; never deploy a prompt that contains real patient data as few-shot examples
- Re-evaluate the metric rubric whenever clinical documentation guidelines change

---

## Scenario 3: Finance — Transaction Narrative Extraction

### Problem Statement

A fintech company provides personal finance software. One pipeline reads raw bank transaction descriptions (e.g., "SQ*COFFEEHAUS 04/17 AUSTIN TX") and extracts three structured fields: Merchant Name, Category, and Location. The prompt was designed by a team that built it while onboarding a new bank data source — it includes 12 worked examples covering different bank formatting conventions, 5 defensive clauses for ambiguous merchant names, and a 300-token normalization guide for state abbreviations and common merchant name patterns.

The pipeline processes 2 million transactions per day across all users. The system prompt is 2,100 tokens. Even at $0.15/1M input tokens, that is $630/day — $229,950/year — in prompt tokens alone, before the transaction text itself.

### Solution

**Layman version:** Think of it as a postal sorting guide. The original guide is a thick binder covering every zip code format, every address abbreviation, every edge case for international mail. But 95% of parcels are domestic standard addresses. The sorting team realises they can write a one-page guide that handles 95% of parcels correctly, and flag the remaining 5% for manual review rather than trying to cover every case in the guide.

Technically: the team sampled 1,000 transactions from the production stream with verified gold extractions. They held out 200. They set a budget of 400 tokens (target: eliminate the normalization guide and reduce examples to 3). The metric was a weighted F1: merchant name counted 50%, category 30%, location 20%. MIPROv2 ran 40 candidates. The best student prompt was 390 tokens with a weighted F1 of 0.942 vs. the teacher's 0.951. Edge-case transactions (ambiguous merchants, foreign locations) were routed to a fallback pipeline with the full teacher prompt — reducing the blast radius of the accuracy delta.

### Outcome

- Student prompt handles 91% of transactions (routine cases) at 390 tokens
- Teacher prompt handles 9% (flagged as edge cases) at 2,100 tokens
- Blended daily cost drops from $630 to $78 — an 87.6% reduction
- Annual saving: $203,580

### Benefits

- **Tiered architecture:** Not all traffic needs the full teacher prompt — distillation enables a fast/cheap lane for routine inputs and a careful/expensive lane for edge cases
- **Cost transparency:** Every prompt version now has a documented cost-per-million-calls figure, enabling product managers to make informed trade-off decisions
- **Incremental rollout:** The student prompt can be A/B tested on a random 10% of transactions before full rollout, with automatic rollback if F1 drops below 0.92

### Best Practices

- Design a routing layer that sends ambiguous or low-confidence inputs to the teacher prompt rather than accepting a uniform accuracy drop
- Monitor per-merchant-category accuracy separately — some categories (e.g., international merchants) may have much higher error rates in the student prompt
- Review the distilled prompt's few-shot examples quarterly; merchant naming conventions and bank formats change over time

---

## Scenario 4: IT Helpdesk — Ticket Priority Assignment

### Problem Statement

An enterprise IT helpdesk uses an LLM to assign priority levels (P1 Critical, P2 High, P3 Medium, P4 Low) to incoming tickets before routing them to human agents. The prompt has grown over 2 years to 2,400 tokens: a 600-token SLA policy description, 8 worked examples covering various failure types, and 12 priority-override rules added after priority mis-assignments caused SLA breaches.

The helpdesk processes 8,000 tickets per day. The prompt costs $2.88/day. More critically, the prompt is impossible to maintain — no single engineer understands all 12 override rules or knows which ones are still necessary. When a new engineer modifies the prompt, they invariably break an edge case that another rule was protecting.

### Solution

**Layman version:** Imagine a referee's rulebook that started as 10 clear rules and has grown to 120 rules over 10 years, half of which were added to clarify edge cases that came up once in a championship game. The new referee coach says: "Let's watch 500 real games, record every correct call, and write the shortest rulebook that produces the same calls on those games." The result is a 15-rule rulebook that handles 97% of real game situations — and the 3% edge cases get escalated to a senior referee rather than being covered by an unwieldy rule.

Technically: the team extracted 600 historical tickets with agreed-upon correct priority assignments (validated by the helpdesk manager). They held out 120. The metric was weighted accuracy with a penalty for under-prioritisation (assigning P3 to a P1 is worse than the reverse). MIPROv2 ran 25 candidates at a 700-token budget. The winning student prompt was 680 tokens with a weighted accuracy of 94.2% vs. the teacher's 95.1%. Crucially, the under-prioritisation rate (the safety-critical metric) was 1.1% for the student vs. 0.9% for the teacher — within the team's stated tolerance of ≤ 2%.

### Outcome

- Prompt token count reduced from 2,400 to 680 — 71.7% reduction
- Daily cost drops from $2.88 to $0.82
- The distilled prompt is 680 tokens that two engineers can read and understand in 5 minutes, compared to the 2,400-token original that required 30 minutes and still left ambiguity
- New engineer onboarding time for the prompt system reduced from 3 hours to 20 minutes

### Benefits

- **Operational clarity:** A shorter, distilled prompt is also more auditable — engineers can reason about what it will do on a novel input without wading through 12 override clauses
- **Safety metric separation:** Defining under-prioritisation as a separate metric floor (not just aggregate accuracy) prevents the optimizer from trading away safety-critical accuracy for overall score
- **Change management:** Versioned prompt artefacts with documented accuracy scores give change-approval boards concrete evidence to evaluate before approving a prompt update

### Best Practices

- For safety-sensitive priority systems, add explicit metric floors for the highest-stakes output classes (P1 Critical must achieve ≥ 98% recall in the student prompt)
- Include a human review step in the canary rollout: have 3 experienced agents manually review 50 priority assignments made by the student prompt before full deployment
- Document the business justification for each few-shot example selected by the optimizer — this becomes the audit trail for compliance reviews

---

## Summary

| Dimension | Without Prompt Distillation | With Prompt Distillation |
|---|---|---|
| Prompt size | 1,600–2,400 tokens (accumulated ad-hoc) | 390–700 tokens (evidence-optimized) |
| Cost basis | Every call pays for every defensive clause | Only essential instructions paid per call |
| Accuracy evidence | "We tested it and it seemed fine" | Documented held-out score with delta vs. teacher |
| Maintainability | Engineers fear changing the prompt | Small, auditable prompt with version history |
| Edge case handling | Covered by ever-growing rule list | Routed to teacher prompt fallback |
| Deployment safety | Manual review before changes | Canary rollout with automatic rollback trigger |
| Re-evaluation cadence | Ad hoc, when something breaks | Scheduled (quarterly) + distribution-shift trigger |
| PII risk in few-shot | Unknown — examples may contain real data | Audited before compilation, synthetic alternatives used |
