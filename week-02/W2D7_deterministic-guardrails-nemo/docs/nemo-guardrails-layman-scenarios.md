# Deterministic Guardrails (NeMo) in Simple Words — Real-World QA Scenarios

No machine learning background needed — if you have ever seen a bouncer at a club door, you already understand the core idea.

---

## Core Idea

Imagine you run a customer service desk. You hire a very smart assistant who can answer almost any question. But you also need to make sure that assistant never gives out competitors' prices, never promises something the company cannot deliver, and always adds a legal disclaimer when discussing financial products.

You have two ways to enforce this. Option one: hire a second assistant to listen to every conversation and shout "stop!" if the first assistant says something wrong. Option two: give the first assistant a printed rulebook at the door — a list of things they must never say, topics they must always handle a specific way, and phrases they must always include. The rulebook works the same way every single day, regardless of how the first assistant is feeling.

That second option is deterministic guardrails. The rules execute the same way on every conversation, every time — no judgment, no variance.

| Concept | Analogy |
|---|---|
| Input rail | Bouncer checking IDs at the door — the question never gets inside if it fails the check |
| Output rail | Proofreader reviewing the letter before it is sealed and mailed |
| Colang policy | The printed rulebook the bouncer and proofreader both follow |
| Canonical flow | A scripted procedure the assistant must follow step-by-step for specific topics |
| Probabilistic guardrail | A second assistant who listens and might stop the first one — but has good and bad days |
| Deterministic guardrail | A metal detector that triggers for every piece of metal, every single time |

---

## Scenario 1 — Customer Support (Retail Bank)

### Problem Statement

A retail bank deploys an AI chat assistant to handle customer questions about accounts, loans, and savings products. Regulators require that any discussion of investment products must include a specific legal disclaimer. The bank uses a second AI model to check responses and add the disclaimer when needed — but sometimes the checker agrees with the main AI that a response is fine when it is not.

**Solution:** Replace the AI-based checker with deterministic output rails that inspect every response for investment vocabulary and automatically inject the required disclaimer text before the response reaches the customer.

**Layman version:** The bank replaced its "second opinion" AI reviewer with a simple automatic stamp machine. Every letter that mentions the word "investment", "returns", or "portfolio" gets a legal disclaimer stamp applied before it goes out — no judgment required, no exceptions. The stamp machine cannot be talked out of it.

**Outcome:**
- 0 compliance violations over a 6-month audit period (down from 312 in the prior period)
- False positive rate (legitimate queries incorrectly blocked) dropped from 4.2% to 0.3%
- Compliance team can audit the rulebook file directly without needing ML expertise

**Benefits:**
- **Regulatory certainty:** The bank can demonstrate to auditors exactly which rule caused each disclaimer to be added, with a named rule and a timestamp — no black-box model decisions
- **Cost savings:** Blocked queries never reach the LLM, saving token costs on 5% of daily traffic
- **Team separation:** The compliance team owns the rulebook file; the engineering team owns the application — each team works independently

**Best Practices:**
- Keep the investment vocabulary list in a separate configuration file that the compliance team can update without a code deployment
- Test every rule change against 500+ real customer messages before pushing to production
- Monitor the block rate per rule daily — a sudden spike usually means a new marketing campaign introduced a term that triggers an existing rule unintentionally

---

## Scenario 2 — Healthcare (Patient Intake Chatbot)

### Problem Statement

A hospital deploys an AI assistant to handle patient intake questions — appointment booking, general symptom triage, and FAQ responses. The assistant must never provide specific medication dosages, never diagnose conditions, and always recommend consulting a doctor for anything beyond general wellness information. Previous attempts to enforce this with an LLM-based moderator produced inconsistent results: on some days it blocked 8% of legitimate queries; on others it missed clear policy violations.

**Solution:** Implement NeMo Guardrails input rails that intercept messages containing clinical dosage keywords ("mg", "milligrams", "take X pills") and canonical dialogue flows that route symptom questions through a mandatory "I am not a doctor" acknowledgment before providing any information.

**Layman version:** The hospital put up a very specific set of road signs inside the chat. If the conversation ever reaches a "dosage" signpost, the car is automatically redirected to the "please call your doctor" exit — no matter what the driver typed. The signs do not think; they redirect. Every car, every time.

**Outcome:**
- Zero medication dosage responses provided to patients over a 3-month period (verified by audit log review)
- Patient satisfaction score maintained at 4.2/5 — the mandatory acknowledgment step was phrased naturally and did not feel like a barrier
- On-call clinical staff escalation rate for chatbot-originated queries dropped 40% because fewer patients received ambiguous partial answers

**Benefits:**
- **Patient safety:** The deterministic input rail for dosage keywords provides a hard guarantee — not a probabilistic estimate — that dosage information will not be delivered through the chat channel
- **Auditability:** Hospital risk management can pull a report of every triggered rail with timestamps, for inclusion in regulatory filings
- **Model-independence:** When the hospital upgraded from one LLM provider to another, the guardrail policy required zero changes — it works the same regardless of which model is behind it

**Best Practices:**
- Include both exact terms ("milligrams") and common abbreviations ("mg", "mcg", "IU") in the dosage keyword list — patients use informal language
- Define a clear, warm fallback message for blocked dosage queries: "For medication questions, please call our pharmacy line at [number]" — not a generic error
- Review the escalation flow quarterly with the clinical team to ensure the routing destinations (phone numbers, portal links) are still correct

---

## Scenario 3 — Finance (Investment Advisory Platform)

### Problem Statement

A fintech platform offers an AI-powered investment research assistant to retail investors. Regulations prohibit the platform from providing personalised investment recommendations without a licensed advisor. The platform initially used prompt instructions to prevent the AI from making recommendations, but discovered that specific phrasings — "what would you do if you were me?" — reliably caused the model to cross the line into personalised advice.

**Solution:** Implement a canonical Colang dialogue flow for investment-related queries. Any query matching the `investment_recommendation` intent triggers a two-step flow: the bot first presents a disclosure and asks the user to confirm they understand this is not personalised advice; only after confirmation does the LLM generate a response, which is then checked by an output rail for first-person recommendation language ("you should buy", "I recommend").

**Layman version:** The platform built a two-door entry system. To get investment information, users must walk through the first door (the disclosure room) and press a "I understand" button before the second door opens. The second door only opens after the button is pressed — no amount of clever phrasing lets anyone skip the first room. And even after the second door opens, a guard checks outgoing messages to make sure they do not contain the phrase "you should buy."

**Outcome:**
- Zero regulatory violations for personalised advice delivery over a 12-month audit period
- User drop-off at the disclosure step was only 6% — most users who genuinely wanted research information completed the acknowledgment
- Legal team reduced its manual response review sampling rate from 10% to 2% of responses, saving approximately 15 hours per week

**Benefits:**
- **Regulatory compliance:** The two-step canonical flow provides a documented, reproducible user consent mechanism — required under MiFID II for automated investment tools in the EU
- **Reduced legal overhead:** The output rail for first-person recommendation language catches edge cases that the canonical flow alone might not — defence in depth
- **Investor trust:** Users who see the disclosure step perceive the platform as more trustworthy than platforms that simply answer investment questions without any acknowledgment

**Best Practices:**
- Write the disclosure prompt in plain language — regulatory legalese at the acknowledgment step causes drop-off; confirm with the legal team that plain language still meets the regulatory requirement
- The output rail for recommendation language should check for first-person constructions ("you should", "I suggest", "my recommendation") as multi-word phrases, not individual words — "risk" alone should not trigger a block
- Log every canonical flow completion (user pressed "I understand") as a compliance event with timestamp, user ID, and session ID for audit trail purposes

---

## Scenario 4 — IT Helpdesk (Enterprise Internal Tool)

### Problem Statement

An enterprise deploys an internal AI assistant for IT helpdesk queries — password resets, software access requests, and troubleshooting guides. The assistant has access to a knowledge base of internal policies. Security discovers that employees are successfully extracting internal salary data and HR policy details by asking the assistant to "summarise all documents about compensation." The knowledge base includes both technical and HR documents, and the retrieval system does not distinguish between them.

**Solution:** Implement input rails that detect queries matching HR and compensation vocabulary and block them before retrieval occurs, returning a message that directs the employee to the HR portal. A separate output rail scans responses for patterns matching salary formats (currency symbols followed by numbers) and blocks any response containing them, regardless of how the query was phrased.

**Layman version:** The company added two filters to the helpdesk's filing cabinet. Filter one: a lock on the cabinet drawer labelled "HR" — asking about compensation means that drawer stays locked and you get a "wrong desk, please go to HR" card. Filter two: a photocopier that refuses to print any page containing a number that looks like a salary — even if you somehow got the drawer open, the printout never comes out.

**Outcome:**
- Zero successful compensation data extractions over a 4-month monitoring period (verified by security team red team exercises)
- 98% of IT helpdesk queries (password resets, software access, hardware troubleshooting) were unaffected by the new rails
- Time to detect and respond to a policy violation dropped from "next security audit cycle" to "real-time alert within 30 seconds" — the rail fires synchronously and writes to the security event log

**Benefits:**
- **Data security:** The dual-layer approach (input + output rail) means an attacker must bypass both the query filter and the response filter — two independent security controls rather than one
- **Zero ML dependency:** Both rails use deterministic pattern matching — there is no model to retrain, fine-tune, or prompt-engineer around
- **Operational simplicity:** The security team manages a vocabulary list and a regex pattern; no ML expertise required for ongoing maintenance

**Best Practices:**
- Scope the HR vocabulary list carefully — words like "benefits" and "bonus" appear in both IT and HR contexts; use multi-word phrases ("compensation package", "base salary", "pay grade") rather than single tokens
- Run the salary format regex against a sample of real IT helpdesk responses before deploying — confirm the false positive rate is below 0.5% before going live
- Include a clear and helpful redirect message: "For compensation questions, visit hr.company.com or contact your HR Business Partner" — employees who hit the block should know exactly where to go next

---

## Summary

| Without Deterministic Guardrails | With Deterministic Guardrails |
|---|---|
| Safety decisions vary based on model state and context | Safety decisions are identical for identical inputs, every time |
| A jailbreak that fools the LLM may also fool the LLM-based evaluator | A jailbreak cannot change what a pattern match returns |
| Policy changes require model retraining or few-shot prompt updates | Policy changes are edits to a text configuration file |
| Compliance auditors must understand model behaviour to verify safety | Compliance auditors can read the Colang rulebook directly |
| False positive rates fluctuate with model updates and prompt drift | False positive rates are stable until a rule is explicitly changed |
| Adding a new prohibited topic requires a model evaluation cycle | Adding a new prohibited topic requires adding one line to the config file |
| Safety layer adds 200–500ms per request (LLM evaluator call) | Safety layer adds 2–8ms per request (pattern matching, no LLM call) |
