# Distributed Tracing (LangSmith) in Simple Words — Real-World QA Scenarios

No background in observability required. If you have ever wondered "why did my AI give that answer?" — this is the tool that answers that question.

---

## Core Idea

When your LLM application answers a question, it does not just call one function. It retrieves documents, ranks them, assembles a prompt, calls a model, and maybe calls a tool or two along the way. Each of those steps can fail silently. Distributed tracing is the practice of recording every one of those steps — its inputs, outputs, timing, and relationship to the steps around it — so that when something goes wrong, you can see exactly where.

Think of it like a flight data recorder for your AI pipeline. A plane's black box does not just record "the plane crashed." It records engine thrust, altitude, control surface positions, and cockpit audio — every variable across every system, timestamped and ordered. Distributed tracing does the same for your LLM pipeline. When the final answer is wrong, you open the trace and read the flight log backwards to find where the first deviation occurred.

LangSmith is the platform that stores, displays, and lets you query these flight logs for LLM workloads. It adds LLM-specific features — token counts, prompt/completion text, eval score binding, and training data export — that general-purpose tracing tools like Jaeger or Zipkin do not provide out of the box.

| Concept | Analogy |
|---|---|
| **Trace** | The complete flight log for one user request |
| **Span** | One entry in the log (e.g., "landing gear deployed at 14:32:05") |
| **Root span** | The cockpit — the top-level function that received the user's request |
| **Child span** | A subsystem log — retriever, reranker, LLM call, tool invocation |
| **Feedback score** | The post-flight safety rating assigned by the inspector |
| **Dataset export** | The incident report archive used to train future pilots |

---

## Scenario 1 — Customer Support (E-Commerce)

### Problem Statement

A large online retailer deployed an AI assistant to handle returns and refund questions. The assistant handles 12,000 conversations per day. Roughly 2% of responses are wrong — they cite the wrong return window or the wrong refund method. That is 240 bad responses per day. Each bad response triggers a human escalation costing approximately $8 in agent time. The team cannot identify why these errors occur because they only log the final response text.

**Solution**

The team instruments their pipeline with LangSmith tracing. Each conversation now produces a run tree with four spans: `classify_intent`, `retrieve_policy_documents`, `assemble_context`, and `generate_response`.

**Layman version:** Imagine the AI's thought process is a relay race with four runners. Before tracing, you only watched the last runner cross the finish line. With tracing, you can watch all four runners and see exactly which one dropped the baton. In this case, the trace revealed that the `retrieve_policy_documents` span was consistently returning a policy document from 18 months ago — before a return window change — ranked higher than the current policy. The retrieval step dropped the baton, not the language model.

**Outcome**
- Root cause identified in 45 minutes after enabling tracing (previously unresolved for 3 weeks)
- Return policy document index refreshed; error rate dropped from 2% to 0.1%
- 216 fewer escalations per day, saving ~$1,700/day in agent time

**Benefits**
- **Causal visibility:** The tree structure showed that the generation step was correct given its input — the retrieval step was the actual failure point
- **Speed of diagnosis:** 45 minutes vs. 3 weeks without tracing
- **Targeted fix:** The fix was a data freshness issue, not a model issue — tracing prevented a costly and unnecessary model swap

**Best Practices**
- Refresh your document index on a schedule tied to the update frequency of your source data
- Set up an alert that fires when the average retrieval score across spans drops below a threshold
- Tag spans with the document version date so staleness is visible directly in the trace

---

## Scenario 2 — Healthcare (Clinical Decision Support)

### Problem Statement

A hospital network uses an AI assistant to help nurses look up medication interaction information during patient intake. Accuracy is non-negotiable. The team wants to verify that every response cites a known interaction database entry and does not introduce information not present in the retrieved context.

**Solution**

The pipeline is instrumented with LangSmith. After each generation span completes, an automated LLM-as-a-Judge evaluator checks whether every claim in the response is grounded in the retrieved documents. It attaches a `grounding_score` (0.0–1.0) to the root span. Responses with a score below 0.85 are flagged for human pharmacist review before display.

**Layman version:** It is like having a quality control inspector standing at the end of an assembly line. The inspector does not build the product — they check it against the blueprint (the retrieved documents) before it ships. The "blueprint" in this case is the set of database entries the AI retrieved. If the response contains anything not in the blueprint, the inspector catches it and sends it for human review. LangSmith is the factory floor where all of this is recorded — what the blueprint said, what the AI built, and what the inspector scored.

**Outcome**
- 98.7% of responses scored above 0.85 grounding threshold and were displayed without delay
- 1.3% of responses flagged for pharmacist review, catching 12 genuine hallucinations per day
- Zero patient safety incidents attributable to AI responses in the 6 months post-deployment

**Benefits**
- **Automated safety gate:** The grounding evaluator acts as a last line of defence before a response reaches a clinical user
- **Audit trail:** Every interaction is permanently recorded with its grounding score — satisfies clinical audit requirements
- **Continuous improvement:** Flagged traces are exported weekly as a dataset for fine-tuning the generation step

**Best Practices**
- In high-stakes domains, run the grounding evaluator synchronously (blocking) rather than asynchronously so no response is displayed before scoring
- Store the human pharmacist's correction as feedback on the original span — this creates labelled training data automatically
- Set the grounding threshold conservatively at first (0.90) and lower it only after reviewing 1,000 flagged traces

---

## Scenario 3 — Finance (Regulatory Compliance Q&A)

### Problem Statement

A financial services firm built an internal assistant that answers questions about regulatory compliance from a corpus of 14,000 regulatory documents. Analysts use it daily. The firm's compliance team noticed that some answers blend information from two different regulatory regimes (e.g., EU MiFID II and US SEC rules) without clearly distinguishing them. This is a compliance risk.

**Solution**

LangSmith tracing is added with a custom `regulation_source` metadata tag on each retrieval span, recording which regulatory regime each retrieved document belongs to. A deterministic guardrail checks whether the retrieved documents span more than one regime and, if so, adds a `mixed_source_warning` tag to the generation span.

**Layman version:** Think of the retrieval step as a research assistant pulling files from a giant cabinet. Without tracing, you hand the analyst a report without knowing which folders the assistant pulled from. With tracing, every retrieved document is labelled with the drawer it came from (EU rules, US rules, UK rules). If the assistant pulled from multiple drawers, a flag appears on the report saying "This answer combines rules from different jurisdictions — verify before acting." That flag is the `mixed_source_warning` span tag, visible in LangSmith.

**Outcome**
- Mixed-source warnings triggered on 8% of queries — previously invisible
- Compliance team reviewed all 8% and found 23% of those answers contained genuine cross-regime ambiguity requiring a disclaimer
- Zero compliance incidents in the 4 months post-instrumentation

**Benefits**
- **Domain-specific metadata:** Custom span tags make regulatory regime visible in the trace without changing the UI
- **Deterministic guardrail at the span level:** The check fires on the retrieval span before generation, not after — earlier is cheaper
- **Quantified risk:** The 8% mixed-source rate is now a tracked metric, not an unknown

**Best Practices**
- Add domain-specific metadata to spans early — retrofitting metadata tags after incidents is harder than adding them at instrumentation time
- Use deterministic rules (not LLM judges) for guardrails where the check is well-defined (e.g., "did we retrieve from two regulatory regimes?")
- Export mixed-source traces to a separate dataset for analyst review — do not mix them with clean traces in your training data

---

## Scenario 4 — IT Helpdesk (Enterprise Internal Tools)

### Problem Statement

A large enterprise deployed an internal IT helpdesk bot that answers questions about VPN setup, software installation, and access request procedures. The bot has a 4-step pipeline: intent classification, knowledge base retrieval, procedure generation, and a validation step that checks whether the generated steps reference real software version numbers. The team is spending significant time on post-deployment debugging because they have no visibility into which step is producing incorrect procedure steps.

**Solution**

All four pipeline steps are decorated with `@traceable`. LangSmith traces reveal that the validation step — which was supposed to reject hallucinated version numbers — had a bug: it only checked the first numbered step in the procedure, not all steps. Hallucinated version numbers in steps 3–8 of long procedures were passing the validator undetected.

**Layman version:** Imagine a checklist inspector who only reads the first item on the checklist and stamps "passed" for the whole thing. That is exactly what was happening in the validation span. The trace showed the validator's input (the full procedure) and its output (a "valid" decision), and also showed that only one item was ever evaluated. Without the trace, this bug looked like a language model accuracy problem. With the trace, it was clearly a logic bug in the validator — a one-line code fix.

**Outcome**
- Bug identified in 20 minutes of trace inspection; fix took 10 minutes to implement
- Procedure accuracy (verified against an IT expert's ground truth) improved from 71% to 94%
- Time spent on IT helpdesk escalations dropped by 38% in the following month

**Benefits**
- **Bug vs. model confusion eliminated:** The trace proved the LLM was generating correct outputs — the validator was the broken component
- **Fast iteration:** Short feedback loop (trace → identify → fix → redeploy) meant the improvement shipped the same day
- **Baseline established:** With tracing in place, the 94% accuracy figure is now a monitored metric with alerts

**Best Practices**
- Instrument validation and guardrail steps as separate spans — they are often where silent bugs hide
- Capture both the validator's input and its decision in the span outputs — "passed" alone is not enough information
- Add `@pytest.mark.parametrize` tests for validator edge cases (long inputs, edge version formats) to prevent regressions

---

## Summary

| Aspect | Without Distributed Tracing | With Distributed Tracing |
|---|---|---|
| Debugging a wrong answer | 2–6 hours of log archaeology per incident | 5–15 minutes of trace tree navigation |
| Identifying the failing pipeline step | Guesswork; requires re-running the pipeline with print statements | Pinpointed to the exact span by comparing inputs and outputs |
| Measuring quality per step | Not possible; only the final response is observable | Every span has a measurable output quality indicator |
| Building training data from failures | Manual effort; requires reproducing failures | One-click export of low-scored spans to a labelled dataset |
| Detecting silent regressions | Only visible when users complain | Span error rate and eval score trends are alertable metrics |
| Security and compliance auditing | Final response log only; no reconstruction of context | Full prompt, retrieved documents, and scores permanently stored per request |
| Cost attribution | Total API cost visible; per-step cost invisible | Token counts per span enable cost attribution to retrieval, reranking, or generation |
