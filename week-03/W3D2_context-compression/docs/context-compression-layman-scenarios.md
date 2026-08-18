# Context Compression in Simple Words — Real-World QA Scenarios

No ML background needed: if you have ever handed someone a 50-page report and watched them read only the first and last page, you already understand the problem this solves.

---

## Core Idea

Imagine you hired an incredibly smart assistant who can read anything — but you pay them by the word. Every morning you hand them a stack of documents, emails, and meeting notes and ask a single question. They read every word, bill you for every word, and answer your question. Now imagine that 60% of what you handed them was irrelevant to the question you asked. You paid for it anyway, and — here is the surprising part — all that irrelevant material in the middle actually made them *worse* at finding the answer.

Context Compression is the practice of having a cheaper, faster assistant pre-read the stack first, remove everything that does not relate to today's question, and hand the smart (expensive) assistant a trimmed, focused set of materials. The smart assistant gets better information, finishes faster, and you pay a fraction of the original cost.

There are three ways to do this trimming. **Extractive compression** physically removes low-relevance sentences — like cutting irrelevant paragraphs with scissors. **Abstractive compression** rewrites and condenses the content into a shorter summary — like having someone write a briefing note. **Token-level compression** removes individual redundant words throughout the text — like editing out every filler word and repeated phrase.

| Concept | Real-World Analogy |
|---|---|
| Context window | The inbox tray the assistant reads before answering |
| Token budget | The maximum number of pages you will pay for |
| Extractive compression | Cutting irrelevant paragraphs with scissors |
| Abstractive compression | A junior analyst writing a briefing note |
| Token-level compression | A copy editor removing filler words throughout |
| Query-aware compression | Highlighting only the parts relevant to today's specific question |
| Compression ratio | Percentage of original pages you kept |
| Lost-in-the-middle | The assistant ignoring the middle pages of a very long stack |

---

## Scenario 1 — Customer Support (E-Commerce)

### Problem Statement

A large online retailer's AI support agent handles 15,000 tickets per day. Each conversation can span 20+ messages — order history, shipping updates, past complaints, promotional codes applied, and lengthy product descriptions. The agent sends all of this to the LLM every time a customer asks even a simple follow-up question like "Where is my package?"

### Solution

The team inserts a context compression step before every LLM call. The compressor reads the conversation and the customer's current question, then:

- Removes all turns unrelated to the current question (past complaints about unrelated orders, promotional chatter).
- Condenses the shipping update history into a single paragraph: "Order #4821 shipped on Jan 3, carrier USPS, last scan Jan 5 at Chicago facility, estimated delivery Jan 7."
- Strips verbose product descriptions down to the product name and SKU.

**Layman version:** The compressor acts like a customer service supervisor who, before passing a call to the expert, gives them a 30-second briefing: "Customer bought a blender, it shipped, it's stuck in Chicago, they want to know when it arrives." The expert never has to read the full transcript.

### Outcome

- Average context length dropped from 8,400 tokens to 2,100 tokens (75% reduction).
- LLM response accuracy for tracking-related queries improved from 71% to 89% (fewer middle-context misses).
- Daily input token cost reduced from $504 to $126, saving $137,460 per year.

### Benefits

- **Cost savings:** Three-quarters of input token spend eliminated with no change to the LLM model.
- **Speed:** Shorter context means faster LLM inference; average response time dropped from 3.2s to 1.8s.
- **Accuracy:** Relevant shipping data now appears near the start of the context where attention is strongest.

### Best Practices

- Always condition the compressor on the current question — "Where is my package?" should retain shipping data but can safely drop payment history.
- Protect order numbers, tracking codes, and dates as named entities that must survive compression.
- Cache compressed product descriptions; the same product description compresses identically for any shipping question.

---

## Scenario 2 — Healthcare (Clinical Decision Support)

### Problem Statement

A hospital's clinical decision support tool retrieves relevant sections from patient records, clinical guidelines, and drug interaction databases to help physicians during consultations. A typical query context includes a 6,000-word patient history, three 2,000-word clinical guidelines, and a 1,500-word drug database excerpt — over 9,500 tokens before the physician's actual question.

### Solution

The system applies a two-stage compression pipeline:

1. **Document trimmer (extractive):** Each clinical guideline is scored sentence-by-sentence against the physician's query (e.g., "Is metformin safe for this patient given the renal function values?"). Only sentences mentioning renal function, metformin dosing, or contraindications are retained.
2. **History summariser (abstractive):** The 6,000-word patient history is condensed into a 400-token structured summary preserving lab values, diagnoses, current medications, and allergies — the entities most relevant to any drug decision.

**Layman version:** Instead of handing the physician a filing cabinet and saying "it's in there somewhere," a medical scribe pulls only the relevant pages, highlights the key numbers, and clips them to the front of the chart. The physician makes the same decision with 10% of the reading.

### Outcome

- Context reduced from 9,500 tokens to 1,800 tokens (81% reduction).
- Physician-rated answer relevance score increased from 3.2/5 to 4.4/5 in blind evaluation.
- Time-to-response for the AI tool reduced from 4.8 seconds to 1.9 seconds.

### Benefits

- **Clinical safety:** Compressing against the specific query (drug safety) ensures contraindication data is always near the top of the context.
- **Compliance:** Abstractive summarisation can strip patient identifiers before sending to a cloud LLM, reducing PHI exposure.
- **Efficiency:** Physicians receive faster answers during time-critical consultations.

### Best Practices

- Never compress the medication list or allergy section — these are safety-critical and short enough to pass verbatim.
- Use the abstractive compression model under the same BAA (Business Associate Agreement) as the main clinical LLM.
- Run a weekly evaluation where clinicians score compressed vs. uncompressed answers to detect quality regressions.

---

## Scenario 3 — Finance (Investment Research Assistant)

### Problem Statement

An investment bank's research assistant aggregates earnings call transcripts, SEC filings, and analyst notes to answer portfolio manager questions. A single query may pull in 40 pages of transcript text. Portfolio managers ask targeted questions like "What did the CFO say about Q3 margin guidance?" but pay for 40 pages of context every time.

### Solution

Token-level compression (LLMLingua-2) is applied to the transcript corpus before it reaches the main LLM. The proxy model scores each token in the transcript for redundancy. Conversational filler, repeated boilerplate ("As I mentioned earlier", "Thank you for that question"), and off-topic analyst questions are scored as low-importance and removed. The remaining tokens — financial figures, forward guidance statements, and management commentary — are passed to the main LLM.

**Layman version:** A research analyst is given the full 40-page earnings transcript but asked to circle only the sentences that contain numbers, forecasts, or direct executive statements about margins. Everything else gets crossed out before the portfolio manager sees it.

### Outcome

- 40-page transcript (approximately 12,000 tokens) compressed to 2,400 tokens (80% reduction).
- Answer accuracy on financial fact-retrieval benchmark: 91% at 80% compression vs. 93% with full context (2% quality delta for 80% cost saving).
- Annual savings for a team running 500 research queries per day: approximately $219,000.

### Benefits

- **Cost efficiency:** Token-level compression requires no additional LLM call — only a small local model, making it cheaper per query than abstractive compression.
- **Audit trail:** Removed tokens are logged, so compliance teams can verify what the LLM did and did not see.
- **Speed:** The proxy model runs in under 100ms on a GPU, adding negligible latency.

### Best Practices

- Validate the compression model on a financial domain corpus — general-purpose models may incorrectly score financial jargon as low-importance.
- Retain all sentences containing numerical values unconditionally; financial figures are always high-importance.
- Set compression ratio conservatively (retain 25–30% rather than 15%) for regulatory filings where any omission carries legal risk.

---

## Scenario 4 — IT Helpdesk (Internal Enterprise Assistant)

### Problem Statement

An enterprise IT helpdesk bot answers employee questions about internal tools, policies, and procedures. Each query retrieves relevant sections from a 500-page IT policy handbook, software installation guides, and the employee's IT ticket history. The retrieved context averages 7,000 tokens, but most policy documents contain extensive boilerplate (legal disclaimers, version history, table of contents) that is irrelevant to any specific question.

### Solution

A two-pass hybrid compressor is applied:

1. **Deterministic stripping (pre-processing):** Known boilerplate patterns (version tables, legal footers, navigation headers) are removed with regex rules before compression even begins. This eliminates 30% of tokens with zero ML cost.
2. **Extractive compression (TF-IDF):** The remaining document text is scored against the employee's question. Low-scoring sentences are dropped until the context fits within an 800-token budget.

**Layman version:** Before the IT expert answers your question, their assistant removes the table of contents, the legal disclaimer at the bottom, and every section that starts with "This policy applies to..." because you just want to know how to reset your VPN password.

### Outcome

- Average context reduced from 7,000 tokens to 1,100 tokens (84% reduction).
- First-contact resolution rate (employee got the right answer without follow-up) improved from 58% to 74%.
- Total cost per helpdesk query reduced by 82% (mostly driven by token reduction on the main LLM).

### Benefits

- **Zero ML cost for the first pass:** Regex-based boilerplate stripping is free — no model calls, no latency.
- **Deterministic and auditable:** IT governance teams can inspect exactly which document sections were removed.
- **Self-improving:** As teams identify recurring boilerplate patterns, they add them to the stripping rules, continuously improving the baseline compression.

### Best Practices

- Maintain a curated list of boilerplate patterns specific to your internal documents — generic patterns will not catch organisation-specific header/footer formats.
- Run extractive compression after boilerplate stripping, not before — compressing boilerplate wastes scoring compute.
- Set a maximum compression ratio (e.g., never reduce a document below 15% of original) to avoid accidentally discarding all content for edge-case queries.

---

## Summary

| Dimension | Without Context Compression | With Context Compression |
|---|---|---|
| Token cost per query | Billed for 100% of retrieved context, including irrelevant content | Billed for 20–60% of retrieved context — only relevant segments |
| Answer accuracy | Degrades as context grows due to lost-in-the-middle effect | Improves because relevant content is placed at high-attention positions |
| Response latency | Increases linearly with context length | Reduced by 40–60% due to shorter input |
| Compression overhead | None | 5–500ms depending on strategy (TF-IDF vs. abstractive) |
| Memory in multi-turn agents | Context window fills up after 10–20 turns | Rolling summarisation keeps context bounded indefinitely |
| Debugging ability | Full context visible; root cause of errors is traceable | Requires logging of original context to preserve debuggability |
| Implementation complexity | None — send everything | Requires segment routing, budget allocation, and quality monitoring |
