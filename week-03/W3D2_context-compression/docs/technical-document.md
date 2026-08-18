# W3D2 — Context Compression
## AI Engineering Production Playbook | Week 3, Day 2
**Vertical:** Context Engineering & Tokens

---

## 1. Overview

Context Compression is a family of techniques that reduce the token length of runtime context — conversation history, retrieved documents, tool outputs — before it is handed to a large language model, while preserving the information most relevant to answering the current query. As context windows grow from 8k to 128k tokens, the cost and latency of processing them scale linearly, but model quality does not: research on positional attention bias shows that models systematically under-attend to tokens placed in the middle of long inputs. Context Compression directly addresses this mismatch by making the input shorter, denser, and more query-relevant. It is production-relevant today because frontier model pricing is billed per token, multi-turn agents accumulate context rapidly, and retrieval pipelines regularly surface documents containing only partial relevance.

---

## 2. Learning Objectives

By the end of this document, you will be able to:

1. **Explain** why longer context does not linearly improve LLM answer quality, citing positional attention decay.
2. **Distinguish** between extractive compression (trimming), abstractive compression (summarisation), and token-level compression (LLMLingua).
3. **Implement** a query-aware extractive compressor using TF-IDF sentence scoring in Python.
4. **Evaluate** the quality-cost trade-off of each compression strategy using compression ratio and factual recall metrics.
5. **Design** a production compression pipeline that applies the correct strategy based on context type (conversation, document, tool output).
6. **Apply** LangChain's ContextualCompressionRetriever to an existing RAG pipeline.
7. **Benchmark** the latency overhead of a compression pre-processing step against the token savings it produces.
8. **Build** fallback logic that bypasses compression when the compressor itself would consume more tokens than it saves.

---

## 3. Problem Statement

Every multi-turn LLM application accumulates context. A customer support agent handling a complex billing dispute may process 20+ turns of conversation, 3–5 retrieved knowledge-base documents, and verbose JSON tool outputs — totalling 10,000–15,000 tokens before the system prompt is even counted. Two distinct failure modes emerge.

**Cost explosion:** At GPT-4o pricing (input tokens billed per 1M tokens), a pipeline processing 10,000 requests/day with an average context of 12,000 tokens spends roughly $72/day on input tokens alone — before any output. A 50% context reduction translates directly to a $36/day saving at that volume, or $13,000/year.

**Quality degradation:** The "lost-in-the-middle" phenomenon (Liu et al., 2023, arXiv:2307.03172) demonstrates that transformer attention is biased toward the beginning and end of the context window. Documents placed in positions 3–8 of a 10-document context receive substantially less attention weight than those at positions 1 or 10, regardless of their relevance. Simply giving the model more context actively hurts retrieval-augmented tasks when relevant information lands in the middle.

The naive approach — sending everything and relying on the model to filter — fails on both dimensions simultaneously: it maximises cost while degrading quality for middle-positioned content. A context compression layer inserted between retrieval and generation resolves both failure modes without requiring model fine-tuning or architectural changes.

---

## 4. Real-World Scenarios

### Scenario A — The Problem: Uncompressed Agent Context

A financial services chatbot processes customer portfolio queries. Each conversation includes: a 15-turn chat history (4,200 tokens), 5 retrieved product documents (3,800 tokens), and 2 tool call results in JSON format (1,600 tokens) — 9,600 tokens total per query. The system prompt adds another 800 tokens, bringing the total to 10,400 tokens per LLM call.

At 500 queries/day, the input token cost alone is $26/day ($9,490/year). More critically, a post-launch audit reveals that 22% of incorrect answers cite information from the retrieved documents — but the documents were retrieved correctly. The actual problem is positional: the relevant paragraph in each document was placed in positions 4–6 of the context, where attention weight is lowest. The model consistently ignores it.

### Scenario B — The Solution: Query-Aware Compression Pipeline

The same chatbot is retrofitted with a three-stage compression pipeline inserted between retrieval and generation:

1. **Conversation summariser** — turns the 15-turn history into a 280-token structured summary preserving entities, decisions, and open questions. Reduction: 4,200 → 280 tokens (93%).
2. **Extractive document trimmer** — scores each sentence in retrieved documents against the current query using embedding cosine similarity, retains top-30% sentences. Reduction: 3,800 → 1,140 tokens (70%).
3. **Tool output formatter** — a deterministic JSON→prose converter strips null fields and array padding. Reduction: 1,600 → 400 tokens (75%).

Post-pipeline context: 280 + 1,140 + 400 + 800 (system) = 2,620 tokens. Token reduction: 75%. Incorrect answer rate attributable to missed context: drops from 22% to 6%. Daily input token cost: drops from $26 to $6.50.

---

## 5. Solution Architecture

A context compression pipeline sits as a middleware layer between the retrieval/memory layer and the LLM call. It receives raw context segments (conversation history, documents, tool outputs), applies a type-specific compressor to each segment, and assembles a compressed context that is then passed to the LLM unchanged.

The architecture has three logical components:

**Segment router** classifies each context segment by type (conversation, document, structured data) and dispatches it to the appropriate compressor. This routing is rule-based, not ML-based, keeping latency minimal.

**Compressor registry** holds pluggable compressor implementations. Each compressor exposes a standard interface: `compress(segment: str, query: str, budget: int) -> str`. The budget parameter enforces a token ceiling per segment, enabling the pipeline to target a total context budget.

**Budget allocator** distributes the total token budget across segments proportionally to their estimated importance (scored by query relevance). Segments with low relevance scores receive smaller budgets, triggering more aggressive compression.

The compressed segments are concatenated in a fixed order — system prompt, compressed history, compressed documents, compressed tool outputs — placing the most query-relevant content near the beginning and end of the context to exploit positional attention bias.

---

## 6. Internal Working Mechanics

### 6.1 Extractive Compression (Sentence Trimming)

Extractive compression treats the input as a set of independent sentences and scores each for relevance to the query. Only the top-k sentences are retained; all others are removed.

**Scoring function options:**
- **TF-IDF cosine similarity** — fast, no model required, works well for keyword-heavy technical documents. Fails on paraphrase and synonym matching.
- **Embedding cosine similarity** — more semantically accurate (e.g., using `text-embedding-3-small`). Adds one embedding API call per compression operation.
- **Cross-encoder reranking** — highest quality but slowest; typically reserved for document-level compression, not sentence-level.

**Algorithm steps:**
1. Split input into sentences using a rule-based splitter (sentence boundaries at `.`, `!`, `?` followed by whitespace).
2. Encode each sentence and the query into a vector space.
3. Compute cosine similarity between each sentence vector and the query vector.
4. Sort sentences by similarity score descending.
5. Greedily select sentences until the token budget is reached.
6. Reassemble selected sentences in their original order (preserving narrative flow).

Step 6 is critical: returning sentences in similarity-rank order rather than original order produces incoherent text that confuses the LLM.

### 6.2 Abstractive Compression (Summarisation)

Abstractive compression calls a smaller, cheaper LLM to produce a condensed prose summary of the input. It can cross-sentence boundaries, fuse information across paragraphs, and produce output significantly shorter than any extractive approach.

**Prompt pattern:**
```
System: You are a precise summariser. Preserve all named entities, 
        numerical values, dates, and decisions. Omit pleasantries, 
        repetitions, and off-topic content.
User:   Summarise the following conversation to answer this query: 
        [QUERY]
        
        Conversation:
        [CONVERSATION]
        
        Respond in at most [BUDGET] tokens.
```

The query conditioning is essential — without it, the summariser optimises for general coverage and may discard query-specific details.

**Cost consideration:** Abstractive compression requires one additional LLM call. For a 4,000-token conversation compressed to 300 tokens using `gpt-4o-mini`, the compression call costs approximately $0.00060. The savings on the main call (3,700 tokens × main model price) break even at approximately 1.5× the compression model's per-token price. At current pricing, using `gpt-4o-mini` to compress for `gpt-4o` always saves money.

### 6.3 Token-Level Compression (LLMLingua)

LLMLingua (Jiang et al., 2023, arXiv:2310.05736) scores individual tokens — not sentences — using a small language model (GPT-2 scale) as a proxy. Tokens with low conditional probability in the proxy model's distribution are candidates for removal. The key insight is that a token that the proxy model can predict easily from context is likely redundant; a token it cannot predict is information-dense.

**Algorithm steps:**
1. Run the prompt through the proxy LM to obtain per-token conditional probabilities.
2. Set a target compression ratio (e.g., 0.3 = retain 30% of tokens).
3. Rank tokens by their negative log-likelihood (lower probability = higher importance).
4. Remove the lowest-importance tokens until the target ratio is reached.
5. The remaining tokens form a compressed, grammatically approximate prompt.

The resulting text is not natural language but is interpretable by large LLMs because they have seen similar compressed text patterns during training. LLMLingua-2 improves on the original by training a dedicated compression model rather than using an off-the-shelf LM, achieving better quality at equivalent compression ratios.

### 6.4 Edge Cases and Failure Modes

- **Over-compression:** If the budget is set too low, even critical entities (account numbers, error codes) may be dropped. Mitigation: extract and protect named entities before compression; re-inject them into the compressed output.
- **Compression of already-short segments:** A segment of 200 tokens should not trigger compression overhead. Mitigation: set a minimum segment size threshold (e.g., 500 tokens) below which compression is bypassed.
- **Cross-segment dependencies:** A term introduced in one segment may be referenced in another. Extractive compression of individual segments can sever these references. Mitigation: run a cross-segment entity consistency check post-compression.

---

## 7. Architecture Diagram

See `diagrams/architecture.mmd`.

```
[Retrieval Layer] → [Segment Router] → [Compressor Registry]
                                              ↓
                                    [Budget Allocator]
                                              ↓
                              [Compressed Context Assembler]
                                              ↓
                                        [LLM Call]
```

---

## 8. Sequence Diagram

See `diagrams/sequence.mmd`.

---

## 9. Implementation Guide

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
# Key packages: openai, tiktoken, scikit-learn, sentence-transformers
```

### Step 2: Configure environment

```bash
cp .env.example .env
# Set OPENAI_API_KEY, MODEL, COMPRESSION_STRATEGY, TOKEN_BUDGET
```

### Step 3: Run the PoC

```bash
# Demo mode (no API key needed)
DEMO_MODE=true python src/main.py

# Live mode
python src/main.py
```

### Step 4: Understand the core compressor

The `context_compression_core.py` module exposes three compressors behind a unified interface:

```python
from context_compression_core import compress_context

result = compress_context(
    segments={"history": conversation_text, "docs": retrieved_docs},
    query="What is the refund policy for annual subscriptions?",
    strategy="extractive",   # or "abstractive" or "llmlingua"
    token_budget=1000
)
print(result.compressed_text)   # compressed context string
print(result.original_tokens)   # token count before compression
print(result.compressed_tokens) # token count after compression
print(result.compression_ratio) # e.g. 0.42 means 42% of original
```

### Step 5: Integrate into a RAG pipeline

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=your_vector_store.as_retriever()
)
```

### Step 6: Verify compression quality

After implementing, validate that critical information is preserved by running the included evaluation harness:

```bash
python src/main.py --eval
# Outputs: compression_ratio, entity_retention_rate, answer_accuracy
```

---

## 10. Benefits and Trade-offs

| Benefit | Trade-off |
|---|---|
| Reduces input token cost by 40–80% | Adds compression latency (10–200ms for extractive; 500–2000ms for abstractive) |
| Improves model answer quality by reducing middle-context noise | Abstractive compression can hallucinate — the summary may introduce errors not in the source |
| Enables longer effective memory in agents | Compression overhead cost (LLM call for abstractive) must be recouped by savings on main model |
| Query-aware compression surfaces the most relevant content | Over-compression discards critical context; requires tuning per use-case |
| Extractive compression is fully deterministic and auditable | Extractive methods cannot fuse cross-sentence information |

---

## 11. Performance Characteristics

**Extractive (TF-IDF):**
- Latency: P50 ≈ 5ms, P95 ≈ 20ms for 4k-token input (pure CPU, no model call)
- Memory: O(n × m) where n = sentences, m = vocabulary size (TF-IDF matrix)
- Scales linearly with input length

**Extractive (Embedding):**
- Latency: P50 ≈ 80ms, P95 ≈ 150ms (one embedding API call for the query + batch sentence embedding)
- Batching sentences into a single embedding request keeps latency bounded

**Abstractive (LLM summarisation with gpt-4o-mini):**
- Latency: P50 ≈ 800ms, P95 ≈ 2,000ms
- Adds one full LLM round-trip; can be parallelised with other pre-processing steps
- Token cost: ~$0.0006 per 4k-token compression with gpt-4o-mini

**LLMLingua-2:**
- Latency: P50 ≈ 300ms on CPU, P50 ≈ 50ms on GPU
- Requires loading a ~100MB proxy model; unsuitable for serverless cold starts
- Compression ratio 2–5× on general benchmarks (LongBench, arXiv:2310.05736)

**Recommendation:** Use extractive (TF-IDF) as the default for document segments; abstractive for conversation history where coherence matters; LLMLingua-2 for ultra-high compression scenarios (>5×) where GPU inference is available.

---

## 12. Security Considerations

**Prompt injection via compressed content (OWASP LLM01):** A retrieved document may contain adversarial text designed to survive compression and manipulate the LLM. Extractive compression preserves full sentences verbatim, including injected instructions. Mitigation: apply an input sanitisation pass before compression that strips instruction-like patterns (`"Ignore previous instructions"`, `"System:"` prefixes).

**Data leakage via abstractive compression (OWASP LLM06):** A summarisation LLM call sends potentially sensitive context (PII, financial data) to a third-party model. Ensure the compression model operates under the same data handling agreements as the main model, or use a locally-hosted compression model for sensitive segments.

**Compression ratio as an oracle (OWASP LLM04):** An attacker who can measure token usage in API responses may infer the effectiveness of their injection (high compression ratio = their content was preserved). Normalise or add noise to token count telemetry exposed externally.

**Entity extraction leakage:** If entity protection re-injects extracted named entities into the compressed output, ensure the entity extractor does not surface entities from confidential segments into non-confidential ones in multi-tenant contexts.

---

## 13. Cost Analysis

**Baseline (no compression):** 10,000 tokens/request × 500 requests/day × $5.00/1M tokens (GPT-4o input) = **$25.00/day**

**With extractive compression (60% reduction):** 4,000 tokens/request × 500 requests/day × $5.00/1M = **$10.00/day**. Zero compression overhead cost (CPU-only).

**With abstractive compression (75% reduction):** 2,500 tokens/request × 500 × $5.00/1M = **$6.25/day** on main model. Plus compression cost: 10,000 tokens/request × 500 × $0.15/1M (gpt-4o-mini input) = **$0.75/day**. Net: **$7.00/day**.

**Break-even analysis:** Abstractive compression saves $18.00/day vs baseline but costs $0.75/day overhead. Net saving: $17.25/day ($6,296/year). Break-even is immediate at any volume above ~50 requests/day.

**LLMLingua-2 (on-premise GPU):** Eliminates abstractive compression API cost entirely. At 500 req/day, GPU amortisation cost is typically lower than the $0.75/day API cost at >1M requests/month.

---

## 14. Best Practices

1. **Always condition compression on the query.** A generic summary preserves different content than a query-conditioned one. Pass the current user query to every compressor call.

2. **Set a minimum segment size threshold.** Segments under 500 tokens should bypass compression — the overhead exceeds the saving. Implement a token count check before dispatching to any compressor.

3. **Protect named entities before compressing.** Extract entities (names, IDs, amounts, dates) before compression and verify they survive. Re-inject any that are dropped.

4. **Use the cheapest compressor that meets quality requirements.** TF-IDF extraction costs nothing extra; use it as the default before escalating to embedding-based or abstractive methods.

5. **Measure compression quality, not just compression ratio.** A 90% compression ratio that drops the answer is worse than a 40% ratio that preserves it. Use a held-out QA evaluation set to calibrate target ratios.

6. **Parallelise the compression step.** If multiple segments require compression, compress them concurrently. The total compression latency equals the slowest segment, not the sum.

7. **Cache compressed versions of static content.** System prompt sections and frequently-retrieved documents compress to the same output for any given query type. Cache the compressed form with a TTL matching document update frequency.

8. **Apply compression after retrieval re-ranking, not before.** Re-ranking selects the most relevant documents; compressing pre-ranked documents may discard useful content from documents that would have been dropped anyway.

9. **Log original and compressed token counts per request.** This telemetry is the primary signal for monitoring compression effectiveness and detecting regressions when compressors are updated.

10. **Test compression with adversarial inputs.** Include injection attempts in your test suite to validate that sanitisation survives compression.

---

## 15. Anti-Patterns

### Anti-Pattern 1: The Uniform Compressor
**What it looks like:** Applying the same compression ratio (e.g., 50%) to every context segment regardless of its relevance to the query.
**Why it fails:** A segment containing the direct answer to the query needs 0% compression; a segment of pleasantries needs 100%. Uniform compression randomly discards answers.
**Fix:** Route segments to query-aware compressors; allocate budget proportionally to relevance scores.

### Anti-Pattern 2: Compress Then Retrieve
**What it looks like:** Compressing the document corpus before indexing it, then retrieving compressed documents.
**Why it fails:** Compression discards tokens that may be relevant to future queries not known at index time. Retrieval quality degrades because compressed text has lower embedding fidelity.
**Fix:** Index full documents; compress retrieved documents at query time against the actual query.

### Anti-Pattern 3: The Summary Loop
**What it looks like:** Summarising the conversation at every turn, including previous summaries in subsequent summaries.
**Why it fails:** Each summarisation pass introduces potential hallucination. After 5–10 passes, the summary diverges significantly from the original conversation.
**Fix:** Summarise the raw conversation turns, not prior summaries. Keep a pointer to the last summarised turn and only summarise new turns incrementally.

### Anti-Pattern 4: Ignoring Compression Latency in SLA Budgets
**What it looks like:** Adding a 500ms abstractive compression step to a pipeline with a 600ms end-to-end SLA.
**Why it fails:** The compression step consumes the entire latency budget before the main LLM call begins.
**Fix:** Profile compression latency in your infrastructure before deploying. Use extractive compression for latency-sensitive paths; reserve abstractive for async or background processing.

### Anti-Pattern 5: Over-Trusting the Compressor
**What it looks like:** Removing all logging of the original context after compression, relying solely on the compressed version for debugging.
**Why it fails:** When an answer is wrong, the root cause may be in what the compressor discarded — which is invisible if the original is not stored.
**Fix:** Log both original and compressed context (with appropriate data retention policies). Store compression diffs during the evaluation phase.

---

## 16. Common Mistakes

**Mistake 1: Splitting on periods without handling abbreviations**
- Symptom: Sentences are split mid-abbreviation (e.g., "Dr. Smith" → ["Dr.", "Smith attended..."]).
- Root cause: Naive `text.split('.')` used as sentence splitter.
- Fix: Use a proper sentence boundary detector (spaCy `sentencizer`, NLTK `sent_tokenize`, or the `sentence-splitter` library). These handle abbreviations, decimal numbers, and ellipses.

**Mistake 2: Compressing the system prompt**
- Symptom: LLM ignores behavioural constraints (tone, format, safety rules) that were in the system prompt.
- Root cause: The system prompt was included in the compression budget and had tokens removed.
- Fix: Mark the system prompt as a protected segment — never include it in the compression budget. Only compress conversation history, documents, and tool outputs.

**Mistake 3: Token budget specified in words, not tokens**
- Symptom: Compressed context exceeds the target context window size.
- Root cause: "300 words" was used as the budget proxy; 300 words ≈ 400–500 tokens for typical English text.
- Fix: Always measure budgets in tokens using the model's actual tokeniser (`tiktoken` for OpenAI models). A 300-token budget is not 300 words.

---

## 17. Production Checklist

- [ ] Compression is bypassed for segments under 500 tokens
- [ ] System prompt is excluded from all compression budgets
- [ ] Named entity extraction runs before compression and verifies retention
- [ ] Token budgets are measured using the target model's tokeniser (tiktoken)
- [ ] Compression step is parallelised across independent segments
- [ ] Abstractive compression model operates under the same data handling agreement as the main model
- [ ] Input sanitisation strips instruction-injection patterns before compression
- [ ] Original and compressed token counts are logged per request
- [ ] Compression latency (P95) fits within overall request SLA budget
- [ ] Fallback bypasses compression if the compressor itself raises an exception
- [ ] A QA evaluation set validates compression quality before production deployment
- [ ] Static document compressions are cached with appropriate TTL
- [ ] Compression ratios are monitored via dashboard; alerts fire if ratio degrades >10%
- [ ] Incremental summarisation is used for conversation history (not recursive re-summarisation)
- [ ] Load tests confirm compressor throughput matches the peak request rate

---

## 18. References

[1] Liu, N. F. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172. https://arxiv.org/abs/2307.03172

[2] Jiang, H. et al. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." EMNLP 2023. arXiv:2310.05736. https://arxiv.org/abs/2310.05736

[3] Pan, Z. et al. (2024). "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression." arXiv:2403.12968. https://arxiv.org/abs/2403.12968

[4] LangChain Documentation (2024). "Contextual Compression." https://python.langchain.com/docs/how_to/contextual_compression/

[5] OpenAI (2024). "Tiktoken — Fast BPE Tokeniser for Use with OpenAI Models." https://github.com/openai/tiktoken

[6] Microsoft Research (2024). "LLMLingua GitHub Repository." https://github.com/microsoft/LLMLingua

---

## 19. Summary

Context Compression exists because longer context is not the same as better context. Transformer attention is positionally biased, token costs scale linearly, and most runtime context in production agents contains 40–60% content irrelevant to the current query. By inserting a compression middleware layer that applies query-aware extractive, abstractive, or token-level compression to each context segment, production systems routinely achieve 50–75% token reductions with minimal quality loss. The key discipline is to compress selectively — different segment types require different strategies, and over-compression is a real failure mode that requires ongoing QA monitoring rather than a one-time configuration.

---

## 20. Exercises

**Beginner:** Run `DEMO_MODE=true python src/main.py` and examine the printed compression ratio. Change the `TOKEN_BUDGET` in `.env.example` from 1000 to 500 and observe how the output changes.

**Intermediate:** Modify `context_compression_core.py` to replace the TF-IDF scorer with an embedding cosine similarity scorer using OpenAI's `text-embedding-3-small`. Compare the two approaches on the included test set (`sample_input.json`) and report which sentences each method retains.

**Advanced:** Extend the PoC to support incremental conversation summarisation: maintain a rolling summary that only processes new turns (turns not yet included in the previous summary) and prepends the existing summary rather than re-summarising the entire history.

**Expert:** Implement a compression quality benchmark that runs the `sample_input.json` questions through both the uncompressed and compressed pipeline, uses an LLM-as-judge to score answer quality (1–5), and produces a trade-off chart of compression ratio vs. answer quality score across five compression ratio settings (0.2, 0.4, 0.6, 0.8, 1.0).

**Research:** Read LLMLingua-2 (arXiv:2403.12968) and identify one limitation of the data distillation approach for domain-specific corpora (e.g., medical or legal text) that is not addressed in the paper. Propose a mitigation strategy.

---

## 21. Interview Questions

1. **Conceptual:** Explain to a non-engineer why sending more text to an LLM can make its answers worse.

2. **Technical:** What is the difference between extractive and abstractive context compression, and when would you choose one over the other?

3. **Technical:** Why must sentence selection in extractive compression return sentences in their original order rather than in similarity-rank order?

4. **Design:** How would you architect a context compression pipeline for an agent that processes three types of context segments — conversation history, retrieved documents, and tool call results — each requiring different compression strategies?

5. **Trade-off:** When does abstractive compression save money, and when does it cost more than it saves? Show your reasoning with approximate token counts.

6. **Design:** How would you handle a query where the answer depends on information distributed across three separate sentences in a retrieved document, and extractive compression would only retain one of them?

7. **Debugging:** A production agent begins giving incorrect answers after a context compression layer is deployed. Describe your diagnostic process step by step.

8. **Security:** A red team report flags that an adversarial document in the retrieval corpus survived context compression and caused the LLM to follow injected instructions. What compression-layer and pre-compression defences would you implement?

9. **Design:** How would you design a context compression system for a 10M requests/day workload where P95 latency must stay under 200ms end-to-end?

10. **Trade-off:** When would you choose LLMLingua-2 over TF-IDF-based extractive compression, and what infrastructure requirement does that choice impose?
