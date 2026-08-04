# W1D3 — Naive vs. Agentic RAG

**Series:** AI Engineering Production Playbook
**Vertical:** Advanced RAG
**Week 1 / Day 3**

## What This Demonstrates

Naive RAG retrieves documents by vector similarity in a single pass. Agentic RAG decomposes a complex query into atomic sub-questions, retrieves targeted evidence for each, validates it against a similarity threshold, and synthesises a final answer with citations. This PoC runs both pipelines on the same query so you can see the structural difference in retrieval calls, evidence coverage, and answer quality.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode available without one)

## Quickstart

```bash
# 1. Navigate to this folder
cd artifacts/W1D3_naive-vs-agentic-rag/03_poc-code

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY, or leave blank for demo mode

# 4. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode uses a five-document in-memory corpus and keyword-overlap similarity. No network calls are made. The output demonstrates the same structural difference between naive and agentic retrieval as the live mode.

Expected demo output:

```
RAG Pipeline Demo — Naive vs. Agentic
==========================================
Running in demo mode (no API key). Output uses pre-computed corpus.

Query: My order was marked delivered but I never received it — what do I do
       and does my Gold membership change my options?

--- Naive RAG ---
Sub-questions : ['My order was marked delivered...']
Retrieval calls: 1  |  Chunks used: 1
Answer:
  Standard members must report missing or undelivered orders within 7 days... [1]

--- Agentic RAG ---
Sub-questions : ['...what do I do?', 'Does my Gold membership change my options?']
Retrieval calls: 2  |  Chunks used: 2
Answer:
  Standard members must report missing orders within 7 days... [1]
  Gold members receive an extended 30-day dispute window... [2]
```

## Run Tests

```bash
pytest tests/ -v
```

All tests run offline — no API key required. The test suite covers:

- Similarity scoring logic
- Query decomposition (single-hop and multi-hop)
- Chunk retrieval with threshold filtering
- Naive pipeline end-to-end
- Agentic pipeline end-to-end (including unanswerable sub-question handling)
- Parametrised tests for known query/chunk pairs
- Integration comparison: agentic makes more retrieval calls than naive on multi-hop queries

## File Structure

```
03_poc-code/
├── src/
│   ├── main.py          # Entry point — runs both pipelines and prints comparison
│   ├── rag_core.py      # QueryDecomposer, ChunkRetriever, AgenticRAGPipeline, NaiveRAGPipeline
│   └── config.py        # Config dataclass loaded from environment variables
├── tests/
│   └── test_rag_core.py # pytest unit tests (offline)
├── requirements.txt
├── .env.example
├── sample_input.json    # Demo query
└── sample_output.json   # Expected output for the demo query
```

## Key Concepts

**QueryDecomposer** — breaks a complex query into atomic sub-questions (demo: keyword heuristics; live: LLM call).

**ChunkRetriever** — fetches top-k chunks from the corpus, filters by similarity threshold, increments a call counter for comparison reporting.

**AgenticRAGPipeline** — orchestrates the full loop: decompose → retrieve per sub-question → validate → reformulate on failure → synthesise from verified evidence.

**NaiveRAGPipeline** — single retrieval pass against the raw query; no decomposition, validation, or retry.

## Extending to Live Mode

1. Set `OPENAI_API_KEY` in your `.env` file and set `DEMO_MODE=false`.
2. Replace `ChunkRetriever._live_retrieve()` with your vector store client (FAISS, Pinecone, Weaviate).
3. The decomposition and synthesis steps already call OpenAI — they activate automatically when `demo_mode=False`.

## Read More

- [Technical Documentation](../02_technical-doc/technical-document.md)
- [Architecture Diagram](../02_technical-doc/diagrams/architecture.mmd)
- [Sequence Diagram](../02_technical-doc/diagrams/sequence.mmd)
- [LinkedIn Post](../01_linkedin-post/linkedin_post.md)

## References

- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv:2005.11401
- Asai et al. (2023). "Self-RAG." arXiv:2310.11511
- LangGraph: https://github.com/langchain-ai/langgraph
- LlamaIndex Sub Question Query Engine: https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/
