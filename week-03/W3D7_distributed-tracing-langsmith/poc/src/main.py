#!/usr/bin/env python3
"""
W3D7 — Distributed Tracing (LangSmith)
=======================================
Demonstrates: End-to-end RAG pipeline with per-span tracing via LangSmith.
Each pipeline step (retrieve, rerank, assemble, generate, validate) emits
a child span. In live mode, the full run tree appears in your LangSmith
project. In demo mode, the pipeline runs locally with pre-computed output.

Run (demo mode):  DEMO_MODE=true python src/main.py
Run (live mode):  python src/main.py   (requires OPENAI_API_KEY + LANGCHAIN_API_KEY)
Run tests:        pytest tests/ -v
"""

import os
import json
import time
from pathlib import Path

from config import load_config
from tracing_core import (
    run_demo_pipeline,
    retrieve_documents,
    rerank_documents,
    assemble_context,
    validate_answer,
    PipelineResult,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = load_config()

SAMPLE_INPUT_PATH = Path(__file__).parent.parent / "sample_input.json"


def load_sample_input() -> dict:
    if SAMPLE_INPUT_PATH.exists():
        return json.loads(SAMPLE_INPUT_PATH.read_text())
    return {"query": "What is the return window for electronics?"}


# ---------------------------------------------------------------------------
# Live mode — real OpenAI call with LangSmith tracing via @traceable
# ---------------------------------------------------------------------------

def run_live(query: str) -> PipelineResult:
    """
    Execute the pipeline with real API calls.
    @traceable decorators are applied here so that imports only happen
    when live mode is actually requested (keeps demo mode dependency-free).
    """
    try:
        from openai import OpenAI
        from langsmith import traceable
        import uuid, time as _time
        from tracing_core import (
            RetrievedDocument, PipelineResult,
            DEMO_DOCUMENTS, validate_answer
        )

        client = OpenAI(api_key=cfg.openai_api_key)
        run_id = str(uuid.uuid4())
        start = _time.time()
        spans = 0

        # Each step is wrapped with @traceable at call time via a helper.
        # In a production codebase, place @traceable directly on the function
        # definitions in tracing_core.py; here we wrap inline for clarity.

        @traceable(run_type="retriever", name="retrieve_documents")
        def _retrieve(q: str):
            return retrieve_documents(q, top_k=3)

        @traceable(run_type="chain", name="rerank_documents")
        def _rerank(q: str, docs):
            return rerank_documents(q, docs)

        @traceable(run_type="chain", name="assemble_context")
        def _assemble(docs):
            return assemble_context(docs)

        @traceable(run_type="llm", name="generate_answer")
        def _generate(context: str, question: str) -> dict:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": f"Answer using only:\n{context}"},
                    {"role": "user", "content": question},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            return {
                "answer": resp.choices[0].message.content,
                "tokens": resp.usage.total_tokens,
                "model": resp.model,
            }

        @traceable(run_type="chain", name="validate_answer")
        def _validate(answer: str, docs):
            return validate_answer(answer, docs)

        docs   = _retrieve(query);   spans += 1
        ranked = _rerank(query, docs); spans += 1
        ctx    = _assemble(ranked);  spans += 1
        gen    = _generate(ctx, query); spans += 1
        _validate(gen["answer"], ranked); spans += 1

        return PipelineResult(
            run_id=run_id,
            answer=gen["answer"],
            retrieved_docs=ranked,
            tokens_used=gen["tokens"],
            latency_ms=round((_time.time() - start) * 1000, 2),
            model=gen["model"],
            spans_captured=spans,
        )

    except ImportError as e:
        print(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")
        raise
    except Exception as e:
        print(f"Live mode failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Demo mode wrapper
# ---------------------------------------------------------------------------

def run_demo(query: str) -> PipelineResult:
    print("\n⚠️  Running in DEMO MODE — output is pre-computed (no API calls made)\n")
    return run_demo_pipeline(query)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("\nDistributed Tracing (LangSmith) Demo")
    print("=" * 45)

    input_data = load_sample_input()
    query = input_data.get("query", "What is the return policy?")
    print(f"Query: {query}\n")

    if cfg.demo_mode:
        result = run_demo(query)
    else:
        print(f"Live mode | Model: {cfg.model} | Tracing: {cfg.tracing_enabled}")
        result = run_live(query)

    print(f"Answer:         {result.answer}")
    print(f"Run ID:         {result.run_id}")
    print(f"Spans captured: {result.spans_captured}")
    print(f"Tokens used:    {result.tokens_used}")
    print(f"Latency:        {result.latency_ms} ms")
    print(f"Model:          {result.model}")

    if cfg.tracing_enabled:
        print(f"\nTrace URL: https://smith.langchain.com/o/projects/{cfg.langsmith_project}/runs/{result.run_id}")

    print("\n✅ Concept demonstrated: 5-span RAG pipeline with per-step input/output capture.")
    print("   In live mode, open LangSmith to see the full run tree.")
    print("\n📚 See 02_technical-doc/technical-document.md for the full deep dive.")


if __name__ == "__main__":
    main()
