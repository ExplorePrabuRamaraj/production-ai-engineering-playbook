# W3D2 — Context Compression

**Series:** AI Engineering Production Playbook
**Vertical:** Context Engineering & Tokens
**Week 3 / Day 2**

## What This Demonstrates

Query-aware context compression: how to reduce LLM input tokens by 40–80% using extractive (TF-IDF sentence scoring), abstractive (LLM summarisation), and hybrid strategies — without discarding the information the model needs to answer the current query.

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode available without one)

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/your-org/production-ai-engineering-playbook

# 2. Navigate to this folder
cd week-03/W3D2_context-compression/poc

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API key — or leave blank for demo mode

# 5. Run
python src/main.py
```

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Demo mode runs the extractive (TF-IDF) compressor entirely offline. No API call is made. Output mirrors the structure of `sample_output.json`.

## Run Tests

```bash
pytest tests/ -v
```

All tests pass offline. External API calls are mocked via `unittest.mock`.

## Compression Strategies

| Strategy | API Call Required | Speed | Best For |
|---|---|---|---|
| `extractive` | No | < 20ms | Documents, tool outputs |
| `abstractive` | Yes (cheap model) | 500–2000ms | Conversation history |
| `hybrid` | Yes (fallback only) | Varies | Mixed segment types |

Set the strategy via environment variable:

```bash
COMPRESSION_STRATEGY=abstractive python src/main.py
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (empty) | API key — blank activates demo mode |
| `MODEL` | `gpt-4o-mini` | Main LLM model |
| `COMPRESSION_MODEL` | `gpt-4o-mini` | Model for abstractive compression |
| `TOKEN_BUDGET` | `1000` | Total token budget across all segments |
| `COMPRESSION_STRATEGY` | `extractive` | `extractive`, `abstractive`, or `hybrid` |
| `MIN_SEGMENT_TOKENS` | `50` | Segments below this bypass compression |
| `DEMO_MODE` | `false` | `true` = run without API key |

## File Structure

```
poc/
├── src/
│   ├── main.py                      # Entry point — run this file
│   ├── context_compression_core.py  # Core compression logic (importable)
│   └── config.py                    # Config dataclass + env loader
├── tests/
│   └── test_context_compression.py  # pytest unit tests (offline)
├── requirements.txt
├── .env.example
├── sample_input.json                # Example multi-segment input
└── sample_output.json               # Expected compressed output
```

## Expected Output (Demo Mode)

```
🚀 Context Compression Demo
==================================================
Query: What is the refund policy for annual subscriptions?

Segments to compress: ['history', 'docs', 'tool_output']

⚠️  Running in DEMO MODE — no API call made

Results:
{
  "overall_compression_ratio": 0.445,
  "total_original_tokens": 454,
  "total_compressed_tokens": 202,
  ...
}

✅ Concept demonstrated: 55% token reduction via query-aware context compression.
```

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Layman Scenarios](../docs/context-compression-layman-scenarios.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [LinkedIn Post](../README.md)

## Key References

- Liu et al. (2023). "Lost in the Middle." arXiv:2307.03172
- Jiang et al. (2023). "LLMLingua." arXiv:2310.05736
- LangChain ContextualCompressionRetriever: https://python.langchain.com/docs/how_to/contextual_compression/
