# W3D1 — Prompt Distillation

**Series:** AI Engineering Production Playbook
**Vertical:** Prompt Engineering & Schemas
**Week 3 / Day 1**

---

## What This Demonstrates

How to compress a verbose, high-accuracy "teacher" prompt into a minimal "student" prompt that retains ≥90% of the teacher's accuracy — reducing token cost by 60%+ without retraining the model.

The PoC implements a greedy sentence-pruning distillation loop with a configurable accuracy floor, plus a cost-savings calculator that projects monthly and annual savings based on call volume.

---

## The Problem It Solves

Production LLM prompts accumulate tokens over time — edge-case instructions, defensive rules, and redundant examples added after each incident. A 200-token prompt becomes 1,800 tokens within months. At 200,000 calls/month, that overhead costs ~$54/month on gpt-4o-mini — before any user content. Prompt distillation reclaims that cost systematically, with a quantified accuracy trade-off.

---

## Prerequisites

- Python 3.10+
- OpenAI API key (optional — demo mode runs fully offline)

---

## Quickstart

```bash
# 1. Navigate to this folder
cd week-03/W3D1_prompt-distillation/poc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (skip for demo mode)
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

# 4. Run
python src/main.py
```

---

## Demo Mode (No API Key Required)

```bash
DEMO_MODE=true python src/main.py
```

Expected output:

```
 Prompt Distillation Demo
==================================================

Task       : classify_document
Input text : This Non-Disclosure Agreement ("Agreement") is entered into as of...

[DEMO MODE] Running with pre-computed output — no API key required.

  Teacher prompt tokens : 1800
  Student prompt tokens : 640
  Token reduction       : 64.4%
  Teacher accuracy      : 96.2%
  Student accuracy      : 95.4%
  Accuracy delta        : -0.8%
  Model                 : demo
  Latency (ms)          : 0

 Concept demonstrated: A 1800-token teacher prompt distilled into a
  640-token student prompt with <1pp accuracy delta on held-out eval.
```

---

## Run Tests

```bash
# All tests — fully offline
pytest tests/ -v

# With demo mode forced
DEMO_MODE=true pytest tests/ -v
```

Expected: **17 tests pass, 0 fail, 0 network calls.**

---

## Project Structure

```
poc/
├── src/
│   ├── main.py                # Entry point — run this file
│   ├── distillation_core.py   # Core distillation logic (pure functions)
│   └── config.py              # Config dataclass + env loader
├── tests/
│   └── test_distillation.py   # 17 unit tests across 4 test classes
├── README.md
├── requirements.txt
├── .env.example
├── sample_input.json          # Legal NDA document classification task
└── sample_output.json         # Pre-computed demo result
```

---

## Key Functions

| Function | Module | Purpose |
|---|---|---|
| `build_teacher_prompt()` | `distillation_core` | Assemble the verbose starting prompt |
| `build_student_prompt()` | `distillation_core` | Assemble the minimal distilled prompt |
| `score_prompt_candidate()` | `distillation_core` | Evaluate a prompt against a labelled eval set |
| `distill_prompt()` | `distillation_core` | Greedy pruning loop with accuracy floor |
| `compute_token_savings()` | `distillation_core` | Project monthly/annual cost savings |
| `run_distillation_demo()` | `distillation_core` | Offline pre-computed path |
| `run_distillation_live()` | `distillation_core` | Live path via OpenAI API |
| `load_config()` | `config` | Load all config from environment variables |

---

## Configuration

All configuration is driven by environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | API key — absent triggers demo mode |
| `MODEL` | `gpt-4o-mini` | LLM model for scoring |
| `DEMO_MODE` | `false` | Force offline mode even with a key |
| `ACCURACY_FLOOR` | `0.90` | Minimum student accuracy fraction |
| `MAX_DISTILLATION_ITERATIONS` | `5` | Max pruning rounds |
| `DAILY_CALLS` | `6667` | Projected calls/day for savings calc |
| `COST_PER_1M_TOKENS` | `0.15` | Input token price USD/1M (gpt-4o-mini) |

---

## How the Distillation Loop Works

```
1. Build teacher prompt (verbose, edge-case-covering)
2. Score teacher against held-out eval set  →  teacher_accuracy
3. For each pruning iteration:
   a. Identify longest removable sentence
   b. Remove it → create candidate student prompt
   c. Score candidate against eval set
   d. If accuracy >= accuracy_floor: accept removal, continue
   e. Else: sentence is load-bearing, stop
4. Return student prompt + metrics
```

This greedy heuristic is illustrative. Production implementations use DSPy's `MIPROv2` optimizer, which searches over instruction candidates and few-shot example combinations via Bayesian optimisation.

---

## Production Considerations

**Eval set quality is everything.** The accuracy floor is only meaningful if the eval set reflects your real traffic distribution. A skewed eval set produces a student prompt that overfits to the eval and degrades on production inputs.

**Version your prompts.** Tag each teacher/student pair with a hash of their content. Store accuracy scores alongside. This gives you a rollback path if a distilled prompt degrades after a model update.

**Re-distill after model updates.** A student prompt optimised against `gpt-4o-mini-2024-07-18` may not transfer cleanly to `gpt-4o-mini-2025-xx-xx`. Re-run the distillation loop after each model version change.

**Monitor accuracy in production.** Use LLM-as-a-judge (see W1D7) or deterministic label comparison (when ground truth is available) to detect accuracy drift between teacher and student in live traffic.

---

## Read More

- [Technical Documentation](../docs/technical-document.md)
- [Architecture Diagram](../diagrams/architecture.mmd)
- [Sequence Diagram](../diagrams/sequence.mmd)
- [Plain-English Scenarios](../docs/prompt-distillation-layman-scenarios.md)
- [LinkedIn Post](../README.md)

---

## References

- Khattab, O. et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." arXiv:2310.03714
- DSPy documentation — MIPROv2 optimizer: https://dspy.ai/deep-dive/optimizers/miprov2/
- OpenAI pricing reference: https://openai.com/api/pricing/

---

**Series:** AI Engineering Production Playbook — Week 3, Day 1 of 28
**Next:** W3D2 — Context Compression
