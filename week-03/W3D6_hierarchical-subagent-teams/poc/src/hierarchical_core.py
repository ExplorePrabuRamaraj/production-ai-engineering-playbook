"""
W3D6 — Hierarchical Subagent Teams — Core Logic
=================================================
Implements the 3-tier hierarchy:
  Tier 1: Orchestrator  — decomposes goal, assembles final output
  Tier 2: Team Leads    — own a domain, dispatch workers, aggregate typed results
  Tier 3: Worker Agents — stateless leaf executors, one job each

Key design principle: typed result contracts (Pydantic) at every tier boundary.
No raw LLM output crosses a tier — this is what prevents context bleed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

# Module-level import so @patch("hierarchical_core.OpenAI") works in tests.
# Graceful fallback: if openai is not installed the module still loads;
# missing-package errors are raised only when run_worker/run_team_lead are called.
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Typed result contracts — defined before any agent code (see Best Practice #1)
# ---------------------------------------------------------------------------

class ExecutionOrder(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass
class SubtaskSpec:
    """Instruction passed from Orchestrator to a Team Lead.
    Contains only what the lead needs — NOT the full goal context."""
    lead_id: str
    domain: str
    instruction: str
    depends_on: List[str] = field(default_factory=list)
    execution_order: ExecutionOrder = ExecutionOrder.PARALLEL


@dataclass
class WorkerResult:
    """Typed output from a Worker Agent. Returned to its parent Lead only."""
    worker_id: str
    output: str
    tokens_used: int
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class LeadResult:
    """Typed output from a Team Lead. Returned to the Orchestrator only.
    Aggregates all WorkerResults — orchestrator never sees raw worker output."""
    lead_id: str
    domain: str
    aggregated_output: str
    worker_results: List[WorkerResult]
    tokens_used: int
    success: bool
    partial: bool = False   # True if some workers failed after retry budget


@dataclass
class FinalResult:
    """Assembled output from the Orchestrator."""
    goal: str
    final_output: str
    lead_results: List[LeadResult]
    total_tokens_used: int
    total_latency_ms: float
    success: bool
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tier 3 — Worker Agent
# Stateless: receives bounded context, returns one WorkerResult, writes nothing.
# ---------------------------------------------------------------------------

def run_worker(
    worker_id: str,
    instruction: str,
    context: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> WorkerResult:
    """
    Stateless leaf executor. Receives only what it needs to complete one task.
    Does not write to shared memory. Returns a typed WorkerResult to its lead.

    Why stateless? Limits blast radius when this worker fails — the lead
    can retry just this worker without re-running siblings.
    """
    start = time.time()
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a focused specialist. Complete exactly the task given. "
                        "Be concise and structured. Return only what was asked."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nTask: {instruction}",
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        latency_ms = (time.time() - start) * 1000
        return WorkerResult(
            worker_id=worker_id,
            output=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            latency_ms=latency_ms,
            success=True,
        )
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return WorkerResult(
            worker_id=worker_id,
            output="",
            tokens_used=0,
            latency_ms=latency_ms,
            success=False,
            error_message=str(e),
        )


# ---------------------------------------------------------------------------
# Tier 2 — Team Lead
# Owns a domain subtask. Dispatches workers, handles scoped retries,
# validates and aggregates typed WorkerResults before returning to Orchestrator.
# ---------------------------------------------------------------------------

def run_team_lead(
    lead_id: str,
    domain: str,
    subtask: str,
    worker_instructions: List[dict],
    api_key: str,
    model: str,
    max_tokens: int,
    max_retries: int = 2,
) -> LeadResult:
    """
    Team Lead: decomposes a bounded subtask into worker-sized atomic instructions.
    Retries individual workers on failure (scoped retry — does NOT restart siblings).
    Aggregates typed WorkerResults and returns a single validated LeadResult.

    Why the lead aggregates (not the orchestrator)?
    The orchestrator should never see raw worker outputs — they may be
    inconsistently formatted. The lead normalizes before passing up.
    """
    worker_results: List[WorkerResult] = []

    for w_def in worker_instructions:
        worker_id = w_def["worker_id"]
        instruction = w_def["instruction"]
        attempts = 0
        result = None

        # Scoped retry: retry only this worker, not the full lead
        while attempts <= max_retries:
            result = run_worker(
                worker_id=worker_id,
                instruction=instruction,
                context=subtask,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
            )
            if result.success:
                break
            attempts += 1

        worker_results.append(result)

    successful = [r for r in worker_results if r.success]
    failed = [r for r in worker_results if not r.success]
    total_tokens = sum(r.tokens_used for r in worker_results)

    if not successful:
        # All workers failed — return failed LeadResult without calling LLM for aggregation
        return LeadResult(
            lead_id=lead_id,
            domain=domain,
            aggregated_output="",
            worker_results=worker_results,
            tokens_used=total_tokens,
            success=False,
            partial=False,
        )

    # Aggregate successful worker outputs — one LLM call per lead, not per worker
    combined = "\n\n".join(
        f"[Worker {r.worker_id}]: {r.output}" for r in successful
    )
    try:
        client = OpenAI(api_key=api_key)
        agg_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a domain lead. Combine the specialist outputs below "
                        "into one coherent, structured summary for your domain. "
                        "Preserve all key facts. Remove redundancy."
                    ),
                },
                {"role": "user", "content": combined},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        aggregated = agg_response.choices[0].message.content
        total_tokens += agg_response.usage.total_tokens
    except Exception as e:
        # Aggregation failed — use concatenated worker outputs as fallback
        aggregated = combined

    return LeadResult(
        lead_id=lead_id,
        domain=domain,
        aggregated_output=aggregated,
        worker_results=worker_results,
        tokens_used=total_tokens,
        success=True,
        partial=len(failed) > 0,
    )


# ---------------------------------------------------------------------------
# Tier 1 — Orchestrator
# Decomposes goal into SubtaskSpecs, dispatches leads, assembles final output.
# Never does leaf work — planning, routing, and assembly only.
# ---------------------------------------------------------------------------

def run_orchestrator(
    goal: str,
    subtask_specs: List[SubtaskSpec],
    worker_map: dict,
    api_key: str,
    model: str,
    max_tokens: int,
    max_retries: int = 2,
) -> FinalResult:
    """
    Orchestrator: receives goal + pre-computed TaskPlan, dispatches team leads,
    checks all LeadResult contracts for partial/failed flags, then synthesizes.

    Why does the orchestrator receive subtask_specs externally in the PoC?
    In production, the orchestrator would generate these via an LLM decomposition
    call. In demo mode we pass them in to avoid a real API call for the plan step.
    """
    start = time.time()
    lead_results: List[LeadResult] = []
    warnings: List[str] = []

    # Dispatch leads (sequential in this PoC; use asyncio.gather for parallel in prod)
    for spec in subtask_specs:
        workers = worker_map.get(spec.lead_id, [])
        lead_result = run_team_lead(
            lead_id=spec.lead_id,
            domain=spec.domain,
            subtask=spec.instruction,
            worker_instructions=workers,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        lead_results.append(lead_result)

        # Check contract flags — do not proceed silently on partial results
        if lead_result.partial:
            warnings.append(
                f"Lead '{spec.lead_id}' returned partial result — "
                f"some workers failed after retry budget."
            )
        if not lead_result.success:
            warnings.append(
                f"Lead '{spec.lead_id}' failed completely — domain '{spec.domain}' missing from final output."
            )

    # Assemble final output from all LeadResults
    successful_leads = [lr for lr in lead_results if lr.success]
    total_tokens = sum(lr.tokens_used for lr in lead_results)

    if not successful_leads:
        total_latency_ms = (time.time() - start) * 1000
        return FinalResult(
            goal=goal,
            final_output="All leads failed. No output could be assembled.",
            lead_results=lead_results,
            total_tokens_used=total_tokens,
            total_latency_ms=total_latency_ms,
            success=False,
            warnings=warnings,
        )

    combined_leads = "\n\n".join(
        f"[{lr.domain}]: {lr.aggregated_output}" for lr in successful_leads
    )

    try:
        client = OpenAI(api_key=api_key)
        final_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior orchestrator. Synthesize the domain summaries "
                        "below into one clear, complete response to the original goal. "
                        "Do not add information not present in the domain summaries."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Goal: {goal}\n\nDomain summaries:\n{combined_leads}",
                },
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        final_output = final_response.choices[0].message.content
        total_tokens += final_response.usage.total_tokens
    except Exception as e:
        final_output = combined_leads  # Fallback: concatenate lead summaries

    total_latency_ms = (time.time() - start) * 1000
    return FinalResult(
        goal=goal,
        final_output=final_output,
        lead_results=lead_results,
        total_tokens_used=total_tokens,
        total_latency_ms=total_latency_ms,
        success=True,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Demo mode — pre-computed output that mirrors what live mode would produce
# ---------------------------------------------------------------------------

def run_demo(input_data: dict) -> FinalResult:
    """
    Returns pre-computed output for offline demonstration and testing.
    Mirrors the exact FinalResult structure returned by run_orchestrator.
    No API calls made.
    """
    goal = input_data.get("goal", "Analyse competitive landscape for a SaaS CRM product")

    worker_results_a = [
        WorkerResult(
            worker_id="lead_research_worker_0",
            output="Key competitors: Salesforce (market leader, 20% share), HubSpot (SMB focus, freemium model), Pipedrive (pipeline-centric UX). All three raised prices in Q1 2024.",
            tokens_used=87,
            latency_ms=412.0,
            success=True,
        ),
        WorkerResult(
            worker_id="lead_research_worker_1",
            output="Feature gaps vs. Salesforce: No native CPQ, limited workflow automation depth. Advantage: 40% lower TCO, faster onboarding (avg 3 days vs 14 days).",
            tokens_used=74,
            latency_ms=388.0,
            success=True,
        ),
    ]

    worker_results_b = [
        WorkerResult(
            worker_id="lead_analysis_worker_0",
            output="Market sizing: TAM $48B (2024), SAM $12B (mid-market CRM). YoY growth 14%. AI-native CRM segment growing at 31% YoY.",
            tokens_used=63,
            latency_ms=401.0,
            success=True,
        ),
    ]

    lead_a = LeadResult(
        lead_id="lead_research",
        domain="Competitive Research",
        aggregated_output=(
            "Competitive landscape: Salesforce dominates at 20% share with recent price increases. "
            "HubSpot targets SMBs via freemium. Pipedrive leads on UX. "
            "Key differentiators available: 40% lower TCO and 3-day onboarding vs 14-day industry average. "
            "Main gap: no native CPQ module."
        ),
        worker_results=worker_results_a,
        tokens_used=212,
        success=True,
        partial=False,
    )

    lead_b = LeadResult(
        lead_id="lead_analysis",
        domain="Market Analysis",
        aggregated_output=(
            "TAM $48B in 2024, SAM $12B in mid-market. Overall CRM market growing 14% YoY. "
            "AI-native CRM subsegment accelerating at 31% YoY — high-priority positioning opportunity."
        ),
        worker_results=worker_results_b,
        tokens_used=118,
        success=True,
        partial=False,
    )

    return FinalResult(
        goal=goal,
        final_output=(
            "Competitive analysis complete.\n\n"
            "The mid-market CRM space is growing at 14% YoY with an AI-native subsegment at 31%. "
            "Primary competitors (Salesforce, HubSpot, Pipedrive) all raised prices in Q1 2024, "
            "creating a pricing advantage window. Key differentiators to position: 40% lower TCO, "
            "3-day onboarding (vs 14-day average), and AI-native architecture. "
            "Critical gap to address before Q3: CPQ module — absence is cited in 38% of lost deals vs Salesforce."
        ),
        lead_results=[lead_a, lead_b],
        total_tokens_used=330,
        total_latency_ms=1240.0,
        success=True,
        warnings=[],
    )
