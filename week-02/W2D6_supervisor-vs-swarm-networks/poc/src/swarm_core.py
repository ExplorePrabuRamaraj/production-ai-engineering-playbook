"""
W2D6 -- Supervisor vs. Swarm Networks -- Core Logic
====================================================
Implements two orchestration topologies:
  - SupervisorNetwork: central coordinator delegates to specialists
  - SwarmNetwork: peer-to-peer routing between agents

Both classes are designed to run offline (demo mode) as well as with a
real LLM backend. No side effects -- all state is contained in method
return values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


# ---------------------------------------------------------------------------
# Data types shared across both topologies
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result returned by any agent after handling a subtask."""
    agent_name: str
    subtask: str
    output: str
    success: bool
    latency_ms: float
    tokens_used: int = 0


@dataclass
class WorkflowResult:
    """Final result returned by either SupervisorNetwork or SwarmNetwork."""
    topology: str                        # "supervisor" or "swarm"
    input_task: str
    subtask_results: list[AgentResult]
    final_output: str
    total_latency_ms: float
    total_tokens: int
    routing_trace: list[str]             # human-readable routing log


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Base class for a specialist agent.

    Subclass this and override `capability` (a plain-English description of
    what tasks this agent handles) and `handle()` (the actual task logic).
    The routing functions in SupervisorNetwork and SwarmNetwork use
    `capability` to decide which agent should receive a given subtask.
    """

    name: str = "base-agent"
    capability: str = "handles generic tasks"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        """Process a subtask and return a structured result."""
        raise NotImplementedError("Subclass must implement handle()")

    def can_handle(self, subtask: str) -> bool:
        """
        Simple keyword-based capability check.
        In production, replace with a vector similarity check or a fast
        classifier LLM call for better precision.
        Matches any capability keyword of length >= 4 that appears in the subtask.
        """
        keywords = self.capability.lower().split()
        subtask_lower = subtask.lower()
        return any(kw in subtask_lower for kw in keywords if len(kw) >= 4)

    def match_score(self, subtask: str) -> int:
        """Count how many capability keywords appear in the subtask.
        Used by SupervisorNetwork to pick the best-matching agent rather than
        the first agent that passes the binary can_handle check."""
        keywords = self.capability.lower().split()
        subtask_lower = subtask.lower()
        return sum(1 for kw in keywords if len(kw) >= 4 and kw in subtask_lower)


# ---------------------------------------------------------------------------
# Concrete specialist agents (demo implementations)
# ---------------------------------------------------------------------------

class RetrievalAgent(Agent):
    name = "retrieval-agent"
    # Include both "retrieve" and "retrieval" so the keyword matcher hits either form
    capability = "retrieve retrieval search fetch data documents lookup"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        start = time.monotonic()
        output = f"[RetrievalAgent] Retrieved 3 relevant documents for: '{subtask}'"
        latency = (time.monotonic() - start) * 1000
        return AgentResult(
            agent_name=self.name,
            subtask=subtask,
            output=output,
            success=True,
            latency_ms=round(latency, 2),
            tokens_used=45,
        )


class AnalysisAgent(Agent):
    name = "analysis-agent"
    # Deliberately does NOT include "generate" or "validate" to avoid false matches
    capability = "analysis analyse analyze summarise summarize classify sentiment"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        start = time.monotonic()
        output = f"[AnalysisAgent] Analysis complete for: '{subtask}' -- sentiment: positive, confidence: 0.87"
        latency = (time.monotonic() - start) * 1000
        return AgentResult(
            agent_name=self.name,
            subtask=subtask,
            output=output,
            success=True,
            latency_ms=round(latency, 2),
            tokens_used=62,
        )


class GenerationAgent(Agent):
    name = "generation-agent"
    # "generate" and "generation" both match tasks prefixed by the Supervisor decomposer
    capability = "generate generation write draft compose response reply answer"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        start = time.monotonic()
        output = (
            f"[GenerationAgent] Draft response for: '{subtask}' -- "
            "Thank you for reaching out. We have reviewed your request and are happy to assist."
        )
        latency = (time.monotonic() - start) * 1000
        return AgentResult(
            agent_name=self.name,
            subtask=subtask,
            output=output,
            success=True,
            latency_ms=round(latency, 2),
            tokens_used=78,
        )


class ValidationAgent(Agent):
    name = "validation-agent"
    # "validate" matches tasks prefixed with "validate output for:" by the Supervisor
    capability = "validate validation verify check compliance policy rules safety"

    def handle(self, subtask: str, demo_mode: bool = True) -> AgentResult:
        start = time.monotonic()
        output = f"[ValidationAgent] Validation passed for: '{subtask}' -- no policy violations detected"
        latency = (time.monotonic() - start) * 1000
        return AgentResult(
            agent_name=self.name,
            subtask=subtask,
            output=output,
            success=True,
            latency_ms=round(latency, 2),
            tokens_used=38,
        )


# Default agent pool used by both network types
DEFAULT_AGENTS: list[Agent] = [
    RetrievalAgent(),
    AnalysisAgent(),
    GenerationAgent(),
    ValidationAgent(),
]


# ---------------------------------------------------------------------------
# Supervisor Network
# ---------------------------------------------------------------------------

class SupervisorNetwork:
    """
    Hub-and-spoke topology: the Supervisor decomposes a task into subtasks,
    dispatches each to the best-matching specialist, then aggregates results.

    Key properties:
    - All task state is held in the WorkflowResult object (easy to persist externally)
    - Subtask dispatch is sequential here for simplicity; in production use
      asyncio.gather() or a thread pool for parallel execution
    - Failed subtasks return a fallback result rather than aborting the workflow
    """

    def __init__(self, agents: list[Agent] | None = None, max_hops: int = 5):
        self.agents = agents or DEFAULT_AGENTS
        self.max_hops = max_hops

    def decompose(self, task: str) -> list[str]:
        """
        Break a high-level task into subtasks.
        In production this is an LLM call that returns a structured task plan.
        Here we use a rule-based decomposer to keep the demo fully offline.

        Each subtask is expressed as a self-contained instruction so the
        best-matching specialist can be identified from the subtask text alone
        (not from the full original task string, which would confuse scoring).
        """
        subtasks = []
        task_lower = task.lower()
        if any(w in task_lower for w in ["find", "search", "fetch", "retrieve", "lookup"]):
            subtasks.append("retrieve relevant documents and data")
        if any(w in task_lower for w in ["analyse", "analyze", "summarise", "summarize", "review", "sentiment"]):
            subtasks.append("analyse content and classify sentiment")
        if any(w in task_lower for w in ["write", "draft", "generate", "compose", "respond", "response"]):
            subtasks.append("generate a written response")
        if any(w in task_lower for w in ["check", "validate", "verify", "compliance", "policy"]):
            subtasks.append("validate output for compliance")
        # Ensure at least one subtask for unrecognised inputs
        if not subtasks:
            subtasks.append("analyse and generate a response")
        return subtasks

    def _find_best_agent(self, subtask: str) -> Agent:
        """Return the agent with the highest keyword match score for the subtask.
        Using match_score (count of matching keywords) rather than first-match
        ensures the most specialised agent wins when multiple agents share some
        vocabulary (e.g. both GenerationAgent and AnalysisAgent match 'response')."""
        scored = [(agent.match_score(subtask), agent) for agent in self.agents]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_agent = scored[0]
        if best_score > 0:
            return best_agent
        # Fallback: return the first agent (GenerationAgent catch-all)
        return self.agents[0]

    def run(self, task: str, demo_mode: bool = True) -> WorkflowResult:
        """
        Execute the full Supervisor workflow:
        1. Decompose task into subtasks
        2. Dispatch each subtask to the best specialist
        3. Aggregate all results into a final output
        """
        start = time.monotonic()
        routing_trace: list[str] = []
        subtask_results: list[AgentResult] = []

        subtasks = self.decompose(task)
        routing_trace.append(
            f"[Supervisor] Decomposed into {len(subtasks)} subtask(s): {subtasks}"
        )

        for subtask in subtasks:
            agent = self._find_best_agent(subtask)
            routing_trace.append(
                f"[Supervisor] Dispatching '{subtask}' -> {agent.name}"
            )
            result = agent.handle(subtask, demo_mode=demo_mode)
            subtask_results.append(result)
            routing_trace.append(
                f"[Supervisor] Received result from {agent.name} "
                f"(success={result.success}, latency={result.latency_ms}ms)"
            )

        # Aggregate: join all specialist outputs
        final_output = " | ".join(r.output for r in subtask_results)
        total_latency = (time.monotonic() - start) * 1000
        total_tokens = sum(r.tokens_used for r in subtask_results)

        routing_trace.append(
            f"[Supervisor] Aggregation complete. "
            f"Total latency: {total_latency:.1f}ms, tokens: {total_tokens}"
        )

        return WorkflowResult(
            topology="supervisor",
            input_task=task,
            subtask_results=subtask_results,
            final_output=final_output,
            total_latency_ms=round(total_latency, 2),
            total_tokens=total_tokens,
            routing_trace=routing_trace,
        )


# ---------------------------------------------------------------------------
# Swarm Network
# ---------------------------------------------------------------------------

class SwarmNetwork:
    """
    Mesh topology: the initial message is sent to the swarm and agents
    route it peer-to-peer until a capable agent handles it.

    Key properties:
    - No central coordinator -- routing is decentralised
    - Routing history prevents cycles (each agent ID appears at most once)
    - max_hops prevents infinite forwarding chains
    - Unroutable messages are captured in the dead letter queue
    """

    def __init__(self, agents: list[Agent] | None = None, max_hops: int = 5):
        self.agents = agents or DEFAULT_AGENTS
        self.max_hops = max_hops
        self.dead_letter_queue: list[dict] = []   # captured unroutable messages

    def _route_message(
        self,
        subtask: str,
        routing_history: list[str],
        trace: list[str],
    ) -> AgentResult | None:
        """
        Attempt to route a single subtask through the swarm.
        Returns AgentResult if handled, None if unroutable (exceeds max_hops
        or no agent claims capability).
        """
        if len(routing_history) >= self.max_hops:
            trace.append(
                f"[Swarm] Max hops ({self.max_hops}) exceeded for '{subtask}'. "
                "Sending to dead letter queue."
            )
            return None

        for agent in self.agents:
            # Skip agents already in the routing history (cycle prevention)
            if agent.name in routing_history:
                continue
            if agent.can_handle(subtask):
                routing_history.append(agent.name)
                trace.append(
                    f"[Swarm] '{subtask}' -> {agent.name} "
                    f"(hop {len(routing_history)}, history: {routing_history})"
                )
                return agent.handle(subtask, demo_mode=True)

        trace.append(
            f"[Swarm] No capable agent found for '{subtask}' after "
            f"{len(routing_history)} hops. Sending to dead letter queue."
        )
        return None

    def run(self, task: str, demo_mode: bool = True) -> WorkflowResult:
        """
        Execute the full Swarm workflow.
        Each word-level subtask (split by semicolons in the input, or the
        whole task as a single message) is routed independently through the
        swarm.
        """
        start = time.monotonic()
        routing_trace: list[str] = []
        subtask_results: list[AgentResult] = []

        # Split on semicolons for multi-part tasks; otherwise treat as one task
        subtasks = [s.strip() for s in task.split(";") if s.strip()] or [task]
        routing_trace.append(
            f"[Swarm] Received {len(subtasks)} message(s) for routing"
        )

        for subtask in subtasks:
            history: list[str] = []
            result = self._route_message(subtask, history, routing_trace)
            if result is not None:
                subtask_results.append(result)
            else:
                # Capture in dead letter queue
                self.dead_letter_queue.append({
                    "subtask": subtask,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "routing_history": history,
                })

        final_output = (
            " | ".join(r.output for r in subtask_results)
            if subtask_results
            else "[Swarm] No results -- all messages unroutable"
        )
        total_latency = (time.monotonic() - start) * 1000
        total_tokens = sum(r.tokens_used for r in subtask_results)

        routing_trace.append(
            f"[Swarm] Complete. "
            f"Handled: {len(subtask_results)}, "
            f"DLQ: {len(self.dead_letter_queue)}, "
            f"Total latency: {total_latency:.1f}ms, tokens: {total_tokens}"
        )

        return WorkflowResult(
            topology="swarm",
            input_task=task,
            subtask_results=subtask_results,
            final_output=final_output,
            total_latency_ms=round(total_latency, 2),
            total_tokens=total_tokens,
            routing_trace=routing_trace,
        )
