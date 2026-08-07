"""
W1D6 — State Graphs (LangGraph) — Core Logic
=============================================
Defines the shared state schema, node functions, router, and graph builder
for a document triage workflow with conditional edges and checkpointing.

All node functions are pure: they receive state, return a partial update dict,
and have zero side effects outside of that contract. This makes them unit-testable
in complete isolation from the graph runtime.
"""

from typing import TypedDict, Optional, List
from config import load_config

CONFIG = load_config()

# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
# A TypedDict is the single source of truth for all fields the graph uses.
# Every node reads from this schema and returns only the fields it changes.
# Unmodified fields are preserved by LangGraph's state merge logic.

class DocumentReviewState(TypedDict):
    document_text: str              # Raw input text — set at graph invocation
    clauses: List[str]              # Extracted clauses — set by ingest node
    risk_score: float               # 0.0–1.0 — set by classify node
    risk_label: str                 # "low_risk" | "high_risk" — set by classify node
    flags: List[str]                # Specific issues found — set by classify node
    human_approved: Optional[bool]  # None until human_approval node sets it
    summary: Optional[str]          # Final summary — set by finalise node
    retry_count: int                # Tracks retries to enforce recursion_limit guard
    error: Optional[str]            # Set by error_terminal node if processing fails


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------
# Each function signature is: (state: DocumentReviewState) -> dict
# Return only the fields this node changes — LangGraph merges the rest.

def ingest_document(state: DocumentReviewState) -> dict:
    """
    Extract clauses from raw document text.
    In live mode this would call an LLM or a rules-based extractor.
    The split-on-period heuristic here demonstrates the node contract
    without requiring an API key.
    """
    text = state["document_text"]
    # Naive clause extraction — replace with LLM call in production
    raw_clauses = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    clauses = raw_clauses[:8]  # cap at 8 clauses for demo manageability
    return {"clauses": clauses}


def classify_risk(state: DocumentReviewState) -> dict:
    """
    Score the document's risk level based on the presence of high-risk keywords.
    In live mode this would be an LLM classifier with a structured output schema.
    Returns risk_score (float), risk_label (str), and flags (list).
    """
    high_risk_keywords = [
        "indemnify", "liability", "warrant", "arbitration",
        "penalty", "termination for cause", "liquidated damages",
    ]
    text = state["document_text"].lower()
    found_flags = [kw for kw in high_risk_keywords if kw in text]

    # Score is proportional to the fraction of risk keywords present
    risk_score = round(len(found_flags) / len(high_risk_keywords), 2)
    risk_label = "high_risk" if risk_score >= CONFIG.risk_threshold else "low_risk"

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "flags": found_flags,
    }


def auto_process(state: DocumentReviewState) -> dict:
    """
    Low-risk path: generate a brief automated summary without human review.
    In live mode this calls an LLM summarisation endpoint.
    """
    clause_list = "\n".join(f"- {c}" for c in state["clauses"])
    summary = (
        f"[AUTO] Document processed without escalation. "
        f"Risk score: {state['risk_score']:.2f}. "
        f"Clauses reviewed: {len(state['clauses'])}. "
        f"No high-risk flags found."
    )
    return {"summary": summary, "human_approved": None}


def request_human_approval(state: DocumentReviewState) -> dict:
    """
    High-risk path: in a real graph this node would be preceded by
    interrupt_before, pausing execution until a human submits their decision.
    In demo mode we simulate an approval to keep the graph runnable offline.
    The human_approved field is injected via state update on graph resume.
    """
    # In production: graph.compile(interrupt_before=["human_approval"])
    # then graph.invoke(resume_state_update) with human_approved set externally.
    # In demo mode we set human_approved=True to complete the flow.
    print(
        "  [HUMAN-IN-THE-LOOP] High-risk document flagged for review.\n"
        f"  Flags: {state['flags']}\n"
        "  (Demo: auto-approving to complete workflow)"
    )
    return {"human_approved": True}


def finalise_document(state: DocumentReviewState) -> dict:
    """
    Terminal processing node: generate the final summary incorporating
    human approval status and all flags. Runs on both paths.
    """
    approval_str = (
        "Approved by human reviewer."
        if state.get("human_approved") is True
        else "Auto-processed (no human review required)."
    )
    flags_str = (
        f"Flags raised: {', '.join(state['flags'])}."
        if state["flags"]
        else "No flags raised."
    )
    summary = (
        f"Document review complete. Risk score: {state['risk_score']:.2f} "
        f"({state['risk_label']}). {flags_str} {approval_str} "
        f"Clauses reviewed: {len(state['clauses'])}."
    )
    return {"summary": summary}


def error_terminal(state: DocumentReviewState) -> dict:
    """
    Reached when retry_count exceeds MAX_RETRIES. Records the failure
    in state so the caller can inspect it without crashing the graph.
    """
    return {
        "error": (
            f"Processing failed after {state['retry_count']} retries. "
            "Manual intervention required."
        ),
        "summary": None,
    }


# ---------------------------------------------------------------------------
# Conditional Edge Router
# ---------------------------------------------------------------------------

def route_by_risk(state: DocumentReviewState) -> str:
    """
    Router function for the conditional edge after classify_risk.
    Returns a string key that maps to a target node in add_conditional_edges.
    Always returns a key present in the mapping — no KeyError in production.
    """
    label = state.get("risk_label", "low_risk")
    if label == "high_risk":
        return "high_risk"
    return "low_risk"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Assemble and compile the document review state graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (MemorySaver, SqliteSaver, etc.)
                      When None, the graph runs without persistence.

    Returns:
        A compiled LangGraph CompiledGraph ready for .invoke() or .stream().
    """
    # Import inside the function so missing langgraph doesn't break demo mode
    try:
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(DocumentReviewState)

        # Register all nodes
        builder.add_node("ingest", ingest_document)
        builder.add_node("classify", classify_risk)
        builder.add_node("auto_process", auto_process)
        builder.add_node("human_approval", request_human_approval)
        builder.add_node("finalise", finalise_document)
        builder.add_node("error_terminal", error_terminal)

        # Static edges: entry point and convergence to finalise
        builder.add_edge(START, "ingest")
        builder.add_edge("ingest", "classify")
        builder.add_edge("auto_process", "finalise")
        builder.add_edge("human_approval", "finalise")
        builder.add_edge("finalise", END)
        builder.add_edge("error_terminal", END)

        # Conditional edge: route after classify based on risk_label
        builder.add_conditional_edges(
            "classify",
            route_by_risk,
            {
                "low_risk": "auto_process",
                "high_risk": "human_approval",
            },
        )

        return builder.compile(checkpointer=checkpointer)

    except ImportError:
        # langgraph not installed — caller will fall back to demo mode
        return None


# ---------------------------------------------------------------------------
# Demo-mode simulation (no langgraph required)
# ---------------------------------------------------------------------------

def run_demo_graph(document_text: str) -> DocumentReviewState:
    """
    Simulate the graph execution without the langgraph library.
    Executes node functions directly in topological order so the PoC
    produces meaningful output even when langgraph is not installed.
    """
    # Initialise state with all required fields and safe defaults
    state: DocumentReviewState = {
        "document_text": document_text,
        "clauses": [],
        "risk_score": 0.0,
        "risk_label": "low_risk",
        "flags": [],
        "human_approved": None,
        "summary": None,
        "retry_count": 0,
        "error": None,
    }

    def apply(update: dict) -> None:
        state.update(update)

    print("  Node: ingest      ", end="")
    apply(ingest_document(state))
    print(f"-> clauses extracted: {len(state['clauses'])}")

    print("  Node: classify    ", end="")
    apply(classify_risk(state))
    print(f"-> risk_score={state['risk_score']:.2f}, label={state['risk_label']}")

    route = route_by_risk(state)
    print(f"  Router            -> {route} path selected")

    if route == "high_risk":
        print("  Node: human_approval")
        apply(request_human_approval(state))
    else:
        print("  Node: auto_process -> generating automated summary")
        apply(auto_process(state))

    print("  Node: finalise    ", end="")
    apply(finalise_document(state))
    print("-> summary generated")

    return state
