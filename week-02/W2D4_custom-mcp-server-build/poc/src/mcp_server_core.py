"""
W2D4 — Custom MCP Server Build
================================
Core MCP server: tool registry, dispatcher, and handler logic.
Runs in demo mode (no external services) or live mode.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory demo data — used when DEMO_MODE=true
# ---------------------------------------------------------------------------
DEMO_DOCUMENTS = [
    {"doc_id": "DOC-001", "title": "Q4 Revenue Report", "department": "finance",
     "snippet": "Total Q4 revenue reached $4.2M, up 18% YoY..."},
    {"doc_id": "DOC-002", "title": "Data Retention Policy", "department": "legal",
     "snippet": "All customer PII must be retained for no fewer than 7 years..."},
    {"doc_id": "DOC-003", "title": "Microservices Architecture Guide", "department": "engineering",
     "snippet": "Services must expose health endpoints at /health and /ready..."},
    {"doc_id": "DOC-004", "title": "Budget Forecast 2025", "department": "finance",
     "snippet": "Projected headcount growth of 12% across product and engineering..."},
]

DEMO_DEPARTMENTS = ["finance", "legal", "engineering"]

VALID_DEPARTMENTS = {"finance", "legal", "engineering"}


# ---------------------------------------------------------------------------
# Tool definitions — these are the JSON Schemas the LLM sees at inference time
# ---------------------------------------------------------------------------
def get_tool_definitions() -> list[dict]:
    """
    Return the list of tool definitions with full JSON Schemas.
    The 'description' field is the LLM's routing signal — write it for the model.
    """
    return [
        {
            "name": "search_documents",
            "description": (
                "Search the internal document repository by semantic query and department. "
                "Use this when the user needs to find documents, policies, or reports. "
                "Do NOT use this to retrieve a specific document by ID — use get_document instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "department": {
                        "type": "string",
                        "enum": ["finance", "legal", "engineering"],
                        "description": "Department scope for the search"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Number of results to return"
                    }
                },
                "required": ["query", "department"]
            }
        },
        {
            "name": "get_document",
            "description": (
                "Retrieve a single document by its exact document ID. "
                "Use this when you already know the doc_id. "
                "Do NOT use this for keyword search — use search_documents instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "pattern": "^DOC-[0-9]{3,}$",
                        "description": "Document ID in the format DOC-NNN"
                    }
                },
                "required": ["doc_id"]
            }
        },
        {
            "name": "list_departments",
            "description": (
                "Return the list of valid departments available for document search. "
                "Call this first if you are unsure which department to search."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_search_arguments(arguments: dict) -> str | None:
    """
    Validate arguments for search_documents beyond JSON Schema.
    Returns an error message string if invalid, None if valid.
    This is defense-in-depth — the schema layer already validates structure.
    """
    query = arguments.get("query", "").strip()
    if not query:
        return "Field 'query' must not be empty."
    if len(query) > 500:
        return f"Field 'query' exceeds maximum length of 500 characters (got {len(query)})."
    department = arguments.get("department", "")
    if department not in VALID_DEPARTMENTS:
        return f"Field 'department' must be one of {sorted(VALID_DEPARTMENTS)}, got '{department}'."
    top_k = arguments.get("top_k", 5)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        return "Field 'top_k' must be an integer between 1 and 20."
    return None


def validate_get_document_arguments(arguments: dict) -> str | None:
    """Validate arguments for get_document."""
    doc_id = arguments.get("doc_id", "").strip()
    if not doc_id:
        return "Field 'doc_id' must not be empty."
    if not doc_id.startswith("DOC-"):
        return "Field 'doc_id' must match format DOC-NNN (e.g., DOC-001)."
    return None


# ---------------------------------------------------------------------------
# Tool handlers — business logic (demo mode uses in-memory data)
# ---------------------------------------------------------------------------
async def handle_search_documents(arguments: dict, demo_mode: bool = True) -> dict:
    """
    Search documents by query and department.
    Demo mode: keyword match against in-memory DEMO_DOCUMENTS.
    Live mode: would call a real vector search backend.
    """
    error = validate_search_arguments(arguments)
    if error:
        # Return a structured error that the agent can act on
        return {"error": True, "code": -32602, "message": error}

    query = arguments["query"].lower()
    department = arguments["department"]
    top_k = arguments.get("top_k", 5)

    if demo_mode:
        # Simple keyword filter over demo data
        results = [
            doc for doc in DEMO_DOCUMENTS
            if doc["department"] == department
            and any(word in doc["title"].lower() or word in doc["snippet"].lower()
                    for word in query.split())
        ][:top_k]

        # Fallback: return all docs in department if no keyword match
        if not results:
            results = [d for d in DEMO_DOCUMENTS if d["department"] == department][:top_k]
    else:
        # Live mode: replace with actual vector search call
        # results = await vector_search_client.query(query, department, top_k)
        raise NotImplementedError("Live mode requires a vector search backend.")

    log_tool_call("search_documents", arguments, len(results))
    return {"results": results, "count": len(results), "department": department}


async def handle_get_document(arguments: dict, demo_mode: bool = True) -> dict:
    """Retrieve a single document by ID."""
    error = validate_get_document_arguments(arguments)
    if error:
        return {"error": True, "code": -32602, "message": error}

    doc_id = arguments["doc_id"]

    if demo_mode:
        doc = next((d for d in DEMO_DOCUMENTS if d["doc_id"] == doc_id), None)
    else:
        # Live mode: replace with actual document store lookup
        raise NotImplementedError("Live mode requires a document store backend.")

    log_tool_call("get_document", arguments, 1 if doc else 0)

    if doc is None:
        return {"error": True, "code": -32001, "message": f"Document '{doc_id}' not found."}
    return {"document": doc}


async def handle_list_departments(demo_mode: bool = True) -> dict:
    """Return valid departments."""
    log_tool_call("list_departments", {}, len(DEMO_DEPARTMENTS))
    return {"departments": DEMO_DEPARTMENTS}


# ---------------------------------------------------------------------------
# Dispatcher — routes tool_name to handler
# ---------------------------------------------------------------------------
async def dispatch_tool_call(
    tool_name: str,
    arguments: dict,
    demo_mode: bool = True
) -> dict:
    """
    Central dispatcher: maps tool names to handler functions.
    Returns a structured result or error dict.
    Never raises — always returns a serialisable dict.
    """
    try:
        if tool_name == "search_documents":
            return await handle_search_documents(arguments, demo_mode)
        elif tool_name == "get_document":
            return await handle_get_document(arguments, demo_mode)
        elif tool_name == "list_departments":
            return await handle_list_departments(demo_mode)
        else:
            # JSON-RPC -32601: method not found
            return {
                "error": True,
                "code": -32601,
                "message": f"Tool '{tool_name}' is not registered on this server."
            }
    except Exception as exc:
        logger.exception("Unhandled error in tool handler '%s'", tool_name)
        return {
            "error": True,
            "code": -32603,
            "message": f"Internal server error: {type(exc).__name__}"
        }


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
def log_tool_call(tool_name: str, arguments: dict, result_count: int) -> None:
    """
    Structured audit log entry for every tool call.
    In production, ship these to a centralised log aggregator.
    """
    safe_args = {k: v for k, v in arguments.items() if k not in {"password", "token", "secret"}}
    logger.info(
        "TOOL_CALL tool=%s args=%s result_count=%d ts=%s",
        tool_name,
        json.dumps(safe_args),
        result_count,
        datetime.now(timezone.utc).isoformat()
    )
