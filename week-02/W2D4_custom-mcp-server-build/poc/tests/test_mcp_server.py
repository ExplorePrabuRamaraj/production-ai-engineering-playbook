"""
Unit tests for W2D4 — Custom MCP Server Build
Run: pytest tests/ -v
All tests run offline — no external services or API keys required.
"""
import asyncio
import json
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_server_core import (
    dispatch_tool_call,
    get_tool_definitions,
    handle_search_documents,
    handle_get_document,
    handle_list_departments,
    validate_search_arguments,
    validate_get_document_arguments,
    VALID_DEPARTMENTS,
)
from config import load_config


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def run(coro):
    """Run an async coroutine synchronously in tests."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Tool definition tests
# ---------------------------------------------------------------------------
class TestToolDefinitions:

    def test_tool_definitions_returns_three_tools(self):
        tools = get_tool_definitions()
        assert len(tools) == 3

    def test_all_tools_have_required_fields(self):
        tools = get_tool_definitions()
        for tool in tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "inputSchema" in tool, f"Tool missing 'inputSchema': {tool}"

    def test_search_documents_requires_query_and_department(self):
        tools = get_tool_definitions()
        search_tool = next(t for t in tools if t["name"] == "search_documents")
        required = search_tool["inputSchema"]["required"]
        assert "query" in required
        assert "department" in required

    def test_department_field_has_enum_constraint(self):
        tools = get_tool_definitions()
        search_tool = next(t for t in tools if t["name"] == "search_documents")
        dept_schema = search_tool["inputSchema"]["properties"]["department"]
        assert "enum" in dept_schema
        assert set(dept_schema["enum"]) == {"finance", "legal", "engineering"}

    def test_get_document_requires_doc_id(self):
        tools = get_tool_definitions()
        get_doc_tool = next(t for t in tools if t["name"] == "get_document")
        assert "doc_id" in get_doc_tool["inputSchema"]["required"]


# ---------------------------------------------------------------------------
# 2. Input validation tests
# ---------------------------------------------------------------------------
class TestValidation:

    @pytest.mark.parametrize("query,department,top_k,expected_error", [
        ("revenue report", "finance", 5, None),          # valid
        ("", "finance", 5, "must not be empty"),          # empty query
        ("q" * 501, "finance", 5, "exceeds maximum"),     # query too long
        ("report", "marketing", 5, "must be one of"),     # invalid department
        ("report", "finance", 0, "between 1 and 20"),     # top_k too low
        ("report", "finance", 21, "between 1 and 20"),    # top_k too high
    ])
    def test_search_validation_parametrized(self, query, department, top_k, expected_error):
        args = {"query": query, "department": department, "top_k": top_k}
        error = validate_search_arguments(args)
        if expected_error is None:
            assert error is None, f"Expected no error but got: {error}"
        else:
            assert error is not None, "Expected an error but got None"
            assert expected_error in error, f"Expected '{expected_error}' in error: {error}"

    def test_get_document_valid_id(self):
        assert validate_get_document_arguments({"doc_id": "DOC-001"}) is None

    def test_get_document_empty_id(self):
        error = validate_get_document_arguments({"doc_id": ""})
        assert error is not None
        assert "must not be empty" in error

    def test_get_document_wrong_format(self):
        error = validate_get_document_arguments({"doc_id": "123"})
        assert error is not None
        assert "DOC-" in error


# ---------------------------------------------------------------------------
# 3. Handler tests — demo mode (offline)
# ---------------------------------------------------------------------------
class TestHandlers:

    def test_search_returns_finance_results(self):
        result = run(handle_search_documents(
            {"query": "revenue", "department": "finance", "top_k": 5},
            demo_mode=True
        ))
        assert "results" in result
        assert result["count"] >= 1
        for doc in result["results"]:
            assert doc["department"] == "finance"

    def test_search_missing_department_returns_error(self):
        result = run(handle_search_documents(
            {"query": "revenue"},
            demo_mode=True
        ))
        assert result.get("error") is True
        assert result["code"] == -32602

    def test_get_document_known_id(self):
        result = run(handle_get_document({"doc_id": "DOC-001"}, demo_mode=True))
        assert "document" in result
        assert result["document"]["doc_id"] == "DOC-001"

    def test_get_document_unknown_id_returns_not_found(self):
        result = run(handle_get_document({"doc_id": "DOC-999"}, demo_mode=True))
        assert result.get("error") is True
        assert result["code"] == -32001
        assert "not found" in result["message"].lower()

    def test_list_departments_returns_all(self):
        result = run(handle_list_departments(demo_mode=True))
        assert "departments" in result
        assert set(result["departments"]) == {"finance", "legal", "engineering"}


# ---------------------------------------------------------------------------
# 4. Dispatcher tests
# ---------------------------------------------------------------------------
class TestDispatcher:

    def test_dispatch_known_tool_succeeds(self):
        result = run(dispatch_tool_call(
            "list_departments", {}, demo_mode=True
        ))
        assert "departments" in result
        assert "error" not in result

    def test_dispatch_unknown_tool_returns_method_not_found(self):
        result = run(dispatch_tool_call(
            "nonexistent_tool", {}, demo_mode=True
        ))
        assert result["error"] is True
        assert result["code"] == -32601
        assert "not registered" in result["message"]

    def test_dispatch_never_raises(self):
        # Even with a completely malformed call, dispatcher must return a dict
        try:
            result = run(dispatch_tool_call("search_documents", {}, demo_mode=True))
            assert isinstance(result, dict)
        except Exception as exc:
            pytest.fail(f"Dispatcher raised unexpectedly: {exc}")

    @pytest.mark.parametrize("tool_name", ["search_documents", "get_document", "list_departments"])
    def test_all_registered_tools_dispatchable(self, tool_name):
        # Each tool should return a dict (error or success) without raising
        args = {
            "search_documents": {"query": "test", "department": "engineering"},
            "get_document": {"doc_id": "DOC-001"},
            "list_departments": {},
        }[tool_name]
        result = run(dispatch_tool_call(tool_name, args, demo_mode=True))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 5. Config tests
# ---------------------------------------------------------------------------
class TestConfig:

    def test_demo_mode_on_when_no_bearer_token(self, monkeypatch):
        monkeypatch.delenv("BEARER_TOKEN", raising=False)
        monkeypatch.setenv("DEMO_MODE", "false")
        config = load_config()
        # No token → demo mode must be True regardless of DEMO_MODE env var
        assert config.demo_mode is True

    def test_demo_mode_explicit_true(self, monkeypatch):
        monkeypatch.setenv("DEMO_MODE", "true")
        monkeypatch.setenv("BEARER_TOKEN", "some-token")
        config = load_config()
        assert config.demo_mode is True

    def test_max_results_default(self, monkeypatch):
        monkeypatch.delenv("MAX_RESULTS", raising=False)
        config = load_config()
        assert config.max_results == 20
