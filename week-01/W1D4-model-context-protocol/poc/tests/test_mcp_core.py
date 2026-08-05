"""
W1D4 — Model Context Protocol (MCP) Intro — Unit Tests
=======================================================
Run: pytest tests/ -v

All tests run fully offline — no API key, no external services required.
Tests cover: demo mode, core MCP protocol behaviour, error handling, and sample files.
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_core import (
    MCPServer,
    MCPClient,
    ToolDefinition,
    ResourceDefinition,
    ToolCallResult,
    run_mcp_session,
)
from main import run_demo, run_live

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server() -> MCPServer:
    """Fresh MCPServer instance for each test."""
    return MCPServer(server_name="test-crm-server", protocol_version="2024-11-05")


@pytest.fixture
def initialised_client(server) -> MCPClient:
    """MCPClient that has already completed the initialize handshake."""
    client = MCPClient(server)
    client.initialize()
    return client


@pytest.fixture
def valid_lookup_args() -> dict:
    return {"customer_id": "CUST-1001", "account_region": "us-east-1"}


# ---------------------------------------------------------------------------
# TestDemoMode — offline demo mode, no API key required
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Demo mode must run completely offline and produce the expected output shape."""

    def test_demo_returns_required_keys(self):
        """run_demo() output must contain all four required top-level keys."""
        result = run_demo()
        required = {"server_name", "tools_discovered", "resources_discovered", "tool_calls"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_demo_discovers_tools(self):
        """Demo session must discover at least one tool."""
        result = run_demo()
        assert len(result["tools_discovered"]) >= 1

    def test_demo_discovers_resources(self):
        """Demo session must discover at least one resource."""
        result = run_demo()
        assert len(result["resources_discovered"]) >= 1

    def test_demo_executes_multiple_tool_calls(self):
        """Demo session must execute more than one tool call to show diverse paths."""
        result = run_demo()
        assert len(result["tool_calls"]) >= 3

    def test_demo_includes_error_call(self):
        """Demo session must include at least one isError:true call to show error handling."""
        result = run_demo()
        error_calls = [c for c in result["tool_calls"] if c["is_error"]]
        assert len(error_calls) >= 1, "Demo should demonstrate at least one tool error path"

    def test_demo_includes_success_call(self):
        """Demo session must include at least one successful tool call."""
        result = run_demo()
        success_calls = [c for c in result["tool_calls"] if not c["is_error"]]
        assert len(success_calls) >= 1


# ---------------------------------------------------------------------------
# TestCoreConcept — MCPServer and MCPClient protocol behaviour
# ---------------------------------------------------------------------------

class TestCoreConcept:
    """Tests for the MCP protocol primitives: init, discovery, tool calls, resources."""

    def test_server_initialize_returns_protocol_version(self, server):
        """Server must echo back the protocol version in the init response."""
        response = server.initialize()
        assert response["protocolVersion"] == "2024-11-05"

    def test_server_initialize_returns_server_info(self, server):
        """Server init response must include serverInfo with a name."""
        response = server.initialize()
        assert "serverInfo" in response
        assert response["serverInfo"]["name"] == "test-crm-server"

    def test_server_lists_tools(self, server):
        """Server must expose at least one tool."""
        tools = server.list_tools()
        assert len(tools) >= 1
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_lookup_account_tool_is_registered(self, server):
        """The lookup_account tool must be present with correct schema."""
        tools = {t.name: t for t in server.list_tools()}
        assert "lookup_account" in tools
        required = tools["lookup_account"].input_schema.get("required", [])
        assert "customer_id" in required
        assert "account_region" in required

    def test_server_lists_resources(self, server):
        """Server must expose at least one resource."""
        resources = server.list_resources()
        assert len(resources) >= 1
        assert all(isinstance(r, ResourceDefinition) for r in resources)

    def test_tool_call_success(self, server, valid_lookup_args):
        """Valid tool call must return isError=False and parseable JSON content."""
        result = server.call_tool("lookup_account", valid_lookup_args)
        assert not result.is_error
        data = json.loads(result.content)
        assert data["customer_id"] == "CUST-1001"
        assert "status" in data
        assert "tier" in data

    def test_tool_call_unknown_account_returns_error(self, server):
        """Calling lookup_account with a non-existent ID must return isError=True."""
        result = server.call_tool(
            "lookup_account",
            {"customer_id": "CUST-9999", "account_region": "us-east-1"},
        )
        assert result.is_error
        assert "not found" in result.content.lower()

    def test_tool_call_missing_required_param_returns_error(self, server):
        """Missing required parameter must return isError=True with descriptive message."""
        result = server.call_tool("lookup_account", {"customer_id": "CUST-1001"})
        assert result.is_error
        assert "account_region" in result.content

    def test_tool_call_unknown_tool_returns_error(self, server):
        """Calling a tool that does not exist must return isError=True."""
        result = server.call_tool("nonexistent_tool", {})
        assert result.is_error
        assert "unknown tool" in result.content.lower()

    def test_tool_call_records_latency(self, server, valid_lookup_args):
        """Tool call result must record a non-negative latency_ms value."""
        result = server.call_tool("lookup_account", valid_lookup_args)
        assert result.latency_ms >= 0

    def test_update_status_modifies_account(self, server):
        """update_account_status must change the account status persistently."""
        # First confirm starting status
        before = server.call_tool(
            "lookup_account",
            {"customer_id": "CUST-2042", "account_region": "us-east-1"},
        )
        before_data = json.loads(before.content)
        assert before_data["status"] == "suspended"

        # Update the status
        update = server.call_tool(
            "update_account_status",
            {"customer_id": "CUST-2042", "new_status": "active", "reason": "Test"},
        )
        assert not update.is_error

        # Confirm it changed
        after = server.call_tool(
            "lookup_account",
            {"customer_id": "CUST-2042", "account_region": "us-east-1"},
        )
        after_data = json.loads(after.content)
        assert after_data["status"] == "active"

    @pytest.mark.parametrize("customer_id,region,expect_error", [
        ("CUST-1001", "us-east-1", False),
        ("CUST-2042", "eu-west-1", False),
        ("CUST-3300", "ap-south-1", False),
        ("CUST-0000", "us-east-1", True),   # non-existent account
        ("CUST-1001", "invalid-region", True),  # invalid region enum
    ])
    def test_lookup_account_parametrised(self, server, customer_id, region, expect_error):
        """lookup_account must handle valid and invalid inputs predictably."""
        result = server.call_tool(
            "lookup_account",
            {"customer_id": customer_id, "account_region": region},
        )
        assert result.is_error == expect_error

    def test_read_resource_returns_json(self, server):
        """Reading the regions resource must return valid JSON."""
        content = server.read_resource("config://crm/regions")
        regions = json.loads(content)
        assert isinstance(regions, list)
        assert "us-east-1" in regions

    def test_read_unknown_resource_returns_message(self, server):
        """Reading an unknown resource URI must return a not-found message, not raise."""
        content = server.read_resource("config://nonexistent")
        assert "not found" in content.lower()


# ---------------------------------------------------------------------------
# TestLiveMode — MCPClient and session lifecycle (no external calls)
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Tests for MCPClient protocol behaviour and run_mcp_session orchestration."""

    def test_client_requires_initialize_before_list_tools(self, server):
        """MCPClient must raise RuntimeError if list_tools is called before initialize."""
        client = MCPClient(server)
        with pytest.raises(RuntimeError, match="initialize"):
            client.list_tools()

    def test_client_initialize_sets_initialised_flag(self, server):
        """After initialize(), the client must accept subsequent requests."""
        client = MCPClient(server)
        client.initialize()
        # Should not raise
        tools = client.list_tools()
        assert len(tools) >= 1

    def test_client_list_tools_mirrors_server(self, initialised_client, server):
        """Client tool list must match the server's registered tools."""
        client_tools = {t.name for t in initialised_client.list_tools()}
        server_tools = {t.name for t in server.list_tools()}
        assert client_tools == server_tools

    def test_client_call_tool_success(self, initialised_client, valid_lookup_args):
        """Client call_tool must return a ToolCallResult with is_error=False on success."""
        result = initialised_client.call_tool("lookup_account", valid_lookup_args)
        assert isinstance(result, ToolCallResult)
        assert not result.is_error

    def test_run_mcp_session_returns_session_result(self, server):
        """run_mcp_session must return a SessionResult with all expected fields."""
        result = run_mcp_session(
            server,
            [{"name": "lookup_account", "arguments": {"customer_id": "CUST-1001", "account_region": "us-east-1"}}],
        )
        assert result.server_name == "test-crm-server"
        assert "lookup_account" in result.tools_discovered
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "lookup_account"

    def test_run_mcp_session_logs_errors(self, server):
        """run_mcp_session must log is_error=True calls without raising exceptions."""
        result = run_mcp_session(
            server,
            [{"name": "lookup_account", "arguments": {"customer_id": "CUST-9999", "account_region": "us-east-1"}}],
        )
        assert result.tool_calls[0]["is_error"] is True

    def test_run_live_returns_required_shape(self):
        """run_live() must return the same output shape as run_demo()."""
        result = run_live()
        required = {"server_name", "tools_discovered", "resources_discovered", "tool_calls"}
        assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# TestSampleFiles — validate sample_input.json and sample_output.json
# ---------------------------------------------------------------------------

class TestSampleFiles:
    """Validates that sample JSON files exist and match the expected schema."""

    def test_sample_input_is_valid_json(self):
        """sample_input.json must be parseable and a dict."""
        path = Path(__file__).parent.parent / "sample_input.json"
        assert path.exists(), "sample_input.json is missing"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_sample_input_has_tool_requests(self):
        """sample_input.json must contain a 'tool_requests' list."""
        path = Path(__file__).parent.parent / "sample_input.json"
        data = json.loads(path.read_text())
        assert "tool_requests" in data
        assert isinstance(data["tool_requests"], list)
        assert len(data["tool_requests"]) >= 1

    def test_sample_output_is_valid_json(self):
        """sample_output.json must be parseable and a dict."""
        path = Path(__file__).parent.parent / "sample_output.json"
        assert path.exists(), "sample_output.json is missing"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_sample_output_has_required_keys(self):
        """sample_output.json must contain the expected top-level keys."""
        path = Path(__file__).parent.parent / "sample_output.json"
        data = json.loads(path.read_text())
        required = {"server_name", "tools_discovered", "tool_calls"}
        assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"
