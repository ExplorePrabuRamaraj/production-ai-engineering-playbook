"""
W1D4 — Model Context Protocol (MCP) Core
=========================================
Implements a self-contained MCP session simulation:
  - MCPServer: exposes Resources, Tools, and Prompts via the three MCP primitives
  - MCPClient: discovers and calls server capabilities (mirrors real MCP client API)
  - MCPSession: orchestrates the full lifecycle (init → discover → call → shutdown)

This module uses no external MCP SDK so it runs fully offline.
The interface mirrors the real MCP Python SDK (modelcontextprotocol/python-sdk)
so the patterns here transfer directly to production use.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Protocol types — mirror the MCP spec data shapes
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Mirrors mcp.types.Tool — the schema a server registers for one tool."""
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ResourceDefinition:
    """Mirrors mcp.types.Resource — a read-only context item the server exposes."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"


@dataclass
class ToolCallResult:
    """Mirrors the mcp.types result shape returned by tools/call."""
    content: str
    is_error: bool = False
    latency_ms: float = 0.0


@dataclass
class MCPCapabilities:
    """Server capabilities declared during initialisation."""
    tools: bool = True
    resources: bool = True
    prompts: bool = False
    sampling: bool = False


# ---------------------------------------------------------------------------
# MCPServer — simulates a real MCP server (e.g., a CRM integration)
# ---------------------------------------------------------------------------

class MCPServer:
    """
    In-process MCP server simulation.

    In production this would be a subprocess communicating over stdio or
    an HTTP service communicating over SSE.  The public interface — list_tools(),
    list_resources(), call_tool(), read_resource() — matches the real SDK exactly,
    so switching to a real transport requires only changing the transport layer,
    not this business logic.
    """

    def __init__(self, server_name: str, protocol_version: str = "2024-11-05") -> None:
        self.server_name = server_name
        self.protocol_version = protocol_version
        self.capabilities = MCPCapabilities()
        self._tools: dict[str, ToolDefinition] = {}
        self._resources: dict[str, ResourceDefinition] = {}
        # Pre-populated demo data — in production these would come from a real API/DB
        self._account_db: dict[str, dict] = {
            "CUST-1001": {"status": "active", "tier": "gold", "balance_usd": 0.00},
            "CUST-2042": {"status": "suspended", "tier": "silver", "balance_usd": 145.50},
            "CUST-3300": {"status": "active", "tier": "platinum", "balance_usd": 0.00},
        }
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the tools and resources this demo server exposes."""
        # Tool: lookup customer account
        self._tools["lookup_account"] = ToolDefinition(
            name="lookup_account",
            description=(
                "Look up a customer account by ID and region. "
                "Returns account status and billing tier. "
                "Use this tool when the user asks about account standing, "
                "not for transaction history (use get_transactions for that)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer account ID, format CUST-NNNN",
                    },
                    "account_region": {
                        "type": "string",
                        "enum": ["us-east-1", "eu-west-1", "ap-south-1"],
                        "description": "The region the account is registered in",
                    },
                },
                "required": ["customer_id", "account_region"],
            },
        )
        # Tool: update account status (write operation)
        self._tools["update_account_status"] = ToolDefinition(
            name="update_account_status",
            description=(
                "Update the status of a customer account. "
                "Valid statuses: active, suspended, closed. "
                "Requires write permission scope."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "new_status": {
                        "type": "string",
                        "enum": ["active", "suspended", "closed"],
                    },
                    "reason": {"type": "string", "description": "Reason for status change"},
                },
                "required": ["customer_id", "new_status", "reason"],
            },
        )
        # Resource: system configuration (read-only context)
        self._resources["config://crm/regions"] = ResourceDefinition(
            uri="config://crm/regions",
            name="Supported CRM Regions",
            description="List of valid account regions for this CRM instance",
            mime_type="application/json",
        )

    # --- MCP protocol methods (called by MCPClient) ---

    def initialize(self) -> dict[str, Any]:
        """
        Respond to the MCP initialize request.
        Returns server info and declared capabilities.
        """
        return {
            "protocolVersion": self.protocol_version,
            "serverInfo": {"name": self.server_name, "version": "1.0.0"},
            "capabilities": {
                "tools": {} if self.capabilities.tools else None,
                "resources": {} if self.capabilities.resources else None,
            },
        }

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tools. Called by client during discovery."""
        return list(self._tools.values())

    def list_resources(self) -> list[ResourceDefinition]:
        """Return all registered resources. Called by client during discovery."""
        return list(self._resources.values())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """
        Execute a tool by name with the given arguments.

        Validates inputs against JSON Schema before execution.
        Returns isError=True with a descriptive message on failure —
        never raises a raw exception to the client.
        """
        start = time.perf_counter()

        if name not in self._tools:
            return ToolCallResult(
                content=f"Unknown tool: '{name}'. Available: {list(self._tools.keys())}",
                is_error=True,
            )

        tool = self._tools[name]
        # Validate required parameters (lightweight — production uses jsonschema)
        required = tool.input_schema.get("required", [])
        missing = [k for k in required if k not in arguments]
        if missing:
            return ToolCallResult(
                content=f"Invalid params: missing required field(s): {missing}",
                is_error=True,
            )

        # Dispatch to handler
        try:
            if name == "lookup_account":
                result = self._handle_lookup_account(arguments)
            elif name == "update_account_status":
                result = self._handle_update_status(arguments)
            else:
                result = ToolCallResult(content="Handler not implemented", is_error=True)
        except Exception as exc:  # noqa: BLE001
            # Return structured error — never expose raw tracebacks to the LLM
            result = ToolCallResult(
                content=f"Internal error executing '{name}': {exc}",
                is_error=True,
            )

        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def read_resource(self, uri: str) -> str:
        """Return the content of a resource by URI."""
        if uri == "config://crm/regions":
            return json.dumps(["us-east-1", "eu-west-1", "ap-south-1"])
        return f"Resource not found: {uri}"

    # --- Private tool handlers ---

    def _handle_lookup_account(self, args: dict) -> ToolCallResult:
        customer_id = args["customer_id"]
        # Validate enum for account_region (schema validation in production)
        valid_regions = ["us-east-1", "eu-west-1", "ap-south-1"]
        if args["account_region"] not in valid_regions:
            return ToolCallResult(
                content=f"Invalid account_region '{args['account_region']}'. Must be one of {valid_regions}",
                is_error=True,
            )
        account = self._account_db.get(customer_id)
        if not account:
            return ToolCallResult(
                content=f"Account '{customer_id}' not found in region '{args['account_region']}'",
                is_error=True,
            )
        return ToolCallResult(
            content=json.dumps({"customer_id": customer_id, **account}),
            is_error=False,
        )

    def _handle_update_status(self, args: dict) -> ToolCallResult:
        customer_id = args["customer_id"]
        if customer_id not in self._account_db:
            return ToolCallResult(
                content=f"Account '{customer_id}' not found",
                is_error=True,
            )
        old_status = self._account_db[customer_id]["status"]
        self._account_db[customer_id]["status"] = args["new_status"]
        return ToolCallResult(
            content=json.dumps({
                "customer_id": customer_id,
                "previous_status": old_status,
                "new_status": args["new_status"],
                "reason": args["reason"],
            }),
            is_error=False,
        )


# ---------------------------------------------------------------------------
# MCPClient — mirrors the real MCP client interface
# ---------------------------------------------------------------------------

class MCPClient:
    """
    MCP client that connects to an MCPServer and exposes the MCP API surface.

    In production, replace MCPServer with a real transport
    (stdio_client or sse_client from the MCP Python SDK).
    The method signatures here match the real SDK so the swap is mechanical.
    """

    def __init__(self, server: MCPServer) -> None:
        self._server = server
        self._initialised = False
        self._server_capabilities: dict = {}
        self._available_tools: list[ToolDefinition] = []
        self._available_resources: list[ResourceDefinition] = []

    def initialize(self) -> dict[str, Any]:
        """Run the MCP initialisation handshake."""
        response = self._server.initialize()
        self._server_capabilities = response.get("capabilities", {})
        self._initialised = True
        return response

    def list_tools(self) -> list[ToolDefinition]:
        """Discover all tools the server currently exposes."""
        self._require_initialised()
        self._available_tools = self._server.list_tools()
        return self._available_tools

    def list_resources(self) -> list[ResourceDefinition]:
        """Discover all resources the server currently exposes."""
        self._require_initialised()
        self._available_resources = self._server.list_resources()
        return self._available_resources

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Invoke a tool by name with arguments. Returns structured result."""
        self._require_initialised()
        return self._server.call_tool(name, arguments)

    def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        self._require_initialised()
        return self._server.read_resource(uri)

    def _require_initialised(self) -> None:
        if not self._initialised:
            raise RuntimeError("MCPClient.initialize() must be called before making requests")


# ---------------------------------------------------------------------------
# MCPSession — full lifecycle orchestration
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    """Summary of a complete MCP session."""
    server_name: str
    tools_discovered: list[str]
    resources_discovered: list[str]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def run_mcp_session(
    server: MCPServer,
    tool_requests: list[dict[str, Any]],
) -> SessionResult:
    """
    Run a complete MCP session: init → discover → call tools → return summary.

    Args:
        server: The MCPServer to connect to
        tool_requests: List of {"name": str, "arguments": dict} dicts to call

    Returns:
        SessionResult with discovered capabilities and tool call outcomes
    """
    client = MCPClient(server)

    # Phase 1: Initialisation
    init_response = client.initialize()
    server_name = init_response["serverInfo"]["name"]

    # Phase 2: Discovery
    tools = client.list_tools()
    resources = client.list_resources()

    # Phase 3: Tool calls
    call_log: list[dict[str, Any]] = []
    for request in tool_requests:
        result = client.call_tool(request["name"], request["arguments"])
        call_log.append({
            "tool": request["name"],
            "arguments": request["arguments"],
            "result": result.content,
            "is_error": result.is_error,
            "latency_ms": round(result.latency_ms, 2),
        })

    return SessionResult(
        server_name=server_name,
        tools_discovered=[t.name for t in tools],
        resources_discovered=[r.uri for r in resources],
        tool_calls=call_log,
    )
