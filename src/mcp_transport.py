"""MCP Streamable HTTP Transport for Snipara.

This module implements the MCP (Model Context Protocol) Streamable HTTP transport
specification, enabling direct connections from MCP-compatible AI clients.

Supported Clients:
    - Claude Code (Anthropic)
    - Cursor IDE
    - ChatGPT (with MCP support)
    - Windsurf
    - Any MCP-compatible client

Protocol:
    Uses JSON-RPC 2.0 over HTTP with the following methods:
    - initialize: Establish connection and exchange capabilities
    - tools/list: List available tools
    - tools/call: Execute a tool with arguments
    - ping: Keep-alive check

Endpoints:
    POST /mcp/{project_id}  - Main JSON-RPC endpoint for tool execution
    GET  /mcp/{project_id}  - SSE endpoint for server-initiated messages

Authentication:
    Accepts either:
    - X-API-Key header: Project API key (rlm_...) or Team API key
    - Authorization: Bearer header: API key or OAuth token (snipara_at_...)

Example Configuration (Claude Code .mcp.json):
    {
        "mcpServers": {
            "snipara": {
                "type": "http",
                "url": "https://api.snipara.com/mcp/{project_slug}",
                "headers": {"X-API-Key": "rlm_..."}
            }
        }
    }

Note:
    Team-scoped queries (/mcp/team/{team_id}) are handled in server.py
    to avoid circular imports with execute_multi_project_query.
"""

import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .mcp import TOOL_DEFINITIONS, jsonrpc_error, jsonrpc_response
from .mcp.validation import validate_request
from .models import Plan, ToolName
from .rlm_engine import RLMEngine
from .usage import track_usage

# ============ HELPERS ============


def _get_client_ip(request: Request) -> str | None:
    """Extract client IP from X-Forwarded-For header or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


# ============ ROUTER CONFIGURATION ============

router = APIRouter(prefix="/mcp", tags=["MCP Transport"])

#: MCP protocol version (spec: 2024-11-05)
MCP_VERSION = "2024-11-05"


# ============ REQUEST HANDLERS ============


async def handle_call_tool(
    id: Any, params: dict, project_id: str, plan: Plan, access_level: str = "EDITOR"
) -> dict:
    """Handle MCP tools/call request.

    Executes a tool through the RLMEngine and tracks usage.

    Args:
        id: JSON-RPC request ID
        params: Tool call parameters containing:
            - name: Tool name (e.g., "rlm_context_query")
            - arguments: Tool-specific arguments
        project_id: Database project ID
        plan: Subscription plan for rate limiting
        access_level: API key access level (VIEWER, EDITOR, ADMIN)

    Returns:
        JSON-RPC response with tool result or error
    """
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    try:
        tool_enum = ToolName(tool_name)
    except ValueError:
        return jsonrpc_error(id, -32602, f"Unknown tool: {tool_name}")

    try:
        engine = RLMEngine(project_id, plan=plan, access_level=access_level)
        result = await engine.execute(tool_enum, arguments)

        await track_usage(
            project_id=project_id,
            tool=tool_name,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=0,
            success=True,
        )

        return jsonrpc_response(
            id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(result.data, indent=2, default=str)}
                ],
            },
        )
    except Exception as e:
        await track_usage(
            project_id=project_id,
            tool=tool_name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            success=False,
            error=str(e),
        )
        return jsonrpc_error(id, -32000, str(e))


async def handle_request(
    body: dict, project_id: str, plan: Plan, access_level: str = "EDITOR"
) -> dict | None:
    """Handle a single JSON-RPC request.

    Routes requests to appropriate handlers based on method.

    Supported Methods:
        - initialize: Returns server info and capabilities
        - tools/list: Returns available tool definitions
        - tools/call: Executes a tool
        - ping: Returns empty response (keep-alive)

    Args:
        body: JSON-RPC request body
        project_id: Database project ID
        plan: Subscription plan
        access_level: API key access level (VIEWER, EDITOR, ADMIN)

    Returns:
        JSON-RPC response dict, or None for notifications (requests without id)
    """
    method, id, params = body.get("method"), body.get("id"), body.get("params", {})

    if id is None:  # Notification
        return None

    if method == "initialize":
        return jsonrpc_response(
            id,
            {
                "protocolVersion": MCP_VERSION,
                "serverInfo": {"name": "snipara", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        )
    elif method == "tools/list":
        return jsonrpc_response(id, {"tools": TOOL_DEFINITIONS})
    elif method == "tools/call":
        return await handle_call_tool(id, params, project_id, plan, access_level)
    elif method == "ping":
        return jsonrpc_response(id, {})
    else:
        return jsonrpc_error(id, -32601, f"Method not found: {method}")


# ============ HTTP ENDPOINTS ============


@router.post("/{project_id}")
async def mcp_endpoint(
    project_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """MCP Streamable HTTP endpoint.

    Accepts authentication via either X-API-Key or Authorization: Bearer header.

    Config example (Claude Code):
    ```json
    {"mcpServers": {"snipara": {"type": "http", "url": "https://api.snipara.com/mcp/{project_id}", "headers": {"X-API-Key": "rlm_..."}}}}
    ```

    Alternative (Authorization Bearer):
    ```json
    {"mcpServers": {"snipara": {"type": "http", "url": "https://api.snipara.com/mcp/{project_id}", "headers": {"Authorization": "Bearer rlm_..."}}}}
    ```
    """
    # Accept X-API-Key header (preferred) or Authorization: Bearer
    if x_api_key:
        api_key = x_api_key
    elif authorization:
        api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    else:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing authentication. Get started free (100 queries/month, no credit card):\n"
                "- Claude Code: Run /snipara:quickstart\n"
                "- VS Code: Install 'Snipara' extension and click 'Sign in with GitHub'\n"
                "- Manual: Get an API key at https://snipara.com/dashboard\n"
                "Docs: https://snipara.com/docs/quickstart"
            ),
        )

    client_ip = _get_client_ip(request)
    api_key_info, plan, error, actual_project_id = await validate_request(project_id, api_key, client_ip=client_ip)
    if error:
        raise HTTPException(status_code=401 if "Invalid" in error else 429, detail=error)

    # Extract access level from validated API key (defaults to EDITOR if not set)
    access_level = api_key_info.get("access_level", "EDITOR") if api_key_info else "EDITOR"

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    # Use actual database ID for all operations (not URL slug)
    if isinstance(body, list):
        responses = [r for req in body if (r := await handle_request(req, actual_project_id, plan, access_level))]
        return JSONResponse(responses)

    response = await handle_request(body, actual_project_id, plan, access_level)
    return JSONResponse(response) if response else Response(status_code=204)


@router.get("/{project_id}")
async def mcp_sse(
    project_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """MCP Server-Sent Events (SSE) endpoint.

    Provides a persistent connection for server-initiated messages.
    Currently used for keep-alive pings every 30 seconds.

    Args:
        project_id: Project ID or slug
        x_api_key: API key via X-API-Key header
        authorization: API key via Authorization: Bearer header

    Returns:
        SSE stream with JSON messages:
        - {"type": "connected"} on connection
        - {"type": "ping"} every 30 seconds
    """
    # Accept X-API-Key header (preferred) or Authorization: Bearer
    if x_api_key:
        api_key = x_api_key
    elif authorization:
        api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    else:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing authentication. Get started free (100 queries/month, no credit card):\n"
                "- Claude Code: Run /snipara:quickstart\n"
                "- VS Code: Install 'Snipara' extension and click 'Sign in with GitHub'\n"
                "- Manual: Get an API key at https://snipara.com/dashboard\n"
                "Docs: https://snipara.com/docs/quickstart"
            ),
        )

    client_ip = _get_client_ip(request)
    _, _, error, _ = await validate_request(project_id, api_key, client_ip=client_ip)
    if error:
        raise HTTPException(status_code=401, detail=error)

    async def stream():
        import asyncio

        yield f"data: {json.dumps({'type': 'connected'})}\n\n"
        try:
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
