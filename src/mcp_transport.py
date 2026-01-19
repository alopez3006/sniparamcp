"""
Streamable HTTP transport for MCP protocol.

Implements MCP Streamable HTTP transport specification for direct
connection from MCP clients (Cursor, Claude Code, ChatGPT, Windsurf).
"""

import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import get_project_with_team, validate_api_key
from .config import settings
from .models import Plan, ToolName
from .rlm_engine import RLMEngine
from .usage import check_rate_limit, check_usage_limits, track_usage

router = APIRouter(prefix="/mcp", tags=["MCP Transport"])

MCP_VERSION = "2024-11-05"

# Tool definitions for MCP list_tools
TOOL_DEFINITIONS = [
    {
        "name": "rlm_context_query",
        "description": "Query optimized context from documentation. Returns ranked sections within token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or topic"},
                "max_tokens": {"type": "integer", "default": 4000, "minimum": 100, "maximum": 100000},
                "search_mode": {"type": "string", "enum": ["keyword", "semantic", "hybrid"], "default": "hybrid"},
                "include_metadata": {"type": "boolean", "default": True},
                "prefer_summaries": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_ask",
        "description": "Query documentation with a question (basic). Use rlm_context_query for better results.",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "rlm_search",
        "description": "Search documentation for a regex pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_results": {"type": "integer", "default": 20},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "rlm_decompose",
        "description": "Break complex query into sub-queries with execution order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_multi_query",
        "description": "Execute multiple queries in one call with shared token budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"query": {"type": "string"}, "max_tokens": {"type": "integer"}}, "required": ["query"]},
                    "minItems": 1, "maxItems": 10,
                },
                "max_tokens": {"type": "integer", "default": 8000},
            },
            "required": ["queries"],
        },
    },
    {
        "name": "rlm_inject",
        "description": "Set session context for subsequent queries.",
        "inputSchema": {
            "type": "object",
            "properties": {"context": {"type": "string"}, "append": {"type": "boolean", "default": False}},
            "required": ["context"],
        },
    },
    {
        "name": "rlm_context",
        "description": "Show current session context.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_clear_context",
        "description": "Clear session context.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_stats",
        "description": "Show documentation statistics.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_sections",
        "description": "List all indexed document sections.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rlm_read",
        "description": "Read specific lines from documentation.",
        "inputSchema": {
            "type": "object",
            "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            "required": ["start_line", "end_line"],
        },
    },
    {
        "name": "rlm_store_summary",
        "description": "Store an LLM-generated summary for a document.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_path": {"type": "string"},
                "summary": {"type": "string"},
                "summary_type": {"type": "string", "enum": ["concise", "detailed", "technical", "keywords", "custom"], "default": "concise"},
                "generated_by": {"type": "string"},
            },
            "required": ["document_path", "summary"],
        },
    },
    {
        "name": "rlm_get_summaries",
        "description": "Retrieve stored summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_path": {"type": "string"},
                "summary_type": {"type": "string", "enum": ["concise", "detailed", "technical", "keywords", "custom"]},
                "include_content": {"type": "boolean", "default": True},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_delete_summary",
        "description": "Delete stored summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {"summary_id": {"type": "string"}, "document_path": {"type": "string"}},
            "required": [],
        },
    },
]


def jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


async def validate_request(project_id: str, api_key: str) -> tuple[dict | None, Plan, str | None]:
    """Validate API key and check limits. Returns (api_key_info, plan, error)."""
    api_key_info = await validate_api_key(api_key, project_id)
    if not api_key_info:
        return None, Plan.FREE, "Invalid API key"

    project = await get_project_with_team(project_id)
    if not project:
        return None, Plan.FREE, "Project not found"

    if not await check_rate_limit(api_key_info["id"]):
        return None, Plan.FREE, f"Rate limit exceeded: {settings.rate_limit_requests}/min"

    plan = Plan(project.team.subscription.plan if project.team.subscription else "FREE")
    limits = await check_usage_limits(project_id, plan)
    if limits.exceeded:
        return None, plan, f"Monthly limit exceeded: {limits.current}/{limits.max}"

    return api_key_info, plan, None


async def handle_call_tool(id: Any, params: dict, project_id: str, plan: Plan) -> dict:
    """Handle MCP tools/call request."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    try:
        tool_enum = ToolName(tool_name)
    except ValueError:
        return jsonrpc_error(id, -32602, f"Unknown tool: {tool_name}")

    try:
        engine = RLMEngine(project_id, plan=plan)
        result = await engine.execute(tool_enum, arguments)

        await track_usage(
            project_id=project_id, tool=tool_name,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            latency_ms=0, success=True,
        )

        return jsonrpc_response(id, {
            "content": [{"type": "text", "text": json.dumps(result.data, indent=2, default=str)}],
        })
    except Exception as e:
        await track_usage(
            project_id=project_id, tool=tool_name,
            input_tokens=0, output_tokens=0, latency_ms=0, success=False, error=str(e),
        )
        return jsonrpc_error(id, -32000, str(e))


async def handle_request(body: dict, project_id: str, plan: Plan) -> dict | None:
    """Handle a single JSON-RPC request."""
    method, id, params = body.get("method"), body.get("id"), body.get("params", {})

    if id is None:  # Notification
        return None

    if method == "initialize":
        return jsonrpc_response(id, {
            "protocolVersion": MCP_VERSION,
            "serverInfo": {"name": "snipara", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        })
    elif method == "tools/list":
        return jsonrpc_response(id, {"tools": TOOL_DEFINITIONS})
    elif method == "tools/call":
        return await handle_call_tool(id, params, project_id, plan)
    elif method == "ping":
        return jsonrpc_response(id, {})
    else:
        return jsonrpc_error(id, -32601, f"Method not found: {method}")


@router.post("/{project_id}")
async def mcp_endpoint(project_id: str, request: Request, authorization: str = Header(...)):
    """
    MCP Streamable HTTP endpoint.

    Config example (Claude Code):
    ```json
    {"mcpServers": {"snipara": {"type": "http", "url": "https://api.snipara.com/mcp/{project_id}", "headers": {"Authorization": "Bearer sk-..."}}}}
    ```
    """
    api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization

    api_key_info, plan, error = await validate_request(project_id, api_key)
    if error:
        raise HTTPException(status_code=401 if "Invalid" in error else 429, detail=error)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    if isinstance(body, list):
        responses = [r for req in body if (r := await handle_request(req, project_id, plan))]
        return JSONResponse(responses)

    response = await handle_request(body, project_id, plan)
    return JSONResponse(response) if response else JSONResponse({}, status_code=204)


@router.get("/{project_id}")
async def mcp_sse(project_id: str, authorization: str = Header(...)):
    """MCP SSE endpoint for server-initiated messages (keep-alive)."""
    api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization

    _, _, error = await validate_request(project_id, api_key)
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

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
