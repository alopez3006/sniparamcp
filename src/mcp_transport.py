"""
Streamable HTTP transport for MCP protocol.

Implements MCP Streamable HTTP transport specification for direct
connection from MCP clients (Cursor, Claude Code, ChatGPT, Windsurf).
"""

import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import get_project_with_team, validate_api_key, validate_oauth_token
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
        "name": "rlm_plan",
        "description": "Generate full execution plan for complex questions. Returns steps for decomposition, context queries, and synthesis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The complex question to plan for"},
                "strategy": {
                    "type": "string",
                    "enum": ["breadth_first", "depth_first", "relevance_first"],
                    "default": "relevance_first",
                    "description": "Execution strategy",
                },
                "max_tokens": {"type": "integer", "default": 16000, "minimum": 1000, "maximum": 100000},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_multi_project_query",
        "description": "Query across all projects in a team. Requires team API key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or topic"},
                "max_tokens": {"type": "integer", "default": 4000, "minimum": 100, "maximum": 100000},
                "per_project_limit": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20},
                "project_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional project IDs/slugs to include",
                },
                "exclude_project_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional project IDs/slugs to exclude",
                },
                "search_mode": {"type": "string", "enum": ["keyword", "semantic", "hybrid"], "default": "keyword"},
                "include_metadata": {"type": "boolean", "default": True},
                "prefer_summaries": {"type": "boolean", "default": False},
            },
            "required": ["query"],
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
    # Phase 7: Shared Context Tools
    {
        "name": "rlm_shared_context",
        "description": "Get merged context from linked shared collections. Returns categorized docs with budget allocation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_tokens": {"type": "integer", "default": 4000, "minimum": 100, "maximum": 100000},
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["MANDATORY", "BEST_PRACTICES", "GUIDELINES", "REFERENCE"]},
                    "description": "Filter by categories (default: all)",
                },
                "include_content": {"type": "boolean", "default": True, "description": "Include merged content"},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_list_templates",
        "description": "List available prompt templates from shared collections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category"},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_get_template",
        "description": "Get a specific prompt template by ID or slug. Optionally render with variables.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "description": "Template ID"},
                "slug": {"type": "string", "description": "Template slug"},
                "variables": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Variables to substitute in template",
                },
            },
            "required": [],
        },
    },
    # Phase 8.2: Agent Memory Tools
    {
        "name": "rlm_remember",
        "description": "Store a memory for later semantic recall. Supports types: fact, decision, learning, preference, todo, context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "type": {"type": "string", "enum": ["fact", "decision", "learning", "preference", "todo", "context"], "default": "fact"},
                "scope": {"type": "string", "enum": ["agent", "project", "team", "user"], "default": "project"},
                "category": {"type": "string", "description": "Optional category for grouping"},
                "ttl_days": {"type": "integer", "description": "Days until expiration (null = permanent)"},
                "related_to": {"type": "array", "items": {"type": "string"}, "description": "IDs of related memories"},
                "document_refs": {"type": "array", "items": {"type": "string"}, "description": "Referenced document paths"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "rlm_recall",
        "description": "Semantically recall relevant memories based on a query. Uses embeddings weighted by confidence decay.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "type": {"type": "string", "enum": ["fact", "decision", "learning", "preference", "todo", "context"]},
                "scope": {"type": "string", "enum": ["agent", "project", "team", "user"]},
                "category": {"type": "string", "description": "Filter by category"},
                "limit": {"type": "integer", "default": 5, "description": "Maximum memories to return"},
                "min_relevance": {"type": "number", "default": 0.5, "description": "Minimum relevance score (0-1)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "rlm_memories",
        "description": "List memories with optional filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["fact", "decision", "learning", "preference", "todo", "context"]},
                "scope": {"type": "string", "enum": ["agent", "project", "team", "user"]},
                "category": {"type": "string"},
                "search": {"type": "string", "description": "Text search in content"},
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_forget",
        "description": "Delete memories by ID or filter criteria.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Specific memory ID to delete"},
                "type": {"type": "string", "enum": ["fact", "decision", "learning", "preference", "todo", "context"]},
                "category": {"type": "string", "description": "Delete all in this category"},
                "older_than_days": {"type": "integer", "description": "Delete memories older than N days"},
            },
            "required": [],
        },
    },
    # Phase 9.1: Multi-Agent Swarm Tools
    {
        "name": "rlm_swarm_create",
        "description": "Create a new agent swarm for multi-agent coordination.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Swarm name"},
                "description": {"type": "string"},
                "max_agents": {"type": "integer", "default": 10},
                "config": {"type": "object"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "rlm_swarm_join",
        "description": "Join an existing swarm as an agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string", "description": "Swarm to join"},
                "agent_id": {"type": "string", "description": "Your unique agent identifier"},
                "role": {"type": "string", "enum": ["coordinator", "worker", "observer"], "default": "worker"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["swarm_id", "agent_id"],
        },
    },
    {
        "name": "rlm_claim",
        "description": "Claim exclusive access to a resource (file, function, module). Claims auto-expire.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "resource_type": {"type": "string", "enum": ["file", "function", "module", "component", "other"]},
                "resource_id": {"type": "string", "description": "Resource identifier (e.g., file path)"},
                "timeout_seconds": {"type": "integer", "default": 300},
            },
            "required": ["swarm_id", "agent_id", "resource_type", "resource_id"],
        },
    },
    {
        "name": "rlm_release",
        "description": "Release a claimed resource.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "claim_id": {"type": "string"},
                "resource_type": {"type": "string"},
                "resource_id": {"type": "string"},
            },
            "required": ["swarm_id", "agent_id"],
        },
    },
    {
        "name": "rlm_state_get",
        "description": "Read shared swarm state by key.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "key": {"type": "string", "description": "State key to read"},
            },
            "required": ["swarm_id", "key"],
        },
    },
    {
        "name": "rlm_state_set",
        "description": "Write shared swarm state with optimistic locking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "key": {"type": "string"},
                "value": {"description": "Value to set (any JSON-serializable type)"},
                "expected_version": {"type": "integer", "description": "Expected version for optimistic locking"},
            },
            "required": ["swarm_id", "agent_id", "key", "value"],
        },
    },
    {
        "name": "rlm_broadcast",
        "description": "Send an event to all agents in the swarm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "event_type": {"type": "string", "description": "Event type"},
                "payload": {"type": "object", "description": "Event data"},
            },
            "required": ["swarm_id", "agent_id", "event_type"],
        },
    },
    {
        "name": "rlm_task_create",
        "description": "Create a task in the swarm's distributed task queue.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "integer", "default": 0, "description": "Higher = more urgent"},
                "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Task IDs this depends on"},
                "metadata": {"type": "object"},
            },
            "required": ["swarm_id", "agent_id", "title"],
        },
    },
    {
        "name": "rlm_task_claim",
        "description": "Claim a task from the queue. If task_id not specified, claims highest priority available task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "task_id": {"type": "string", "description": "Specific task to claim (optional)"},
                "timeout_seconds": {"type": "integer", "default": 600},
            },
            "required": ["swarm_id", "agent_id"],
        },
    },
    {
        "name": "rlm_task_complete",
        "description": "Mark a claimed task as completed or failed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "swarm_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "task_id": {"type": "string"},
                "success": {"type": "boolean", "default": True},
                "result": {"description": "Task result data"},
            },
            "required": ["swarm_id", "agent_id", "task_id"],
        },
    },
    # Phase 10: Document Sync Tools
    {
        "name": "rlm_settings",
        "description": "Get current project settings from dashboard (max_tokens, search_mode, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "refresh": {"type": "boolean", "default": False, "description": "Force refresh from API"},
            },
            "required": [],
        },
    },
    {
        "name": "rlm_upload_document",
        "description": "Upload or update a document in the project. Supports .md, .txt, .mdx files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document path (e.g., 'docs/api.md')"},
                "content": {"type": "string", "description": "Document content (markdown)"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "rlm_sync_documents",
        "description": "Bulk sync multiple documents. Use for batch uploads or CI/CD integration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                    "description": "Documents to sync",
                },
                "delete_missing": {"type": "boolean", "default": False, "description": "Delete docs not in list"},
            },
            "required": ["documents"],
        },
    },
]


def jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


async def validate_request(project_id_or_slug: str, api_key: str) -> tuple[dict | None, Plan, str | None, str | None]:
    """Validate API key or OAuth token and check limits. Returns (auth_info, plan, error, actual_project_id)."""
    auth_info = None

    # Check if it's an OAuth token
    if api_key.startswith("snipara_at_"):
        auth_info = await validate_oauth_token(api_key, project_id_or_slug)
        if not auth_info:
            return None, Plan.FREE, "Invalid or expired OAuth token", None
    else:
        # Fall back to API key validation
        auth_info = await validate_api_key(api_key, project_id_or_slug)
        if not auth_info:
            return None, Plan.FREE, "Invalid API key", None

    project = await get_project_with_team(project_id_or_slug)
    if not project:
        return None, Plan.FREE, "Project not found", None

    # Use actual database ID for all operations
    actual_project_id = project.id

    if not await check_rate_limit(auth_info["id"]):
        return None, Plan.FREE, f"Rate limit exceeded: {settings.rate_limit_requests}/min", None

    plan = Plan(project.team.subscription.plan if project.team.subscription else "FREE")
    limits = await check_usage_limits(actual_project_id, plan)
    if limits.exceeded:
        return None, plan, f"Monthly limit exceeded: {limits.current}/{limits.max}", None

    return auth_info, plan, None, actual_project_id


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
async def mcp_endpoint(
    project_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """
    MCP Streamable HTTP endpoint.

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
        raise HTTPException(status_code=401, detail="Missing authentication: X-API-Key or Authorization header required")

    api_key_info, plan, error, actual_project_id = await validate_request(project_id, api_key)
    if error:
        raise HTTPException(status_code=401 if "Invalid" in error else 429, detail=error)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    # Use actual database ID for all operations (not URL slug)
    if isinstance(body, list):
        responses = [r for req in body if (r := await handle_request(req, actual_project_id, plan))]
        return JSONResponse(responses)

    response = await handle_request(body, actual_project_id, plan)
    return JSONResponse(response) if response else JSONResponse({}, status_code=204)


@router.get("/{project_id}")
async def mcp_sse(
    project_id: str,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """MCP SSE endpoint for server-initiated messages (keep-alive)."""
    # Accept X-API-Key header (preferred) or Authorization: Bearer
    if x_api_key:
        api_key = x_api_key
    elif authorization:
        api_key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    else:
        raise HTTPException(status_code=401, detail="Missing authentication: X-API-Key or Authorization header required")

    _, _, error, _ = await validate_request(project_id, api_key)
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
