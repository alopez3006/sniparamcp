"""
GPT Actions REST endpoints for ChatGPT Custom GPTs.

ChatGPT Custom GPTs use "Actions" to call external APIs, but they only support
simple REST endpoints (not JSON-RPC or MCP protocol). This module provides
thin REST wrappers around the existing RLMEngine for use with GPT Actions.

Two authentication modes:

1. **API Key (per-project):** /v1/gpt/{project_slug}/*
   - User creates a private Custom GPT and pastes their API key
   - Project is specified in the URL path

2. **OAuth Bearer (project-agnostic):** /v1/gpt/me/*
   - For the public GPT Store — any Snipara user can sign in via OAuth
   - Project is resolved from the OAuth token (each token is scoped to one project)
   - Users select their project during the OAuth authorization flow

Endpoints (API Key):
- POST /v1/gpt/{project_slug}/query   - Context-optimized documentation query
- POST /v1/gpt/{project_slug}/search  - Regex pattern search
- GET  /v1/gpt/{project_slug}/info    - Project stats and capabilities
- GET  /v1/gpt/{project_slug}/openapi - Auto-generated OpenAPI 3.1 spec

Endpoints (OAuth Bearer — GPT Store):
- POST /v1/gpt/me/query              - Query docs (project from token)
- POST /v1/gpt/me/search             - Search docs (project from token)
- GET  /v1/gpt/me/info               - Project info (project from token)
- POST /v1/gpt/me/remember           - Store memory (project from token)
- POST /v1/gpt/me/recall             - Recall memories (project from token)
- GET  /v1/gpt/me/shared-context     - Team shared context (project from token)
- GET  /v1/gpt/me/openapi            - OpenAPI spec for GPT Store (no auth)
"""

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .models import Plan, ToolName
from .rlm_engine import RLMEngine
from .usage import check_usage_limits, track_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/gpt", tags=["GPT Actions"])


# ============ REQUEST/RESPONSE MODELS ============


class GPTQueryRequest(BaseModel):
    """Request body for GPT context query."""

    query: str = Field(..., description="The question to ask about the documentation")
    max_tokens: int = Field(
        default=4000,
        ge=100,
        le=16000,
        description="Maximum tokens to return (default: 4000)",
    )


class GPTQueryResponse(BaseModel):
    """Response for GPT context query."""

    context: str = Field(..., description="Optimized context from documentation")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source sections with title, file, lines, and relevance",
    )
    tokens_used: int = Field(default=0, description="Tokens used for this response")
    query: str = Field(..., description="Original query")


class GPTSearchRequest(BaseModel):
    """Request body for GPT search."""

    pattern: str = Field(..., description="Regex pattern to search for in documentation")
    max_results: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum results to return (default: 20)",
    )


class GPTSearchResponse(BaseModel):
    """Response for GPT search."""

    matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Matching sections with content, file, and line numbers",
    )
    pattern: str = Field(..., description="Original search pattern")
    total_matches: int = Field(default=0, description="Total number of matches found")


class GPTInfoResponse(BaseModel):
    """Response for GPT project info."""

    project_slug: str = Field(..., description="Project identifier")
    files_loaded: int = Field(default=0, description="Number of documentation files")
    total_lines: int = Field(default=0, description="Total lines of documentation")
    total_sections: int = Field(default=0, description="Number of indexed sections")
    available_actions: list[str] = Field(
        default_factory=list,
        description="Available GPT actions for this project",
    )


class GPTRememberRequest(BaseModel):
    """Request body for storing a memory."""

    content: str = Field(..., description="The memory content to store")
    type: str = Field(
        default="fact",
        description="Memory type: fact, decision, learning, preference, todo, or context",
    )
    category: str | None = Field(default=None, description="Optional category for grouping")


class GPTRememberResponse(BaseModel):
    """Response for storing a memory."""

    success: bool = Field(default=True)
    memory_id: str = Field(..., description="ID of the stored memory")
    message: str = Field(default="Memory stored successfully")


class GPTRecallRequest(BaseModel):
    """Request body for recalling memories."""

    query: str = Field(..., description="Search query to find relevant memories")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum memories to return")
    type: str | None = Field(default=None, description="Filter by memory type")


class GPTRecallResponse(BaseModel):
    """Response for recalling memories."""

    memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Matching memories with content, type, relevance, and metadata",
    )
    query: str = Field(..., description="Original search query")
    total: int = Field(default=0, description="Total memories returned")


class GPTSharedContextResponse(BaseModel):
    """Response for team shared context."""

    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Shared context documents with title, category, and content",
    )
    total_tokens: int = Field(default=0, description="Total tokens used")
    categories: list[str] = Field(
        default_factory=list,
        description="Categories present in the response",
    )


# ============ AUTH DEPENDENCIES ============


async def get_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Extract API key from header."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header. Add your Snipara API key in GPT Builder auth settings.",
        )
    return x_api_key


async def get_bearer_token(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Extract OAuth Bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Sign in via OAuth to use this GPT.",
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header. Expected 'Bearer <token>'.",
        )
    return authorization[7:]  # Strip "Bearer " prefix


# ============ SHARED VALIDATION ============


async def _validate_gpt_request(
    project_slug: str,
    api_key: str,
) -> tuple[dict, Any, Plan, dict | None]:
    """
    Validate GPT Action request using the same auth flow as MCP endpoints.

    Imports validate_and_rate_limit lazily to avoid circular imports.
    """
    # Import here to avoid circular dependency with server.py
    from .server import validate_and_rate_limit

    return await validate_and_rate_limit(project_slug, api_key)


async def _validate_gpt_oauth_request(
    token: str,
) -> tuple[dict, Any, Plan, dict | None]:
    """
    Validate GPT Action request using OAuth Bearer token.

    Resolves the project from the token itself (project-agnostic).
    Used by /v1/gpt/me/* endpoints for the public GPT Store.

    Returns:
        Tuple of (auth_info, project, plan, project_settings)
    """
    from .auth import get_project_settings, validate_oauth_token_any_project
    from .usage import check_rate_limit, is_scan_blocked

    # Anti-scan check
    key_prefix = token[:12]
    if await is_scan_blocked(key_prefix):
        raise HTTPException(status_code=429, detail="Too many failed requests. Try again later.")

    auth_info = await validate_oauth_token_any_project(token)
    if not auth_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired OAuth token. Please sign in again at snipara.com.",
        )

    project = auth_info["project"]
    if not project:
        raise HTTPException(status_code=404, detail="Project not found for this token.")

    # Rate limit
    rate_ok = await check_rate_limit(auth_info["id"])
    if not rate_ok:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again shortly.")

    # Determine plan
    plan = Plan(project.team.subscription.plan if project.team and project.team.subscription else "FREE")

    # Get project settings
    project_settings = await get_project_settings(project.id)

    return auth_info, project, plan, project_settings


# ============ GPT ACTION ENDPOINTS ============


@router.post(
    "/{project_slug}/query",
    response_model=GPTQueryResponse,
    summary="Query documentation",
    description=(
        "Query your project documentation with a natural language question. "
        "Returns optimized, relevant context from your docs within the token budget. "
        "Use this to find information about your codebase, APIs, architecture, etc."
    ),
)
async def gpt_query(
    project_slug: str,
    request: GPTQueryRequest,
    api_key: Annotated[str, Depends(get_api_key)],
) -> GPTQueryResponse:
    """Query documentation and return optimized context."""
    start_time = time.perf_counter()

    # Validate auth, project, rate limit
    api_key_info, project, plan, project_settings = await _validate_gpt_request(
        project_slug, api_key
    )

    # Check usage limits
    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_CONTEXT_QUERY,
            {
                "query": request.query,
                "max_tokens": request.max_tokens,
                "include_metadata": True,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_query",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        # Flatten sections into a single context string for the GPT
        data = result.data
        sections = data.get("sections", []) if isinstance(data, dict) else []

        context_parts = []
        sources = []
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            file_path = section.get("file", "")
            lines = section.get("lines", (0, 0))
            relevance = section.get("relevance_score", 0)

            context_parts.append(f"## {title}\n{content}")
            sources.append({
                "title": title,
                "file": file_path,
                "lines": lines,
                "relevance": round(relevance, 3),
            })

        context_text = "\n\n".join(context_parts) if context_parts else "No relevant documentation found for this query."
        tokens_used = data.get("total_tokens", 0) if isinstance(data, dict) else 0

        return GPTQueryResponse(
            context=context_text,
            sources=sources,
            tokens_used=tokens_used,
            query=request.query,
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_query",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your query. Please try again.",
        )


@router.post(
    "/{project_slug}/search",
    response_model=GPTSearchResponse,
    summary="Search documentation",
    description=(
        "Search your project documentation using a regex pattern. "
        "Returns matching sections with file paths and line numbers. "
        "Useful for finding specific code patterns, function names, or keywords."
    ),
)
async def gpt_search(
    project_slug: str,
    request: GPTSearchRequest,
    api_key: Annotated[str, Depends(get_api_key)],
) -> GPTSearchResponse:
    """Search documentation with regex pattern."""
    start_time = time.perf_counter()

    api_key_info, project, plan, project_settings = await _validate_gpt_request(
        project_slug, api_key
    )

    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_SEARCH,
            {
                "pattern": request.pattern,
                "max_results": request.max_results,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_search",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data
        matches = []
        if isinstance(data, dict):
            for match in data.get("matches", []):
                matches.append({
                    "content": match.get("content", ""),
                    "file": match.get("file", ""),
                    "line": match.get("line", 0),
                    "context": match.get("context", ""),
                })

        return GPTSearchResponse(
            matches=matches,
            pattern=request.pattern,
            total_matches=len(matches),
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_search",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your search. Please try again.",
        )


@router.get(
    "/{project_slug}/info",
    response_model=GPTInfoResponse,
    summary="Get project info",
    description=(
        "Get information about the documentation project including "
        "file count, section count, and available actions. "
        "Use this to understand what documentation is available."
    ),
)
async def gpt_info(
    project_slug: str,
    api_key: Annotated[str, Depends(get_api_key)],
) -> GPTInfoResponse:
    """Get project documentation stats."""
    start_time = time.perf_counter()

    api_key_info, project, plan, project_settings = await _validate_gpt_request(
        project_slug, api_key
    )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(ToolName.RLM_STATS, {})

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_info",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}

        return GPTInfoResponse(
            project_slug=project_slug,
            files_loaded=data.get("files_loaded", 0),
            total_lines=data.get("total_lines", 0),
            total_sections=data.get("sections", 0),
            available_actions=["query", "search", "info", "remember", "recall", "shared-context"],
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_info",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred fetching project info. Please try again.",
        )


# ============ MEMORY ENDPOINTS ============


@router.post(
    "/{project_slug}/remember",
    response_model=GPTRememberResponse,
    summary="Store a memory",
    description=(
        "Store a fact, decision, learning, preference, or context for later recall. "
        "Memories persist across sessions and can be retrieved semantically."
    ),
)
async def gpt_remember(
    project_slug: str,
    request: GPTRememberRequest,
    api_key: Annotated[str, Depends(get_api_key)],
) -> GPTRememberResponse:
    """Store a memory for later semantic recall."""
    start_time = time.perf_counter()

    api_key_info, project, plan, project_settings = await _validate_gpt_request(
        project_slug, api_key
    )

    # Memory features require at least PRO plan
    if plan == Plan.FREE:
        raise HTTPException(
            status_code=403,
            detail="Memory features require a paid plan. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_REMEMBER,
            {
                "content": request.content,
                "type": request.type,
                "category": request.category,
                "scope": "project",
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_remember",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}

        return GPTRememberResponse(
            success=True,
            memory_id=data.get("memory_id", ""),
            message=data.get("message", "Memory stored successfully"),
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_remember",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT remember error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred storing the memory. Please try again.",
        )


@router.post(
    "/{project_slug}/recall",
    response_model=GPTRecallResponse,
    summary="Recall memories",
    description=(
        "Semantically recall relevant memories based on a query. "
        "Returns previously stored facts, decisions, learnings, and context."
    ),
)
async def gpt_recall(
    project_slug: str,
    request: GPTRecallRequest,
    api_key: Annotated[str, Depends(get_api_key)],
) -> GPTRecallResponse:
    """Recall memories by semantic similarity."""
    start_time = time.perf_counter()

    api_key_info, project, plan, project_settings = await _validate_gpt_request(
        project_slug, api_key
    )

    if plan == Plan.FREE:
        raise HTTPException(
            status_code=403,
            detail="Memory features require a paid plan. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_RECALL,
            {
                "query": request.query,
                "limit": request.limit,
                "type": request.type,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_recall",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}
        memories = data.get("memories", [])

        return GPTRecallResponse(
            memories=memories,
            query=request.query,
            total=len(memories),
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_recall",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT recall error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred recalling memories. Please try again.",
        )


@router.get(
    "/{project_slug}/shared-context",
    response_model=GPTSharedContextResponse,
    summary="Get team shared context",
    description=(
        "Get merged context from team shared collections including coding standards, "
        "best practices, and guidelines. Requires Team plan or higher."
    ),
)
async def gpt_shared_context(
    project_slug: str,
    api_key: Annotated[str, Depends(get_api_key)],
    max_tokens: int = Query(
        default=4000,
        ge=100,
        le=16000,
        description="Token budget for shared context",
    ),
) -> GPTSharedContextResponse:
    """Get team shared context documents."""
    start_time = time.perf_counter()

    api_key_info, project, plan, project_settings = await _validate_gpt_request(
        project_slug, api_key
    )

    if plan not in (Plan.TEAM, Plan.ENTERPRISE):
        raise HTTPException(
            status_code=403,
            detail="Shared context requires Team plan or higher. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_SHARED_CONTEXT,
            {
                "max_tokens": max_tokens,
                "include_content": True,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_shared_context",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}
        documents = data.get("documents", [])
        categories = list({doc.get("category", "") for doc in documents if doc.get("category")})

        return GPTSharedContextResponse(
            documents=documents,
            total_tokens=data.get("total_tokens", 0),
            categories=categories,
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_shared_context",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT shared context error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred fetching shared context. Please try again.",
        )


# ============ OPENAPI SPEC GENERATOR ============


@router.get(
    "/{project_slug}/openapi",
    summary="Generate OpenAPI spec for GPT Builder",
    description=(
        "Auto-generates an OpenAPI 3.1 specification for this project. "
        "Copy and paste this into ChatGPT's GPT Builder to create a Custom GPT "
        "that can query your documentation."
    ),
)
async def gpt_openapi_spec(
    project_slug: str,
    api_key: Annotated[str, Depends(get_api_key)],
    server_url: str = Query(
        default="https://api.snipara.com",
        description="Base URL of the Snipara API server",
    ),
) -> dict:
    """Generate per-project OpenAPI 3.1 spec for GPT Builder."""
    # Validate auth to ensure the project exists and key is valid
    await _validate_gpt_request(project_slug, api_key)

    spec = {
        "openapi": "3.1.0",
        "info": {
            "title": f"Snipara - {project_slug}",
            "description": (
                f"Query, search, and manage the '{project_slug}' documentation project on Snipara. "
                "Use query for documentation context, search for patterns, "
                "remember/recall for persistent memory, and shared-context for team standards."
            ),
            "version": "1.0.0",
        },
        "servers": [
            {"url": server_url, "description": "Snipara API"},
        ],
        "paths": {
            f"/v1/gpt/{project_slug}/query": {
                "post": {
                    "operationId": "queryDocumentation",
                    "summary": "Query documentation with a natural language question",
                    "description": (
                        "Ask a question about the project documentation. "
                        "Returns relevant, optimized context within the token budget."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The question to ask about the documentation",
                                        },
                                        "max_tokens": {
                                            "type": "integer",
                                            "default": 4000,
                                            "minimum": 100,
                                            "maximum": 16000,
                                            "description": "Maximum tokens to return",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Documentation context",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "context": {
                                                "type": "string",
                                                "description": "Relevant documentation context",
                                            },
                                            "sources": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "title": {"type": "string"},
                                                        "file": {"type": "string"},
                                                        "relevance": {"type": "number"},
                                                    },
                                                },
                                                "description": "Source sections",
                                            },
                                            "tokens_used": {
                                                "type": "integer",
                                                "description": "Tokens used",
                                            },
                                            "query": {
                                                "type": "string",
                                                "description": "Original query",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            f"/v1/gpt/{project_slug}/search": {
                "post": {
                    "operationId": "searchDocumentation",
                    "summary": "Search documentation with a regex pattern",
                    "description": (
                        "Search for specific patterns, function names, or keywords "
                        "in the project documentation using regex."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["pattern"],
                                    "properties": {
                                        "pattern": {
                                            "type": "string",
                                            "description": "Regex pattern to search for",
                                        },
                                        "max_results": {
                                            "type": "integer",
                                            "default": 20,
                                            "minimum": 1,
                                            "maximum": 50,
                                            "description": "Maximum results to return",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "matches": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "content": {"type": "string"},
                                                        "file": {"type": "string"},
                                                        "line": {"type": "integer"},
                                                    },
                                                },
                                            },
                                            "pattern": {"type": "string"},
                                            "total_matches": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            f"/v1/gpt/{project_slug}/info": {
                "get": {
                    "operationId": "getProjectInfo",
                    "summary": "Get project documentation info",
                    "description": (
                        "Get information about the documentation project including "
                        "file count, section count, and available actions."
                    ),
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "Project information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "project_slug": {"type": "string"},
                                            "files_loaded": {"type": "integer"},
                                            "total_lines": {"type": "integer"},
                                            "total_sections": {"type": "integer"},
                                            "available_actions": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            f"/v1/gpt/{project_slug}/remember": {
                "post": {
                    "operationId": "storeMemory",
                    "summary": "Store a memory for later recall",
                    "description": (
                        "Store a fact, decision, learning, preference, or context. "
                        "Memories persist across sessions and can be recalled semantically."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["content"],
                                    "properties": {
                                        "content": {
                                            "type": "string",
                                            "description": "The memory content to store",
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                                            "default": "fact",
                                            "description": "Memory type",
                                        },
                                        "category": {
                                            "type": "string",
                                            "description": "Optional grouping category",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Memory stored",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "memory_id": {"type": "string"},
                                            "message": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            f"/v1/gpt/{project_slug}/recall": {
                "post": {
                    "operationId": "recallMemories",
                    "summary": "Recall stored memories",
                    "description": (
                        "Semantically search stored memories. "
                        "Returns relevant facts, decisions, and learnings."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Search query for memories",
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "default": 5,
                                            "minimum": 1,
                                            "maximum": 20,
                                            "description": "Max memories to return",
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                                            "description": "Filter by memory type",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Matching memories",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "memories": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "content": {"type": "string"},
                                                        "type": {"type": "string"},
                                                        "relevance": {"type": "number"},
                                                    },
                                                },
                                            },
                                            "query": {"type": "string"},
                                            "total": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            f"/v1/gpt/{project_slug}/shared-context": {
                "get": {
                    "operationId": "getSharedContext",
                    "summary": "Get team shared context and standards",
                    "description": (
                        "Get merged context from team shared collections: "
                        "coding standards, best practices, and guidelines."
                    ),
                    "parameters": [
                        {
                            "name": "max_tokens",
                            "in": "query",
                            "schema": {"type": "integer", "default": 4000},
                            "description": "Token budget",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Shared context documents",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "documents": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "title": {"type": "string"},
                                                        "category": {"type": "string"},
                                                        "content": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "total_tokens": {"type": "integer"},
                                            "categories": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "Your Snipara API key (starts with rlm_)",
                },
            },
        },
        "security": [{"ApiKeyAuth": []}],
    }

    return spec


@router.get(
    "/{project_slug}/setup",
    summary="Get Custom GPT setup instructions",
    description=(
        "Returns everything needed to create a ChatGPT Custom GPT for this project: "
        "system prompt instructions, OpenAPI spec URL, and auth configuration steps."
    ),
)
async def gpt_setup(
    project_slug: str,
    api_key: Annotated[str, Depends(get_api_key)],
    server_url: str = Query(
        default="https://api.snipara.com",
        description="Base URL of the Snipara API server",
    ),
) -> dict:
    """Get all-in-one setup instructions for creating a Custom GPT."""
    await _validate_gpt_request(project_slug, api_key)

    return {
        "project_slug": project_slug,
        "system_prompt": get_gpt_system_prompt(project_slug, server_url),
        "openapi_spec_url": f"{server_url}/v1/gpt/{project_slug}/openapi",
        "auth_config": {
            "type": "API Key",
            "header": "X-API-Key",
            "description": "Paste your Snipara API key (starts with rlm_) into GPT Builder's authentication settings.",
        },
        "setup_steps": [
            "1. Go to https://chat.openai.com/gpts/editor",
            "2. Click 'Create a GPT' or edit an existing one",
            "3. In the 'Configure' tab, paste the system_prompt into 'Instructions'",
            "4. Under 'Actions', click 'Create new action'",
            f"5. Paste the OpenAPI spec from: {server_url}/v1/gpt/{project_slug}/openapi",
            "6. Under 'Authentication', select 'API Key' with header name 'X-API-Key'",
            "7. Paste your Snipara API key (rlm_...) as the API key value",
            "8. Save and test by asking a question about your docs",
        ],
    }


# ============ SYSTEM PROMPT TEMPLATE ============


def get_gpt_system_prompt(project_slug: str, api_url: str = "https://api.snipara.com") -> str:
    """
    Generate system prompt instructions for a Custom GPT.

    This includes:
    - How to use the Snipara query/search actions
    - RLM-Runtime install and usage commands
    - Best practices for answering with documentation context

    Args:
        project_slug: The project slug for the Custom GPT
        api_url: The Snipara API base URL

    Returns:
        System prompt string for GPT Builder
    """
    return f"""You are a documentation assistant for the "{project_slug}" project powered by Snipara.

## How to Answer Questions

1. **Always query documentation first** - Before answering any question about the project, use the `queryDocumentation` action to retrieve relevant context.
2. **Use search for specific lookups** - When looking for specific function names, patterns, or keywords, use `searchDocumentation`.
3. **Cite your sources** - Reference the file paths and section titles from the response.
4. **Stay within context** - Only answer based on the documentation returned. If no relevant docs are found, say so.

## Available Actions

- **queryDocumentation** - Ask a question, get optimized documentation context
- **searchDocumentation** - Search for regex patterns in the docs
- **getProjectInfo** - Check what documentation is available
- **storeMemory** - Save facts, decisions, and learnings for later
- **recallMemories** - Retrieve previously stored memories by semantic search
- **getSharedContext** - Get team coding standards and best practices

## Safe Code Execution with RLM-Runtime

When users need to run or test code from the documentation, recommend RLM-Runtime for safe, isolated execution:

```bash
# Install RLM-Runtime
pip install rlm-runtime[all]

# Initialize configuration
rlm init

# Run code safely in Docker isolation
rlm run --env docker "<task description>"

# Run locally (for trusted code only)
rlm run "<task description>"

# Launch trajectory visualization dashboard
rlm visualize
```

**Key RLM-Runtime features:**
- Docker-isolated execution for safety
- Multi-backend support (Anthropic, OpenAI, LiteLLM)
- Trajectory logging and visualization
- Configurable depth and iteration limits

## Tips

- For broad questions, use `queryDocumentation` with max_tokens=6000-8000
- For targeted lookups, use `searchDocumentation` with specific patterns
- Always check `getProjectInfo` first if unsure what docs are available
"""


# ============================================================
# OAuth Bearer endpoints for GPT Store (/v1/gpt/me/*)
# ============================================================
# These mirror the per-project endpoints above but use OAuth
# Bearer token auth. The project is resolved from the token.
# ============================================================


@router.post(
    "/me/query",
    response_model=GPTQueryResponse,
    summary="Query documentation (OAuth)",
    description=(
        "Query your project documentation with a natural language question. "
        "Your project is determined from your OAuth sign-in. "
        "Returns optimized, relevant context from your docs within the token budget."
    ),
    tags=["GPT Store"],
)
async def gpt_me_query(
    request: GPTQueryRequest,
    token: Annotated[str, Depends(get_bearer_token)],
) -> GPTQueryResponse:
    """Query documentation via OAuth Bearer token."""
    start_time = time.perf_counter()

    auth_info, project, plan, project_settings = await _validate_gpt_oauth_request(token)

    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=auth_info.get("user_id"),
            access_level=auth_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_CONTEXT_QUERY,
            {
                "query": request.query,
                "max_tokens": request.max_tokens,
                "include_metadata": True,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_me_query",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data
        sections = data.get("sections", []) if isinstance(data, dict) else []

        context_parts = []
        sources = []
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            file_path = section.get("file", "")
            lines = section.get("lines", (0, 0))
            relevance = section.get("relevance_score", 0)

            context_parts.append(f"## {title}\n{content}")
            sources.append({
                "title": title,
                "file": file_path,
                "lines": lines,
                "relevance": round(relevance, 3),
            })

        context_text = "\n\n".join(context_parts) if context_parts else "No relevant documentation found for this query."
        tokens_used = data.get("total_tokens", 0) if isinstance(data, dict) else 0

        return GPTQueryResponse(
            context=context_text,
            sources=sources,
            tokens_used=tokens_used,
            query=request.query,
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_me_query",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT me/query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your query. Please try again.",
        )


@router.post(
    "/me/search",
    response_model=GPTSearchResponse,
    summary="Search documentation (OAuth)",
    description=(
        "Search your project documentation using a regex pattern. "
        "Your project is determined from your OAuth sign-in."
    ),
    tags=["GPT Store"],
)
async def gpt_me_search(
    request: GPTSearchRequest,
    token: Annotated[str, Depends(get_bearer_token)],
) -> GPTSearchResponse:
    """Search documentation via OAuth Bearer token."""
    start_time = time.perf_counter()

    auth_info, project, plan, project_settings = await _validate_gpt_oauth_request(token)

    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=auth_info.get("user_id"),
            access_level=auth_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_SEARCH,
            {
                "pattern": request.pattern,
                "max_results": request.max_results,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_me_search",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data
        matches = []
        if isinstance(data, dict):
            for match in data.get("matches", []):
                matches.append({
                    "content": match.get("content", ""),
                    "file": match.get("file", ""),
                    "line": match.get("line", 0),
                    "context": match.get("context", ""),
                })

        return GPTSearchResponse(
            matches=matches,
            pattern=request.pattern,
            total_matches=len(matches),
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_me_search",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT me/search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your search. Please try again.",
        )


@router.get(
    "/me/info",
    response_model=GPTInfoResponse,
    summary="Get project info (OAuth)",
    description=(
        "Get information about your documentation project including "
        "file count, section count, and available actions. "
        "Your project is determined from your OAuth sign-in."
    ),
    tags=["GPT Store"],
)
async def gpt_me_info(
    token: Annotated[str, Depends(get_bearer_token)],
) -> GPTInfoResponse:
    """Get project info via OAuth Bearer token."""
    start_time = time.perf_counter()

    auth_info, project, plan, project_settings = await _validate_gpt_oauth_request(token)

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=auth_info.get("user_id"),
            access_level=auth_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(ToolName.RLM_STATS, {})

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_me_info",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}

        return GPTInfoResponse(
            project_slug=project.slug,
            files_loaded=data.get("files_loaded", 0),
            total_lines=data.get("total_lines", 0),
            total_sections=data.get("sections", 0),
            available_actions=["query", "search", "info", "remember", "recall", "shared-context"],
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_me_info",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT me/info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred fetching project info. Please try again.",
        )


@router.post(
    "/me/remember",
    response_model=GPTRememberResponse,
    summary="Store a memory (OAuth)",
    description=(
        "Store a fact, decision, learning, preference, or context for later recall. "
        "Your project is determined from your OAuth sign-in."
    ),
    tags=["GPT Store"],
)
async def gpt_me_remember(
    request: GPTRememberRequest,
    token: Annotated[str, Depends(get_bearer_token)],
) -> GPTRememberResponse:
    """Store a memory via OAuth Bearer token."""
    start_time = time.perf_counter()

    auth_info, project, plan, project_settings = await _validate_gpt_oauth_request(token)

    if plan == Plan.FREE:
        raise HTTPException(
            status_code=403,
            detail="Memory features require a paid plan. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=auth_info.get("user_id"),
            access_level=auth_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_REMEMBER,
            {
                "content": request.content,
                "type": request.type,
                "category": request.category,
                "scope": "project",
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_me_remember",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}

        return GPTRememberResponse(
            success=True,
            memory_id=data.get("memory_id", ""),
            message=data.get("message", "Memory stored successfully"),
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_me_remember",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT me/remember error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred storing the memory. Please try again.",
        )


@router.post(
    "/me/recall",
    response_model=GPTRecallResponse,
    summary="Recall memories (OAuth)",
    description=(
        "Semantically recall relevant memories based on a query. "
        "Your project is determined from your OAuth sign-in."
    ),
    tags=["GPT Store"],
)
async def gpt_me_recall(
    request: GPTRecallRequest,
    token: Annotated[str, Depends(get_bearer_token)],
) -> GPTRecallResponse:
    """Recall memories via OAuth Bearer token."""
    start_time = time.perf_counter()

    auth_info, project, plan, project_settings = await _validate_gpt_oauth_request(token)

    if plan == Plan.FREE:
        raise HTTPException(
            status_code=403,
            detail="Memory features require a paid plan. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=auth_info.get("user_id"),
            access_level=auth_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_RECALL,
            {
                "query": request.query,
                "limit": request.limit,
                "type": request.type,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_me_recall",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}
        memories = data.get("memories", [])

        return GPTRecallResponse(
            memories=memories,
            query=request.query,
            total=len(memories),
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_me_recall",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT me/recall error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred recalling memories. Please try again.",
        )


@router.get(
    "/me/shared-context",
    response_model=GPTSharedContextResponse,
    summary="Get team shared context (OAuth)",
    description=(
        "Get merged context from team shared collections. "
        "Your project is determined from your OAuth sign-in."
    ),
    tags=["GPT Store"],
)
async def gpt_me_shared_context(
    token: Annotated[str, Depends(get_bearer_token)],
    max_tokens: int = Query(
        default=4000,
        ge=100,
        le=16000,
        description="Token budget for shared context",
    ),
) -> GPTSharedContextResponse:
    """Get team shared context via OAuth Bearer token."""
    start_time = time.perf_counter()

    auth_info, project, plan, project_settings = await _validate_gpt_oauth_request(token)

    if plan not in (Plan.TEAM, Plan.ENTERPRISE):
        raise HTTPException(
            status_code=403,
            detail="Shared context requires Team plan or higher. Upgrade at snipara.com/pricing.",
        )

    try:
        engine = RLMEngine(
            project.id,
            plan=plan,
            settings=project_settings,
            user_id=auth_info.get("user_id"),
            access_level=auth_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(
            ToolName.RLM_SHARED_CONTEXT,
            {
                "max_tokens": max_tokens,
                "include_content": True,
            },
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        await track_usage(
            project_id=project.id,
            tool="gpt_me_shared_context",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        data = result.data if isinstance(result.data, dict) else {}
        documents = data.get("documents", [])
        categories = list({doc.get("category", "") for doc in documents if doc.get("category")})

        return GPTSharedContextResponse(
            documents=documents,
            total_tokens=data.get("total_tokens", 0),
            categories=categories,
        )

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        await track_usage(
            project_id=project.id,
            tool="gpt_me_shared_context",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )
        logger.error(f"GPT me/shared-context error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred fetching shared context. Please try again.",
        )


# ============ PROJECT-AGNOSTIC OPENAPI SPEC (GPT Store) ============


@router.get(
    "/me/openapi",
    summary="OpenAPI spec for GPT Store (OAuth)",
    description=(
        "Returns the OpenAPI 3.1 specification for the public Snipara GPT. "
        "Uses OAuth Bearer authentication — no API key needed. "
        "This spec is used by the GPT Store listing."
    ),
    tags=["GPT Store"],
)
async def gpt_me_openapi_spec(
    server_url: str = Query(
        default="https://api.snipara.com",
        description="Base URL of the Snipara API server",
    ),
) -> dict:
    """Generate project-agnostic OpenAPI 3.1 spec for the GPT Store."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Snipara - Documentation Intelligence",
            "description": (
                "Query, search, and manage your project documentation on Snipara. "
                "Sign in with your Snipara account to access your project. "
                "Use query for documentation context, search for patterns, "
                "remember/recall for persistent memory, and shared-context for team standards."
            ),
            "version": "1.0.0",
        },
        "servers": [
            {"url": server_url, "description": "Snipara API"},
        ],
        "paths": {
            "/v1/gpt/me/query": {
                "post": {
                    "operationId": "queryDocumentation",
                    "summary": "Query documentation with a natural language question",
                    "description": (
                        "Ask a question about your project documentation. "
                        "Returns relevant, optimized context within the token budget. "
                        "Your project is determined from your sign-in."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The question to ask about the documentation",
                                        },
                                        "max_tokens": {
                                            "type": "integer",
                                            "default": 4000,
                                            "minimum": 100,
                                            "maximum": 16000,
                                            "description": "Maximum tokens to return",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Documentation context",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "context": {
                                                "type": "string",
                                                "description": "Relevant documentation context",
                                            },
                                            "sources": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "title": {"type": "string"},
                                                        "file": {"type": "string"},
                                                        "relevance": {"type": "number"},
                                                    },
                                                },
                                                "description": "Source sections",
                                            },
                                            "tokens_used": {
                                                "type": "integer",
                                                "description": "Tokens used",
                                            },
                                            "query": {
                                                "type": "string",
                                                "description": "Original query",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/gpt/me/search": {
                "post": {
                    "operationId": "searchDocumentation",
                    "summary": "Search documentation with a regex pattern",
                    "description": (
                        "Search for specific patterns, function names, or keywords "
                        "in your project documentation using regex."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["pattern"],
                                    "properties": {
                                        "pattern": {
                                            "type": "string",
                                            "description": "Regex pattern to search for",
                                        },
                                        "max_results": {
                                            "type": "integer",
                                            "default": 20,
                                            "minimum": 1,
                                            "maximum": 50,
                                            "description": "Maximum results to return",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "matches": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "content": {"type": "string"},
                                                        "file": {"type": "string"},
                                                        "line": {"type": "integer"},
                                                    },
                                                },
                                            },
                                            "pattern": {"type": "string"},
                                            "total_matches": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/gpt/me/info": {
                "get": {
                    "operationId": "getProjectInfo",
                    "summary": "Get project documentation info",
                    "description": (
                        "Get information about your documentation project including "
                        "file count, section count, and available actions."
                    ),
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "Project information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "project_slug": {"type": "string"},
                                            "files_loaded": {"type": "integer"},
                                            "total_lines": {"type": "integer"},
                                            "total_sections": {"type": "integer"},
                                            "available_actions": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/gpt/me/remember": {
                "post": {
                    "operationId": "storeMemory",
                    "summary": "Store a memory for later recall",
                    "description": (
                        "Store a fact, decision, learning, preference, or context. "
                        "Memories persist across sessions and can be recalled semantically."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["content"],
                                    "properties": {
                                        "content": {
                                            "type": "string",
                                            "description": "The memory content to store",
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                                            "default": "fact",
                                            "description": "Memory type",
                                        },
                                        "category": {
                                            "type": "string",
                                            "description": "Optional grouping category",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Memory stored",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "memory_id": {"type": "string"},
                                            "message": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/gpt/me/recall": {
                "post": {
                    "operationId": "recallMemories",
                    "summary": "Recall stored memories",
                    "description": (
                        "Semantically search stored memories. "
                        "Returns relevant facts, decisions, and learnings."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["query"],
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Search query for memories",
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "default": 5,
                                            "minimum": 1,
                                            "maximum": 20,
                                            "description": "Max memories to return",
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": ["fact", "decision", "learning", "preference", "todo", "context"],
                                            "description": "Filter by memory type",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Matching memories",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "memories": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "content": {"type": "string"},
                                                        "type": {"type": "string"},
                                                        "relevance": {"type": "number"},
                                                    },
                                                },
                                            },
                                            "query": {"type": "string"},
                                            "total": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/v1/gpt/me/shared-context": {
                "get": {
                    "operationId": "getSharedContext",
                    "summary": "Get team shared context and standards",
                    "description": (
                        "Get merged context from team shared collections: "
                        "coding standards, best practices, and guidelines."
                    ),
                    "parameters": [
                        {
                            "name": "max_tokens",
                            "in": "query",
                            "schema": {"type": "integer", "default": 4000},
                            "description": "Token budget",
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "Shared context documents",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "documents": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "title": {"type": "string"},
                                                        "category": {"type": "string"},
                                                        "content": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "total_tokens": {"type": "integer"},
                                            "categories": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "oauth2",
                    "flows": {
                        "authorizationCode": {
                            "authorizationUrl": "https://snipara.com/api/oauth/authorize",
                            "tokenUrl": "https://snipara.com/api/oauth/token",
                            "scopes": {
                                "mcp:read": "Read documentation and context",
                                "mcp:write": "Write memories and context",
                            },
                        },
                    },
                },
            },
        },
        "security": [{"BearerAuth": ["mcp:read", "mcp:write"]}],
    }
