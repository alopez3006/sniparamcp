"""FastAPI MCP Server for RLM SaaS."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from . import __version__
from .auth import (
    check_team_key_project_access,
    get_effective_plan,
    get_project_settings,
    get_project_with_team,
    get_team_by_slug_or_id,
    validate_api_key,
    validate_oauth_token,
    validate_team_api_key,
)
from .config import settings
from .db import close_db, get_db
from .mcp_transport import router as mcp_router
from .models import (
    HealthResponse,
    LimitsInfo,
    MCPRequest,
    MCPResponse,
    MultiProjectQueryParams,
    Plan,
    ReadyResponse,
    ToolName,
    UsageInfo,
)
from .rlm_engine import RLMEngine, count_tokens
from .services.agent_memory import semantic_recall, store_memory
from .usage import (
    check_ip_rate_limit,
    check_rate_limit,
    check_usage_limits,
    close_redis,
    get_usage_stats,
    is_scan_blocked,
    log_security_event,
    record_access_denial,
    track_usage,
)

logger = logging.getLogger(__name__)

# ============ SENTRY INITIALIZATION ============


def _filter_sentry_event(event: dict) -> dict:
    """Remove sensitive data from Sentry events."""
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        for key in ["authorization", "x-api-key"]:
            if key in headers:
                headers[key] = "[REDACTED]"
    return event


# Initialize Sentry if DSN is configured
if settings.sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=0.1 if settings.environment == "production" else 1.0,
            integrations=[
                FastApiIntegration(),
                StarletteIntegration(),
            ],
            before_send=lambda event, hint: _filter_sentry_event(event),
        )
        logger.info("Sentry error tracking initialized")
    except ImportError:
        logger.warning("Sentry DSN configured but sentry-sdk not installed")
else:
    logger.debug("Sentry DSN not configured - error tracking disabled")


# ============ SECURITY HELPERS ============


def sanitize_error_message(error: Exception) -> str:
    """
    Sanitize error messages to prevent information disclosure.

    Returns a generic message for unexpected errors while preserving
    useful information for known error types.
    """
    error_str = str(error)

    # Known safe error patterns that can be returned to client
    safe_patterns = [
        "Invalid API key",
        "Project not found",
        "Rate limit exceeded",
        "Monthly usage limit exceeded",
        "Invalid tool name",
        "Invalid regex pattern",
        "No documentation loaded",
        "Unknown tool",
        "Invalid parameter",
        "Token budget",
        "Plan does not support",
    ]

    for pattern in safe_patterns:
        if pattern.lower() in error_str.lower():
            return error_str

    # Log the actual error for debugging
    logger.error(f"Tool execution error: {error}", exc_info=True)

    # Return generic message for unknown errors
    return "An error occurred processing your request. Please try again."


# ============ SECURITY MIDDLEWARE ============


class SecurityHeadersMiddleware:
    """
    Add security headers to all responses.

    Uses pure ASGI middleware pattern instead of BaseHTTPMiddleware
    to avoid Content-Length mismatch issues with streaming responses.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request ID for tracing
        request_id = str(uuid4())

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Add security headers
                headers.append((b"x-request-id", request_id.encode()))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-xss-protection", b"1; mode=block"))

                # Add HSTS in production (non-debug mode)
                if not settings.debug:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )

                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class IPRateLimitMiddleware:
    """
    Secondary rate limiting layer by client IP address.

    Applies to all HTTP requests before reaching endpoint handlers.
    Uses X-Forwarded-For header (behind reverse proxy) or direct client address.
    Skips health check endpoint to avoid blocking monitoring.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip health/readiness check endpoints
        path = scope.get("path", "")
        if path in ("/health", "/ready"):
            await self.app(scope, receive, send)
            return

        # Extract client IP from X-Forwarded-For (behind proxy) or direct connection
        client_ip = None
        headers = dict(scope.get("headers", []))
        forwarded_for = headers.get(b"x-forwarded-for")
        if forwarded_for:
            # First IP in X-Forwarded-For is the original client
            client_ip = forwarded_for.decode().split(",")[0].strip()
        elif scope.get("client"):
            client_ip = scope["client"][0]

        if client_ip:
            rate_ok = await check_ip_rate_limit(client_ip)
            if not rate_ok:
                response_body = json.dumps({
                    "detail": f"IP rate limit exceeded: {settings.ip_rate_limit_requests} requests per minute"
                }).encode()
                await send({
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(response_body)).encode()),
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": response_body,
                })
                return

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info(f"Starting RLM MCP Server v{__version__}")

    # Validate CORS configuration in production
    if not settings.debug and settings.cors_allowed_origins == "*":
        logger.warning(
            "SECURITY WARNING: CORS is configured to allow all origins ('*'). "
            "Set CORS_ALLOWED_ORIGINS to specific domains in production."
        )

    await get_db()  # Initialize database connection

    # Pre-load embedding models to avoid cold-start blocking workers
    # Primary (bge-large) for pgvector + Light (bge-small) for on-the-fly fallback
    from .services.embeddings import EmbeddingsService

    try:
        EmbeddingsService.preload_all()
    except Exception as e:
        logger.warning(f"Embedding model preload failed (will retry on first use): {e}")

    yield
    # Shutdown
    await close_db()
    await close_redis()


app = FastAPI(
    title="RLM MCP Server",
    description="Hosted MCP endpoint for RLM SaaS - Context-efficient documentation queries",
    version=__version__,
    lifespan=lifespan,
)

# IP-based rate limiting middleware (applied before other middleware)
app.add_middleware(IPRateLimitMiddleware)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware - use configured origins instead of wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Mount MCP Streamable HTTP transport
app.include_router(mcp_router)


# ============ DEPENDENCY INJECTION ============


async def get_api_key(
    x_api_key: Annotated[str, Header(alias="X-API-Key")],
) -> str:
    """Extract API key from header."""
    return x_api_key


def get_client_ip(request: Request) -> str | None:
    """Extract client IP from X-Forwarded-For header or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


async def validate_and_rate_limit(
    project_id: str,
    api_key: str,
    client_ip: str | None = None,
) -> tuple[dict, any, Plan, dict | None]:
    """
    Common validation logic for all endpoints.
    Validates API key or OAuth token, gets project, checks rate limit, and fetches settings.

    Supports both:
    - OAuth tokens (snipara_at_...)
    - API keys (rlm_...)

    Returns:
        Tuple of (auth_info, project, plan, project_settings)

    Raises:
        HTTPException on validation failure
    """
    # 0. Anti-scan: check if this key prefix is blocked
    key_prefix = api_key[:12]
    if await is_scan_blocked(key_prefix):
        log_security_event("scan.blocked", "api_key", key_prefix, key_prefix)
        raise HTTPException(status_code=429, detail="Too many failed requests. Try again later.")

    # 1. Validate auth (OAuth token or API key)
    auth_info = None

    # Check if it's an OAuth token
    if api_key.startswith("snipara_at_"):
        auth_info = await validate_oauth_token(api_key, project_id)
        if not auth_info:
            raise HTTPException(status_code=401, detail="Invalid or expired OAuth token. Re-authenticate at https://snipara.com/dashboard or run /snipara:quickstart")
    else:
        # Fall back to API key validation
        auth_info = await validate_api_key(api_key, project_id)
        if not auth_info:
            raise HTTPException(status_code=401, detail="Invalid API key. Get a free key at https://snipara.com/dashboard (100 queries/month, no credit card)")

    # 2. Check for access denial (team keys with NONE access level)
    if auth_info.get("access_denied"):
        await record_access_denial(key_prefix, project_id)
        log_security_event(
            "access.denied", "project", project_id,
            auth_info.get("id", key_prefix),
            details={"reason": "team_key_no_access"},
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied to this project. Use rlm_request_access tool to request access.",
        )

    # 3. Get project with team subscription
    project = await get_project_with_team(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 4. Check rate limit (demo keys use per-IP limits)
    rate_ok = await check_rate_limit(auth_info["id"], client_ip=client_ip)
    if not rate_ok:
        log_security_event(
            "rate_limit.exceeded", "api_key", auth_info["id"],
            auth_info.get("user_id", auth_info["id"]),
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.rate_limit_requests} requests per minute",
        )

    # 5. Determine plan (considers PRO boost for FREE users)
    plan = get_effective_plan(project.team.subscription if project.team else None)

    # 6. Get project automation settings (from dashboard)
    project_settings = await get_project_settings(project_id)

    return auth_info, project, plan, project_settings


async def validate_team_and_rate_limit(
    team_slug_or_id: str,
    api_key: str,
    client_ip: str | None = None,
) -> tuple[dict, any, Plan]:
    """
    Validate team API key, resolve team, and check rate limits.

    Returns:
        Tuple of (api_key_info, team, plan)
    """
    # Anti-scan check
    key_prefix = api_key[:12]
    if await is_scan_blocked(key_prefix):
        log_security_event("scan.blocked", "api_key", key_prefix, key_prefix)
        raise HTTPException(status_code=429, detail="Too many failed requests. Try again later.")

    team = await get_team_by_slug_or_id(team_slug_or_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    api_key_info = await validate_team_api_key(api_key, team.id)
    if not api_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    rate_ok = await check_rate_limit(api_key_info["id"], client_ip=client_ip)
    if not rate_ok:
        log_security_event(
            "rate_limit.exceeded", "api_key", api_key_info["id"],
            api_key_info.get("user_id", api_key_info["id"]),
            team_id=team.id,
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.rate_limit_requests} requests per minute",
        )

    plan = get_effective_plan(team.subscription)

    return api_key_info, team, plan


async def execute_multi_project_query(
    team: any,
    plan: Plan,
    params: dict,
    user_id: str | None = None,
) -> tuple[dict, int, int]:
    """
    Execute a multi-project query for a team.

    Args:
        team: Team object with projects
        plan: Subscription plan
        params: Query parameters
        user_id: Optional user ID for per-project ACL checks

    Returns:
        Tuple of (result_payload, total_input_tokens, total_output_tokens)
    """
    try:
        parsed = MultiProjectQueryParams.model_validate(params)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    projects = list(team.projects or [])
    include_ids = set(parsed.project_ids)
    exclude_ids = set(parsed.exclude_project_ids)

    if include_ids:
        projects = [
            project
            for project in projects
            if project.id in include_ids or project.slug in include_ids
        ]

    if exclude_ids:
        projects = [
            project
            for project in projects
            if project.id not in exclude_ids and project.slug not in exclude_ids
        ]

    if not projects:
        empty_result = {
            "query": parsed.query,
            "max_tokens": parsed.max_tokens,
            "per_project_limit": parsed.per_project_limit,
            "search_mode": parsed.search_mode.value,
            "projects_queried": 0,
            "projects_skipped": 0,
            "results": [],
        }
        return empty_result, count_tokens(parsed.query), 0

    per_project_budget = max(1, parsed.max_tokens // len(projects))

    async def execute_project(project: any) -> dict:
        # Per-project ACL check (when user_id is available)
        project_access_level = "EDITOR"
        if user_id:
            try:
                access_level_result, _ = await check_team_key_project_access(
                    user_id, project.id, team.id
                )
                if access_level_result == "NONE":
                    log_security_event(
                        "multi_project.access_denied", "project", project.id,
                        user_id, team_id=team.id,
                        details={"project_slug": project.slug},
                    )
                    return {
                        "project_id": project.id,
                        "project_slug": project.slug,
                        "success": False,
                        "skipped": True,
                        "error": "Access denied to this project",
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                project_access_level = access_level_result
            except Exception as e:
                logger.debug(f"ACL check failed for {project.slug}, defaulting to EDITOR: {e}")

        limits = await check_usage_limits(project.id, plan)
        if limits.exceeded:
            return {
                "project_id": project.id,
                "project_slug": project.slug,
                "success": False,
                "skipped": True,
                "error": f"Monthly usage limit exceeded: {limits.current}/{limits.max}",
                "input_tokens": 0,
                "output_tokens": 0,
            }

        project_settings = await get_project_settings(project.id)
        tool_params = {
            "query": parsed.query,
            "max_tokens": per_project_budget,
            "search_mode": parsed.search_mode.value,
            "include_metadata": parsed.include_metadata,
            "prefer_summaries": parsed.prefer_summaries,
        }

        start_time = time.perf_counter()

        try:
            engine = RLMEngine(
                project.id, plan=plan, settings=project_settings,
                user_id=user_id, access_level=project_access_level,
            )
            result = await engine.execute(ToolName.RLM_CONTEXT_QUERY, tool_params)

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            await track_usage(
                project_id=project.id,
                tool=ToolName.RLM_MULTI_PROJECT_QUERY.value,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency_ms,
                success=True,
            )

            result_data = result.data
            if isinstance(result_data, dict) and "sections" in result_data:
                result_data["sections"] = result_data["sections"][: parsed.per_project_limit]

            return {
                "project_id": project.id,
                "project_slug": project.slug,
                "success": True,
                "skipped": False,
                "result": result_data,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            await track_usage(
                project_id=project.id,
                tool=ToolName.RLM_MULTI_PROJECT_QUERY.value,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )
            return {
                "project_id": project.id,
                "project_slug": project.slug,
                "success": False,
                "skipped": False,
                "error": sanitize_error_message(e),
                "input_tokens": 0,
                "output_tokens": 0,
            }

    results = await asyncio.gather(*[execute_project(project) for project in projects])

    total_input_tokens = sum(item["input_tokens"] for item in results)
    total_output_tokens = sum(item["output_tokens"] for item in results)
    projects_queried = sum(1 for item in results if item["success"])
    projects_skipped = sum(1 for item in results if item.get("skipped"))

    payload = {
        "query": parsed.query,
        "max_tokens": parsed.max_tokens,
        "per_project_limit": parsed.per_project_limit,
        "search_mode": parsed.search_mode.value,
        "projects_queried": projects_queried,
        "projects_skipped": projects_skipped,
        "results": [
            {k: v for k, v in item.items() if k not in {"input_tokens", "output_tokens"}}
            for item in results
        ],
    }

    return payload, total_input_tokens, total_output_tokens


# ============ EXCEPTION HANDLERS ============


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "usage": {"latency_ms": 0},
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with sanitized error messages."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal server error occurred. Please try again.",
            "usage": {"latency_ms": 0},
        },
    )


# ============ HEALTH ENDPOINTS ============


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint (lightweight liveness check)."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check - verifies DB and embedding model are operational."""
    from .services.embeddings import EmbeddingsService

    checks: dict[str, bool] = {}
    all_ok = True

    # Check database connectivity
    try:
        db = await get_db()
        await db.query_raw("SELECT 1")
        checks["database"] = True
    except Exception:
        checks["database"] = False
        all_ok = False

    # Check embedding model is loaded
    checks["embedding_model"] = EmbeddingsService.get_instance().is_loaded()
    if not checks["embedding_model"]:
        all_ok = False

    response = ReadyResponse(
        status="ready" if all_ok else "not_ready",
        version=__version__,
        checks=checks,
    )
    return JSONResponse(
        content=response.model_dump(mode="json"),
        status_code=200 if all_ok else 503,
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RLM MCP Server",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


# ============ MCP ENDPOINTS ============


@app.post("/v1/{project_id}/mcp", response_model=MCPResponse, tags=["MCP"])
async def mcp_endpoint(
    project_id: str,
    request: MCPRequest,
    api_key: Annotated[str, Depends(get_api_key)],
    raw_request: Request,
) -> MCPResponse:
    """
    Execute an RLM MCP tool.

    This endpoint validates the API key, checks usage limits,
    executes the requested tool, and tracks usage.

    Args:
        project_id: The project ID
        request: The MCP request with tool and parameters
        api_key: API key from X-API-Key header
        raw_request: The raw FastAPI request (for client IP)

    Returns:
        MCPResponse with result or error
    """
    start_time = time.perf_counter()

    # Validate API key, project, rate limit, and get settings
    api_key_info, project, plan, project_settings = await validate_and_rate_limit(
        project_id, api_key, client_ip=get_client_ip(raw_request)
    )

    # Check usage limits
    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade your plan to continue.",
        )

    # Execute the tool with project settings from dashboard
    try:
        engine = RLMEngine(
            project.id, plan=plan, settings=project_settings,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        )
        result = await engine.execute(request.tool, request.params)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Track usage
        await track_usage(
            project_id=project.id,
            tool=request.tool.value,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        return MCPResponse(
            success=True,
            result=result.data,
            usage=UsageInfo(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=latency_ms,
            ),
        )

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Track failed request (log full error internally)
        await track_usage(
            project_id=project.id,
            tool=request.tool.value,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),  # Full error for internal logging
        )

        # Return sanitized error to client
        return MCPResponse(
            success=False,
            error=sanitize_error_message(e),
            usage=UsageInfo(latency_ms=latency_ms),
        )


@app.post("/v1/team/{team_slug}/mcp", response_model=MCPResponse, tags=["MCP"])
async def team_mcp_endpoint(
    team_slug: str,
    request: MCPRequest,
    api_key: Annotated[str, Depends(get_api_key)],
) -> MCPResponse:
    """
    Execute team-scoped MCP tools.

    This endpoint only allows rlm_multi_project_query with a team API key.
    """
    start_time = time.perf_counter()

    api_key_info, team, plan = await validate_team_and_rate_limit(team_slug, api_key)

    if request.tool != ToolName.RLM_MULTI_PROJECT_QUERY:
        raise HTTPException(status_code=400, detail="Invalid tool for team API key")

    try:
        result_payload, input_tokens, output_tokens = await execute_multi_project_query(
            team, plan, request.params,
            user_id=api_key_info.get("user_id"),
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        return MCPResponse(
            success=True,
            result=result_payload,
            usage=UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return MCPResponse(
            success=False,
            error=sanitize_error_message(e),
            usage=UsageInfo(latency_ms=latency_ms),
        )


# ============ TEAM MCP TRANSPORT (JSON-RPC) ============


def _jsonrpc_response(id: any, result: dict) -> dict:
    """Create a JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: any, code: int, message: str) -> dict:
    """Create a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# Tool definition for team endpoint (only rlm_multi_project_query)
TEAM_TOOL_DEFINITION = {
    "name": "rlm_multi_project_query",
    "description": "Query across all projects in a team. Returns ranked context from multiple documentation sets.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question to answer"},
            "max_tokens": {"type": "integer", "default": 16000, "description": "Total token budget across all projects"},
            "per_project_limit": {"type": "integer", "default": 10, "description": "Max sections per project"},
            "project_ids": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Filter to specific projects (empty = all)"},
            "exclude_project_ids": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Projects to exclude"},
        },
        "required": ["query"],
    },
}


@app.post("/mcp/team/{team_id}", tags=["MCP Transport"])
async def team_mcp_transport_endpoint(
    team_id: str,
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
):
    """
    Team MCP Streamable HTTP endpoint (JSON-RPC format).

    This endpoint supports the MCP protocol for team-scoped queries.
    Only the rlm_multi_project_query tool is available.

    Config example (Claude Code):
    ```json
    {"mcpServers": {"snipara-team": {"type": "http", "url": "https://api.snipara.com/mcp/team/{team_id}", "headers": {"X-API-Key": "rlm_team_..."}}}}
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

    # Validate team and API key
    team = await get_team_by_slug_or_id(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    api_key_info = await validate_team_api_key(api_key, team.id)
    if not api_key_info:
        log_security_event(
            "auth.failed", "team", team_id, api_key[:12],
            team_id=team.id,
            details={"reason": "invalid_team_api_key"},
        )
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not await check_rate_limit(api_key_info["id"]):
        log_security_event(
            "rate_limit.exceeded", "api_key", api_key_info["id"],
            api_key_info.get("user_id", api_key_info["id"]),
            team_id=team.id,
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    plan = get_effective_plan(team.subscription)

    # Parse JSON-RPC request
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    # Extract user_id for ACL checks
    team_user_id = api_key_info.get("user_id")

    # Handle batch requests
    if isinstance(body, list):
        responses = []
        for req in body:
            resp = await _handle_team_request(req, team, plan, user_id=team_user_id)
            if resp:  # Skip notifications (no id)
                responses.append(resp)
        return JSONResponse(responses)

    # Handle single request
    response = await _handle_team_request(body, team, plan, user_id=team_user_id)
    return JSONResponse(response) if response else Response(status_code=204)


async def _handle_team_request(body: dict, team: any, plan: Plan, user_id: str | None = None) -> dict | None:
    """Handle a single JSON-RPC request for team endpoint."""
    method = body.get("method")
    id = body.get("id")
    params = body.get("params", {})

    if id is None:  # Notification - no response
        return None

    if method == "initialize":
        return _jsonrpc_response(id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "snipara-team", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        })
    elif method == "tools/list":
        return _jsonrpc_response(id, {"tools": [TEAM_TOOL_DEFINITION]})
    elif method == "tools/call":
        return await _handle_team_call_tool(id, params, team, plan, user_id=user_id)
    elif method == "ping":
        return _jsonrpc_response(id, {})
    else:
        return _jsonrpc_error(id, -32601, f"Method not found: {method}")


async def _handle_team_call_tool(id: any, params: dict, team: any, plan: Plan, user_id: str | None = None) -> dict:
    """Handle MCP tools/call request for team endpoint."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name != "rlm_multi_project_query":
        return _jsonrpc_error(id, -32602, f"Tool not available on team endpoint: {tool_name}. Only rlm_multi_project_query is supported.")

    try:
        result_payload, input_tokens, output_tokens = await execute_multi_project_query(
            team, plan, arguments, user_id=user_id,
        )

        return _jsonrpc_response(id, {
            "content": [{"type": "text", "text": json.dumps(result_payload, indent=2, default=str)}],
        })
    except HTTPException as e:
        return _jsonrpc_error(id, -32000, e.detail)
    except Exception as e:
        return _jsonrpc_error(id, -32000, str(e))


@app.get("/v1/{project_id}/context", tags=["MCP"])
async def get_context(
    project_id: str,
    api_key: Annotated[str, Depends(get_api_key)],
):
    """
    Get the current session context for a project.

    Args:
        project_id: The project ID
        api_key: API key from X-API-Key header

    Returns:
        Current session context
    """
    # Validate API key, project, and rate limit
    api_key_info, project, _, _ = await validate_and_rate_limit(project_id, api_key)

    engine = RLMEngine(
        project.id,
        user_id=api_key_info.get("user_id"),
        access_level=api_key_info.get("access_level", "EDITOR"),
    )
    await engine.load_session_context()

    return {
        "project_id": project.id,
        "context": engine.session_context,
        "has_context": bool(engine.session_context),
    }


@app.get("/v1/{project_id}/limits", response_model=LimitsInfo, tags=["MCP"])
async def get_limits(
    project_id: str,
    api_key: Annotated[str, Depends(get_api_key)],
) -> LimitsInfo:
    """
    Get current usage limits for a project.

    Args:
        project_id: The project ID
        api_key: API key from X-API-Key header

    Returns:
        Current usage and limits
    """
    # Validate API key, project, and rate limit
    _, _, plan, _ = await validate_and_rate_limit(project_id, api_key)

    return await check_usage_limits(project_id, plan)


@app.get("/v1/{project_id}/stats", tags=["MCP"])
async def get_stats(
    project_id: str,
    api_key: Annotated[str, Depends(get_api_key)],
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
):
    """
    Get usage statistics for a project.

    Args:
        project_id: The project ID
        api_key: API key from X-API-Key header
        days: Number of days to look back (default: 30, max: 365)

    Returns:
        Usage statistics
    """
    # Validate API key, project, and rate limit
    _, _, _, _ = await validate_and_rate_limit(project_id, api_key)

    stats = await get_usage_stats(project_id, days)
    return {"project_id": project_id, **stats}


# ============ MEMORY REST API (Automation Hooks) ============


@app.get("/v1/{project_id}/memories/recall", tags=["Memories"])
async def recall_memories(
    project_id: str,
    api_key: Annotated[str, Depends(get_api_key)],
    query: str = Query(..., description="Search query for semantic recall"),
    type: str | None = Query(default=None, description="Filter by memory type"),
    category: str | None = Query(default=None, description="Filter by category"),
    limit: int = Query(default=10, ge=1, le=50, description="Max memories to return"),
    min_relevance: float = Query(default=0.3, ge=0, le=1, description="Minimum relevance"),
):
    """
    Recall memories semantically based on a query.

    Used by SessionStart hooks to inject relevant memories into new sessions.

    Args:
        project_id: The project ID
        query: Search query for semantic matching
        type: Filter by memory type (fact, decision, learning, preference, todo, context)
        category: Filter by category
        limit: Maximum memories to return
        min_relevance: Minimum relevance score (0-1)

    Returns:
        List of relevant memories with content and metadata
    """
    # Validate API key, project, and rate limit
    _, project, _, _ = await validate_and_rate_limit(project_id, api_key)

    # Use resolved project ID, not the slug from URL
    resolved_project_id = project.id

    result = await semantic_recall(
        project_id=resolved_project_id,
        query=query,
        memory_type=type,
        category=category,
        limit=limit,
        min_relevance=min_relevance,
    )

    return {
        "project_id": resolved_project_id,
        "query": query,
        "memories": result.get("memories", []),
        "total_searched": result.get("total_searched", 0),
        "timing_ms": result.get("timing_ms", 0),
    }


@app.post("/v1/{project_id}/memories", tags=["Memories"])
async def create_memory(
    project_id: str,
    api_key: Annotated[str, Depends(get_api_key)],
    request: Request,
):
    """
    Store a new memory for later recall.

    Used by PreCompact hooks or directly by Claude to persist learnings.

    Request body:
        content: str - The memory content
        type: str - Memory type (fact, decision, learning, preference, todo, context)
        category: str - Optional grouping category
        ttl_days: int - Days until expiration (null = permanent)
        source: str - What created this memory (e.g., "hook", "claude", "manual")

    Returns:
        Created memory with ID and metadata
    """
    # Validate API key, project, and rate limit
    _, project, _, _ = await validate_and_rate_limit(project_id, api_key)

    body = await request.json()

    # Use resolved project ID, not the slug from URL
    resolved_project_id = project.id

    result = await store_memory(
        project_id=resolved_project_id,
        content=body.get("content", ""),
        memory_type=body.get("type", "learning"),
        scope=body.get("scope", "project"),
        category=body.get("category"),
        ttl_days=body.get("ttl_days"),
        source=body.get("source", "hook"),
    )

    return {
        "project_id": resolved_project_id,
        "memory_id": result.get("memory_id"),
        "type": result.get("type"),
        "created": result.get("created", False),
        "message": result.get("message"),
    }


# ============ SSE ENDPOINTS (Continue.dev Integration) ============


async def sse_event_generator(
    project_id: str,
    tool: ToolName,
    params: dict,
    plan: Plan,
    user_id: str | None = None,
    access_level: str = "EDITOR",
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for MCP tool execution.

    Yields SSE-formatted events:
    - start: Tool execution started
    - result: Tool execution complete with result
    - error: Error occurred during execution
    """
    start_time = time.perf_counter()

    # Send start event
    yield f"data: {json.dumps({'type': 'start', 'tool': tool.value})}\n\n"

    try:
        # Execute the tool
        engine = RLMEngine(
            project_id, plan=plan,
            user_id=user_id, access_level=access_level
        )
        result = await engine.execute(tool, params)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Track usage
        await track_usage(
            project_id=project_id,
            tool=tool.value,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

        # Send result event
        yield f"data: {json.dumps({'type': 'result', 'success': True, 'result': result.data, 'usage': {'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens, 'latency_ms': latency_ms}})}\n\n"

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Track failed request
        await track_usage(
            project_id=project_id,
            tool=tool.value,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )

        # Send sanitized error event
        yield f"data: {json.dumps({'type': 'error', 'error': sanitize_error_message(e), 'usage': {'latency_ms': latency_ms}})}\n\n"

    # Send done event to signal stream end
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.get("/v1/{project_id}/mcp/sse", tags=["MCP", "SSE"])
async def mcp_sse_endpoint(
    project_id: str,
    api_key: Annotated[str, Depends(get_api_key)],
    tool: str = Query(..., description="Tool name to execute"),
    params: str = Query(default="{}", description="JSON-encoded parameters"),
):
    """
    Execute an RLM MCP tool via Server-Sent Events (SSE).

    This endpoint is designed for Continue.dev and other clients that
    support SSE transport. It streams the tool execution result.

    Args:
        project_id: The project ID
        api_key: API key from X-API-Key header
        tool: Tool name (e.g., rlm_ask, rlm_context_query)
        params: JSON-encoded parameters

    Returns:
        SSE stream with tool execution events
    """
    # Validate API key, project, and rate limit
    api_key_info, project, plan, _ = await validate_and_rate_limit(project_id, api_key)

    # Check usage limits
    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade your plan to continue.",
        )

    # Validate JSON payload size before parsing
    if len(params) > settings.max_json_payload_size:
        raise HTTPException(
            status_code=413,
            detail=f"JSON payload too large. Maximum size: {settings.max_json_payload_size} bytes",
        )

    # Parse tool name
    try:
        tool_name = ToolName(tool)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tool name: {tool}. Valid tools: {[t.value for t in ToolName]}",
        )

    # Parse params with error sanitization
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in params parameter",
        )

    # Return SSE stream
    return StreamingResponse(
        sse_event_generator(
            project.id, tool_name, parsed_params, plan,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.post("/v1/{project_id}/mcp/sse", tags=["MCP", "SSE"])
async def mcp_sse_endpoint_post(
    project_id: str,
    request: MCPRequest,
    api_key: Annotated[str, Depends(get_api_key)],
):
    """
    Execute an RLM MCP tool via Server-Sent Events (SSE) using POST.

    Alternative to GET for clients that prefer POST requests with JSON body.

    Args:
        project_id: The project ID
        request: The MCP request with tool and parameters
        api_key: API key from X-API-Key header

    Returns:
        SSE stream with tool execution events
    """
    # Validate API key, project, and rate limit
    api_key_info, project, plan, _ = await validate_and_rate_limit(project_id, api_key)

    # Check usage limits
    limits = await check_usage_limits(project.id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade your plan to continue.",
        )

    # Return SSE stream
    return StreamingResponse(
        sse_event_generator(
            project.id, request.tool, request.params, plan,
            user_id=api_key_info.get("user_id"),
            access_level=api_key_info.get("access_level", "EDITOR"),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ============ MAIN ============


def main():
    """Run the server with uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
