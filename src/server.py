"""FastAPI MCP Server for RLM SaaS."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, AsyncGenerator
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .auth import get_project_with_team, validate_api_key
from .config import settings
from .db import close_db, get_db
from .models import (
    HealthResponse,
    LimitsInfo,
    MCPRequest,
    MCPResponse,
    Plan,
    ToolName,
    UsageInfo,
)
from .rlm_engine import RLMEngine
from .usage import (
    check_rate_limit,
    check_usage_limits,
    close_redis,
    get_usage_stats,
    track_usage,
)
from .mcp_transport import router as mcp_router

logger = logging.getLogger(__name__)


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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        # Generate request ID for tracing
        request_id = str(uuid4())

        response = await call_next(request)

        # Add security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Add HSTS in production (non-debug mode)
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


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

# Security headers middleware (applied first)
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


async def validate_and_rate_limit(
    project_id: str,
    api_key: str,
) -> tuple[dict, any, Plan]:
    """
    Common validation logic for all endpoints.
    Validates API key, gets project, and checks rate limit.

    Returns:
        Tuple of (api_key_info, project, plan)

    Raises:
        HTTPException on validation failure
    """
    # 1. Validate API key
    api_key_info = await validate_api_key(api_key, project_id)
    if not api_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2. Get project with team subscription
    project = await get_project_with_team(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 3. Check rate limit
    rate_ok = await check_rate_limit(api_key_info["id"])
    if not rate_ok:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.rate_limit_requests} requests per minute",
        )

    # 4. Determine plan
    plan = Plan(project.team.subscription.plan if project.team.subscription else "FREE")

    return api_key_info, project, plan


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
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
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
) -> MCPResponse:
    """
    Execute an RLM MCP tool.

    This endpoint validates the API key, checks usage limits,
    executes the requested tool, and tracks usage.

    Args:
        project_id: The project ID
        request: The MCP request with tool and parameters
        api_key: API key from X-API-Key header

    Returns:
        MCPResponse with result or error
    """
    start_time = time.perf_counter()

    # Validate API key, project, and rate limit
    api_key_info, project, plan = await validate_and_rate_limit(project_id, api_key)

    # Check usage limits
    limits = await check_usage_limits(project_id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade your plan to continue.",
        )

    # Execute the tool
    try:
        engine = RLMEngine(project_id, plan=plan)
        result = await engine.execute(request.tool, request.params)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Track usage
        await track_usage(
            project_id=project_id,
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
            project_id=project_id,
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
    await validate_and_rate_limit(project_id, api_key)

    engine = RLMEngine(project_id)
    await engine.load_session_context()

    return {
        "project_id": project_id,
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
    _, _, plan = await validate_and_rate_limit(project_id, api_key)

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
    await validate_and_rate_limit(project_id, api_key)

    stats = await get_usage_stats(project_id, days)
    return {"project_id": project_id, **stats}


# ============ SSE ENDPOINTS (Continue.dev Integration) ============


async def sse_event_generator(
    project_id: str,
    tool: ToolName,
    params: dict,
    plan: Plan,
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
        engine = RLMEngine(project_id, plan=plan)
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
    _, _, plan = await validate_and_rate_limit(project_id, api_key)

    # Check usage limits
    limits = await check_usage_limits(project_id, plan)
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
        sse_event_generator(project_id, tool_name, parsed_params, plan),
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
    _, _, plan = await validate_and_rate_limit(project_id, api_key)

    # Check usage limits
    limits = await check_usage_limits(project_id, plan)
    if limits.exceeded:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly usage limit exceeded: {limits.current}/{limits.max} queries. Upgrade your plan to continue.",
        )

    # Return SSE stream
    return StreamingResponse(
        sse_event_generator(project_id, request.tool, request.params, plan),
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
