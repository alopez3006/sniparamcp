"""Usage tracking and rate limiting module."""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta

import redis.asyncio as redis

from .config import settings
from .db import get_db
from .models import LimitsInfo, Plan

logger = logging.getLogger(__name__)

# Redis client for rate limiting
_redis: redis.Redis | None = None
_redis_available: bool | None = None

# In-memory rate limiting fallback (per-process)
_local_rate_limits: dict[str, list[float]] = defaultdict(list)
_local_ip_rate_limits: dict[str, list[float]] = defaultdict(list)
_fallback_warning_logged: bool = False


async def get_redis() -> redis.Redis | None:
    """Get or create Redis client. Returns None if Redis is not configured."""
    global _redis, _redis_available

    # If we already know Redis is unavailable, skip
    if _redis_available is False:
        return None

    if _redis is None:
        if not settings.redis_url:
            _redis_available = False
            return None
        try:
            _redis = redis.from_url(settings.redis_url)
            # Test connection
            await _redis.ping()
            _redis_available = True
        except Exception as e:
            print(f"[Warning] Redis connection failed, rate limiting disabled: {e}")
            _redis_available = False
            _redis = None
            return None
    return _redis


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis, _redis_available
    if _redis is not None:
        await _redis.close()
        _redis = None
    _redis_available = None


def _is_demo_key(api_key_id: str) -> bool:
    """Check if an API key ID is a demo key (public, stricter limits)."""
    if not settings.demo_api_key_ids:
        return False
    demo_ids = {kid.strip() for kid in settings.demo_api_key_ids.split(",") if kid.strip()}
    return api_key_id in demo_ids


def _get_rate_limit_for_key(api_key_id: str) -> tuple[int, int]:
    """Return (max_requests, window_seconds) for the given key."""
    if _is_demo_key(api_key_id):
        return settings.demo_rate_limit_requests, settings.demo_rate_limit_window
    return settings.rate_limit_requests, settings.rate_limit_window


async def check_rate_limit(api_key_id: str, client_ip: str | None = None) -> bool:
    """
    Check if the API key has exceeded rate limits.

    Uses Redis when available, falls back to in-memory sliding window.
    Demo keys use stricter per-IP limits (configured via DEMO_API_KEY_IDS).

    Args:
        api_key_id: The API key ID
        client_ip: Optional client IP for per-IP demo rate limiting

    Returns:
        True if within limits, False if exceeded
    """
    global _fallback_warning_logged

    max_requests, window = _get_rate_limit_for_key(api_key_id)

    # Demo keys: rate limit per IP instead of per key (since key is public)
    is_demo = _is_demo_key(api_key_id)
    if is_demo and client_ip:
        rate_key_id = f"demo_ip:{client_ip}"
    else:
        rate_key_id = api_key_id

    r = await get_redis()

    # Fail-closed: reject when Redis unavailable in production
    if r is None:
        if settings.rate_limit_fail_closed:
            logger.error("Redis unavailable and rate_limit_fail_closed=True - rejecting request")
            return False
        return _check_rate_limit_memory(rate_key_id, max_requests, window)

    try:
        key = f"rate_limit:{rate_key_id}"

        # Get current count
        count = await r.get(key)
        if count is None:
            # First request, set counter with expiry
            await r.setex(key, window, 1)
            return True

        count = int(count)
        if count >= max_requests:
            label = f"demo IP {client_ip}" if is_demo else f"{api_key_id[:8]}..."
            logger.warning(f"Rate limit exceeded for {label}")
            return False

        # Increment counter
        await r.incr(key)
        return True
    except Exception as e:
        # Redis error - fail-closed or fall back to in-memory
        logger.error(f"Redis rate limit check failed: {e}")
        if settings.rate_limit_fail_closed:
            return False
        return _check_rate_limit_memory(rate_key_id, max_requests, window)


def _check_rate_limit_memory(
    rate_key_id: str,
    max_requests: int | None = None,
    window_seconds: int | None = None,
) -> bool:
    """
    In-memory rate limit check (fallback).

    Uses a sliding window algorithm. Note: This is per-process only and
    won't work correctly with multiple server instances.

    Args:
        rate_key_id: The rate limit key (API key ID or demo IP key)
        max_requests: Override for max requests (defaults to settings)
        window_seconds: Override for window (defaults to settings)

    Returns:
        True if within limits, False if exceeded
    """
    global _fallback_warning_logged

    if not _fallback_warning_logged:
        logger.warning(
            "Redis unavailable - using in-memory rate limiting. "
            "This is per-process only and won't work correctly "
            "with multiple server instances."
        )
        _fallback_warning_logged = True

    if max_requests is None:
        max_requests = settings.rate_limit_requests
    if window_seconds is None:
        window_seconds = settings.rate_limit_window

    now = time.time()
    window = _local_rate_limits[rate_key_id]

    # Remove expired entries (sliding window)
    window[:] = [t for t in window if now - t < window_seconds]

    if len(window) >= max_requests:
        logger.warning(f"Rate limit exceeded for {rate_key_id[:8]}... (in-memory)")
        return False

    window.append(now)
    return True


async def check_ip_rate_limit(client_ip: str) -> bool:
    """
    Check if a client IP has exceeded rate limits.

    Secondary rate limiting layer that prevents distributed attacks
    using multiple API keys from the same IP address.

    Args:
        client_ip: The client IP address

    Returns:
        True if within limits, False if exceeded
    """
    if not client_ip:
        return True  # Skip if IP not available

    r = await get_redis()

    if r is None:
        # In-memory fallback for IP rate limiting
        now = time.time()
        window = _local_ip_rate_limits[client_ip]
        window[:] = [t for t in window if now - t < settings.ip_rate_limit_window]
        if len(window) >= settings.ip_rate_limit_requests:
            logger.warning(f"IP rate limit exceeded for {client_ip}")
            return False
        window.append(now)
        return True

    try:
        key = f"ip_rate_limit:{client_ip}"
        count = await r.get(key)

        if count is None:
            await r.setex(key, settings.ip_rate_limit_window, 1)
            return True

        count = int(count)
        if count >= settings.ip_rate_limit_requests:
            logger.warning(f"IP rate limit exceeded for {client_ip}")
            return False

        await r.incr(key)
        return True
    except Exception as e:
        logger.error(f"Redis IP rate limit check failed: {e}")
        return True  # Fail open for IP checks (API key limit is primary)


async def check_usage_limits(project_id: str, plan: Plan) -> LimitsInfo:
    """
    Check if the project has exceeded monthly usage limits.

    Args:
        project_id: The project ID
        plan: The subscription plan

    Returns:
        LimitsInfo with current usage and limits
    """
    db = await get_db()

    # Get the start of the current month
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)

    # Count queries this month
    query_count = await db.query.count(
        where={
            "projectId": project_id,
            "createdAt": {"gte": month_start},
        }
    )

    # Get plan limit
    max_queries = settings.plan_limits.get(plan.value, 100)

    return LimitsInfo(
        current=query_count,
        max=max_queries,
        exceeded=max_queries != -1 and query_count >= max_queries,
        resets_at=next_month,
    )


async def track_usage(
    project_id: str,
    tool: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    success: bool,
    error: str | None = None,
) -> None:
    """
    Track a query for usage analytics and billing.

    Args:
        project_id: The project ID
        tool: The tool that was executed
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: Request latency in milliseconds
        success: Whether the request succeeded
        error: Error message if failed
    """
    db = await get_db()

    await db.query.create(
        data={
            "projectId": project_id,
            "tool": tool,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latencyMs": latency_ms,
            "success": success,
            "errorMessage": error,
        }
    )


async def get_usage_stats(project_id: str, days: int = 30) -> dict:
    """
    Get usage statistics for a project.

    Args:
        project_id: The project ID
        days: Number of days to look back

    Returns:
        Usage statistics dictionary
    """
    db = await get_db()
    since = datetime.utcnow() - timedelta(days=days)

    # Get query counts
    queries = await db.query.find_many(
        where={
            "projectId": project_id,
            "createdAt": {"gte": since},
        }
    )

    total_queries = len(queries)
    successful_queries = sum(1 for q in queries if q.success)
    total_input_tokens = sum(q.inputTokens for q in queries)
    total_output_tokens = sum(q.outputTokens for q in queries)
    avg_latency = sum(q.latencyMs for q in queries) / total_queries if total_queries > 0 else 0

    # Group by tool
    tool_counts: dict[str, int] = {}
    for q in queries:
        tool_counts[q.tool] = tool_counts.get(q.tool, 0) + 1

    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "failed_queries": total_queries - successful_queries,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_latency_ms": round(avg_latency, 2),
        "queries_by_tool": tool_counts,
        "period_days": days,
    }


# ============ SECURITY AUDIT LOGGING ============


async def _write_audit_log(
    action: str,
    entity_type: str,
    entity_id: str,
    actor_id: str,
    team_id: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """Write an audit log entry. Internal - never raises."""
    try:
        db = await get_db()
        await db.auditlog.create(
            data={
                "action": action,
                "entityType": entity_type,
                "entityId": entity_id,
                "actorId": actor_id,
                "teamId": team_id or actor_id,
                "details": details,
                "ipAddress": ip_address,
            }
        )
    except Exception as e:
        logger.debug(f"Audit log write failed (non-fatal): {e}")


def log_security_event(
    action: str,
    entity_type: str,
    entity_id: str,
    actor_id: str,
    team_id: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """
    Fire-and-forget security audit log entry.

    Never blocks the request or raises exceptions.

    Args:
        action: Event type (e.g., "access.denied", "rate_limit.exceeded", "auth.failed")
        entity_type: What was accessed (e.g., "project", "team", "api_key")
        entity_id: ID of the entity
        actor_id: Who performed the action (user ID, key ID, or key prefix)
        team_id: Optional team ID
        details: Optional JSON details (no content/queries - only metadata)
        ip_address: Optional client IP
    """
    try:
        asyncio.create_task(
            _write_audit_log(action, entity_type, entity_id, actor_id, team_id, details, ip_address)
        )
    except RuntimeError:
        # No running event loop (e.g., during shutdown)
        pass


# ============ ANTI-SCAN PROTECTION ============

# Thresholds
SCAN_WINDOW_SECONDS = 300   # 5 minutes
SCAN_THRESHOLD = 10         # unique denied slugs
SCAN_BLOCK_SECONDS = 900    # 15 minute block

# In-memory tracking (per-process fallback)
_scan_denials: dict[str, dict[str, float]] = defaultdict(dict)  # key_prefix -> {slug: timestamp}
_scan_blocks: dict[str, float] = {}  # key_prefix -> block_until timestamp


async def record_access_denial(identifier: str, project_slug: str) -> None:
    """
    Record a denied project access attempt for scan detection.

    Args:
        identifier: Key prefix (first 12 chars of API key)
        project_slug: The project slug/ID that was denied
    """
    now = time.time()

    # Try Redis first
    r = await get_redis()
    if r is not None:
        try:
            redis_key = f"scan_denials:{identifier}"
            await r.hset(redis_key, project_slug, str(now))
            await r.expire(redis_key, SCAN_WINDOW_SECONDS)

            # Count unique denied slugs in window
            all_denials = await r.hgetall(redis_key)
            unique_count = sum(
                1 for ts in all_denials.values()
                if now - float(ts) < SCAN_WINDOW_SECONDS
            )

            if unique_count >= SCAN_THRESHOLD:
                block_key = f"scan_block:{identifier}"
                await r.setex(block_key, SCAN_BLOCK_SECONDS, "1")
                logger.warning(f"Scan blocked: {identifier} ({unique_count} denied slugs)")
            return
        except Exception as e:
            logger.debug(f"Redis scan tracking failed, using in-memory: {e}")

    # In-memory fallback
    denials = _scan_denials[identifier]
    denials[project_slug] = now

    # Prune expired entries
    denials_copy = {s: t for s, t in denials.items() if now - t < SCAN_WINDOW_SECONDS}
    _scan_denials[identifier] = denials_copy

    if len(denials_copy) >= SCAN_THRESHOLD:
        _scan_blocks[identifier] = now + SCAN_BLOCK_SECONDS
        logger.warning(f"Scan blocked (in-memory): {identifier} ({len(denials_copy)} denied slugs)")


async def is_scan_blocked(identifier: str) -> bool:
    """
    Check if a key prefix is blocked due to scan detection.

    Args:
        identifier: Key prefix (first 12 chars of API key)

    Returns:
        True if blocked, False otherwise
    """
    now = time.time()

    # Try Redis first
    r = await get_redis()
    if r is not None:
        try:
            block_key = f"scan_block:{identifier}"
            blocked = await r.get(block_key)
            return blocked is not None
        except Exception:
            pass

    # In-memory fallback
    block_until = _scan_blocks.get(identifier)
    if block_until and now < block_until:
        return True
    elif block_until:
        # Expired - clean up
        _scan_blocks.pop(identifier, None)
    return False
