"""Swarm Events Service for Phase 9.2.

Real-time event broadcasting using Redis pub/sub for multi-agent coordination.
Events are also persisted to the database for agents that poll.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

try:
    from prisma import Json
except ImportError:
    # Fallback for when Json isn't available (use identity function)
    Json = lambda x: x

from ..db import get_db
from .cache import get_redis

logger = logging.getLogger(__name__)

# Channel naming
SWARM_CHANNEL_PREFIX = "swarm:"
SWARM_CHANNEL_SUFFIX = ":events"

# Event retention
MAX_EVENTS_PER_SWARM = 100  # Keep last 100 events in DB


def _get_channel_name(swarm_id: str) -> str:
    """Get Redis channel name for a swarm."""
    return f"{SWARM_CHANNEL_PREFIX}{swarm_id}{SWARM_CHANNEL_SUFFIX}"


async def broadcast_event(
    swarm_id: str,
    agent_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Broadcast an event to all agents in a swarm.

    Events are:
    1. Published to Redis channel for real-time delivery
    2. Persisted to database for polling agents

    Args:
        swarm_id: The swarm ID
        agent_id: Agent sending the event
        event_type: Type of event (e.g., "task_completed", "resource_claimed")
        payload: Event data

    Returns:
        Dict with broadcast status
    """
    db = await get_db()
    redis = await get_redis()

    # Parse payload if it's a string (from MCP)
    parsed_payload = payload
    if isinstance(payload, str):
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError:
            parsed_payload = {"raw": payload}
    elif payload is None:
        parsed_payload = {}

    # Look up SwarmAgent by external agent_id
    swarm_agent = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    # Build event data
    event_data = {
        "type": event_type,
        "agent_id": agent_id,
        "swarm_id": swarm_id,
        "payload": parsed_payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Persist to database
    create_data: dict[str, Any] = {
        "swarmId": swarm_id,
        "eventType": event_type,
        "payload": Json(parsed_payload),
    }
    # Only connect agent if found
    if swarm_agent:
        create_data["agentId"] = swarm_agent.id

    event = await db.swarmevent.create(data=create_data)

    # 2. Publish to Redis (if available)
    redis_published = False
    if redis:
        try:
            channel = _get_channel_name(swarm_id)
            event_data["event_id"] = event.id
            await redis.publish(channel, json.dumps(event_data))
            redis_published = True
            logger.debug(f"Published event to {channel}")
        except Exception as e:
            logger.warning(f"Failed to publish to Redis: {e}")

    # 3. Cleanup old events (keep last N)
    await _cleanup_old_events(swarm_id)

    return {
        "success": True,
        "event_id": event.id,
        "event_type": event_type,
        "redis_published": redis_published,
        "message": "Event broadcast successfully",
    }


async def get_recent_events(
    swarm_id: str,
    since: datetime | None = None,
    event_type: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Get recent events from a swarm.

    For agents that poll instead of subscribing to real-time events.

    Args:
        swarm_id: The swarm ID
        since: Only return events after this timestamp
        event_type: Filter by event type
        limit: Maximum events to return

    Returns:
        Dict with events list
    """
    db = await get_db()

    where: dict[str, Any] = {"swarmId": swarm_id}

    if since:
        where["createdAt"] = {"gt": since}

    if event_type:
        where["eventType"] = event_type

    events = await db.swarmevent.find_many(
        where=where,
        order_by={"createdAt": "desc"},
        take=limit,
    )

    return {
        "events": [
            {
                "event_id": e.id,
                "event_type": e.eventType,
                "agent_id": e.agentId,
                "payload": json.loads(e.payload) if e.payload else None,
                "timestamp": e.createdAt.isoformat() if e.createdAt else None,
            }
            for e in reversed(events)  # Return in chronological order
        ],
        "total": len(events),
        "swarm_id": swarm_id,
    }


async def subscribe_to_swarm(swarm_id: str):
    """Create a Redis subscription for swarm events.

    Note: This is for server-side SSE endpoints, not MCP tools.
    MCP clients should poll via get_recent_events.

    Args:
        swarm_id: The swarm ID

    Returns:
        Redis pubsub object or None if Redis unavailable
    """
    redis = await get_redis()
    if not redis:
        return None

    try:
        pubsub = redis.pubsub()
        channel = _get_channel_name(swarm_id)
        await pubsub.subscribe(channel)
        logger.info(f"Subscribed to {channel}")
        return pubsub
    except Exception as e:
        logger.warning(f"Failed to subscribe to swarm events: {e}")
        return None


async def unsubscribe_from_swarm(pubsub, swarm_id: str):
    """Unsubscribe from swarm events.

    Args:
        pubsub: Redis pubsub object
        swarm_id: The swarm ID
    """
    if pubsub:
        try:
            channel = _get_channel_name(swarm_id)
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            logger.info(f"Unsubscribed from {channel}")
        except Exception as e:
            logger.warning(f"Error unsubscribing: {e}")


async def _cleanup_old_events(swarm_id: str):
    """Delete old events keeping only the most recent N.

    Args:
        swarm_id: The swarm ID
    """
    db = await get_db()

    try:
        # Count total events
        total = await db.swarmevent.count(where={"swarmId": swarm_id})

        if total > MAX_EVENTS_PER_SWARM:
            # Find cutoff event
            events_to_keep = await db.swarmevent.find_many(
                where={"swarmId": swarm_id},
                order_by={"createdAt": "desc"},
                take=MAX_EVENTS_PER_SWARM,
                select={"id": True},
            )
            keep_ids = [e.id for e in events_to_keep]

            # Delete older events
            deleted = await db.swarmevent.delete_many(
                where={
                    "swarmId": swarm_id,
                    "id": {"not_in": keep_ids},
                }
            )
            if deleted > 0:
                logger.debug(f"Cleaned up {deleted} old events for swarm {swarm_id}")
    except Exception as e:
        logger.warning(f"Event cleanup failed: {e}")


# =============================================================================
# CONVENIENCE EVENT TYPES
# =============================================================================


async def broadcast_task_event(
    swarm_id: str,
    agent_id: str,
    task_id: str,
    event_type: str,  # "created", "claimed", "completed", "failed"
    task_title: str | None = None,
    result: Any = None,
) -> dict[str, Any]:
    """Broadcast a task-related event.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent involved
        task_id: Task ID
        event_type: Task event type
        task_title: Task title (optional)
        result: Task result (for completed events)

    Returns:
        Broadcast result
    """
    payload = {
        "task_id": task_id,
        "task_title": task_title,
    }
    if result is not None:
        payload["result"] = result

    return await broadcast_event(
        swarm_id=swarm_id,
        agent_id=agent_id,
        event_type=f"task_{event_type}",
        payload=payload,
    )


async def broadcast_claim_event(
    swarm_id: str,
    agent_id: str,
    resource_type: str,
    resource_id: str,
    event_type: str,  # "acquired", "released", "expired"
) -> dict[str, Any]:
    """Broadcast a resource claim event.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent involved
        resource_type: Type of resource
        resource_id: Resource identifier
        event_type: Claim event type

    Returns:
        Broadcast result
    """
    return await broadcast_event(
        swarm_id=swarm_id,
        agent_id=agent_id,
        event_type=f"claim_{event_type}",
        payload={
            "resource_type": resource_type,
            "resource_id": resource_id,
        },
    )


async def broadcast_agent_event(
    swarm_id: str,
    agent_id: str,
    event_type: str,  # "joined", "left", "heartbeat"
    role: str | None = None,
) -> dict[str, Any]:
    """Broadcast an agent lifecycle event.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent involved
        event_type: Agent event type
        role: Agent role (for join events)

    Returns:
        Broadcast result
    """
    payload = {}
    if role:
        payload["role"] = role

    return await broadcast_event(
        swarm_id=swarm_id,
        agent_id=agent_id,
        event_type=f"agent_{event_type}",
        payload=payload if payload else None,
    )


async def broadcast_state_event(
    swarm_id: str,
    agent_id: str,
    key: str,
    version: int,
) -> dict[str, Any]:
    """Broadcast a shared state change event.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent that changed state
        key: State key
        version: New version number

    Returns:
        Broadcast result
    """
    return await broadcast_event(
        swarm_id=swarm_id,
        agent_id=agent_id,
        event_type="state_changed",
        payload={
            "key": key,
            "version": version,
        },
    )
