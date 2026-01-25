"""Swarm Coordinator Service for Phase 9.1.

Manages multi-agent swarms, resource claims, shared state, and task queues.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from prisma import Json
except ImportError:
    # Fallback for when Json isn't available (use identity function)
    Json = lambda x: x

from ..db import get_db
from .agent_limits import check_swarm_agent_limits, check_swarm_limits

logger = logging.getLogger(__name__)

# Default timeouts
DEFAULT_CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_TASK_TIMEOUT_SECONDS = 600  # 10 minutes


# =============================================================================
# SWARM MANAGEMENT
# =============================================================================


async def create_swarm(
    project_id: str,
    name: str,
    description: str | None = None,
    max_agents: int = 10,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new agent swarm.

    Args:
        project_id: The project ID
        name: Swarm name
        description: Optional description
        max_agents: Maximum agents allowed in swarm
        config: Optional swarm configuration

    Returns:
        Dict with swarm info and status
    """
    # Check limits
    allowed, error = await check_swarm_limits(project_id)
    if not allowed:
        return {
            "success": False,
            "error": error,
            "swarm_id": None,
        }

    db = await get_db()

    swarm = await db.agentswarm.create(
        data={
            "project": {"connect": {"id": project_id}},
            "name": name,
            "description": description,
            "maxAgents": max_agents,
            "isActive": True,
        }
    )

    logger.info(f"Created swarm {swarm.id} for project {project_id}")

    return {
        "success": True,
        "swarm_id": swarm.id,
        "name": swarm.name,
        "description": swarm.description,
        "max_agents": swarm.maxAgents,
        "message": f"Swarm '{name}' created successfully",
    }


async def join_swarm(
    swarm_id: str,
    agent_id: str,
    role: str = "worker",
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    """Join an existing swarm as an agent.

    Args:
        swarm_id: The swarm to join
        agent_id: Unique identifier for this agent
        role: Agent role (coordinator, worker, observer)
        capabilities: List of agent capabilities

    Returns:
        Dict with join status and agent info
    """
    db = await get_db()

    # Check if swarm exists and is active
    swarm = await db.agentswarm.find_unique(where={"id": swarm_id})
    if not swarm:
        return {
            "success": False,
            "error": "Swarm not found",
            "agent_id": None,
        }

    if not swarm.isActive:
        return {
            "success": False,
            "error": "Swarm is not active",
            "agent_id": None,
        }

    # Check agent limits
    allowed, error = await check_swarm_agent_limits(swarm_id)
    if not allowed:
        return {
            "success": False,
            "error": error,
            "agent_id": None,
        }

    # Check if agent already in swarm
    existing = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    if existing:
        # Update last heartbeat
        await db.swarmagent.update(
            where={"id": existing.id},
            data={"lastHeartbeat": datetime.now(timezone.utc)},
        )
        return {
            "success": True,
            "agent_id": existing.id,
            "swarm_id": swarm_id,
            "role": role,  # Return requested role (not stored in DB)
            "message": "Already in swarm, updated heartbeat",
        }

    # Join swarm
    agent = await db.swarmagent.create(
        data={
            "swarm": {"connect": {"id": swarm_id}},
            "agentId": agent_id,
            "name": agent_id,  # Use agent_id as name
            "isActive": True,
            "lastHeartbeat": datetime.now(timezone.utc),
        }
    )

    logger.info(f"Agent {agent_id} joined swarm {swarm_id}")

    return {
        "success": True,
        "agent_id": agent.id,
        "swarm_id": swarm_id,
        "role": role,
        "capabilities": capabilities or [],
        "message": f"Joined swarm as {role}",
    }


async def leave_swarm(swarm_id: str, agent_id: str) -> dict[str, Any]:
    """Leave a swarm.

    Args:
        swarm_id: The swarm to leave
        agent_id: The agent's unique identifier

    Returns:
        Dict with leave status
    """
    db = await get_db()

    # Find agent in swarm
    agent = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    if not agent:
        return {
            "success": False,
            "error": "Agent not found in swarm",
        }

    # Mark as inactive (soft delete)
    await db.swarmagent.update(
        where={"id": agent.id},
        data={"isActive": False},
    )

    # Release any claims held by this agent
    await db.resourceclaim.update_many(
        where={
            "agentId": agent.id,
            "status": "ACTIVE",
        },
        data={
            "status": "RELEASED",
            "releasedAt": datetime.now(timezone.utc),
        },
    )

    logger.info(f"Agent {agent_id} left swarm {swarm_id}")

    return {
        "success": True,
        "message": "Left swarm successfully",
    }


async def get_swarm_info(swarm_id: str) -> dict[str, Any]:
    """Get information about a swarm.

    Args:
        swarm_id: The swarm ID

    Returns:
        Dict with swarm info and agent list
    """
    db = await get_db()

    swarm = await db.agentswarm.find_unique(
        where={"id": swarm_id},
        include={"agents": {"where": {"isActive": True}}},
    )

    if not swarm:
        return {
            "success": False,
            "error": "Swarm not found",
        }

    agents = [
        {
            "agent_id": a.agentId,
            "role": a.role.lower(),
            "capabilities": a.capabilities,
            "last_seen": a.lastSeenAt.isoformat() if a.lastSeenAt else None,
        }
        for a in swarm.agents
    ]

    return {
        "success": True,
        "swarm_id": swarm.id,
        "name": swarm.name,
        "description": swarm.description,
        "max_agents": swarm.maxAgents,
        "is_active": swarm.isActive,
        "agent_count": len(agents),
        "agents": agents,
    }


# =============================================================================
# RESOURCE CLAIMS
# =============================================================================


async def acquire_claim(
    swarm_id: str,
    agent_id: str,
    resource_type: str,
    resource_id: str,
    timeout_seconds: int = DEFAULT_CLAIM_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Acquire exclusive access to a resource.

    Args:
        swarm_id: The swarm ID
        agent_id: The agent's unique identifier
        resource_type: Type of resource (file, function, module, etc.)
        resource_id: Identifier of the resource
        timeout_seconds: Claim timeout in seconds

    Returns:
        Dict with claim status
    """
    db = await get_db()

    # Find agent in swarm
    agent = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    if not agent:
        return {
            "success": False,
            "error": "Agent not in swarm",
            "claim_id": None,
        }

    # Check for existing active claim (with lazy expiration)
    existing = await db.resourceclaim.find_first(
        where={
            "swarmId": swarm_id,
            "resourceType": resource_type,
            "resourceId": resource_id,
            "status": "ACTIVE",
        }
    )

    if existing:
        # Lazy expiration check
        if existing.expiresAt and existing.expiresAt < datetime.now(timezone.utc):
            # Claim expired, mark it
            await db.resourceclaim.update(
                where={"id": existing.id},
                data={"status": "EXPIRED"},
            )
            logger.info(f"Expired stale claim {existing.id}")
        else:
            # Claim is still active
            if existing.agentId == agent.id:
                # Same agent, extend the claim
                new_expires = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
                await db.resourceclaim.update(
                    where={"id": existing.id},
                    data={"expiresAt": new_expires},
                )
                return {
                    "success": True,
                    "claim_id": existing.id,
                    "extended": True,
                    "expires_at": new_expires.isoformat(),
                    "message": "Claim extended",
                }
            else:
                # Another agent has the claim
                return {
                    "success": False,
                    "error": "Resource already claimed by another agent",
                    "claim_id": None,
                    "claimed_by": existing.agentId,
                    "expires_at": existing.expiresAt.isoformat() if existing.expiresAt else None,
                }

    # Create new claim
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)

    claim = await db.resourceclaim.create(
        data={
            "swarm": {"connect": {"id": swarm_id}},
            "agent": {"connect": {"id": agent.id}},
            "resourceType": resource_type,
            "resourceId": resource_id,
            "status": "ACTIVE",
            "expiresAt": expires_at,
        }
    )

    logger.info(f"Agent {agent_id} claimed {resource_type}:{resource_id}")

    return {
        "success": True,
        "claim_id": claim.id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "expires_at": expires_at.isoformat(),
        "message": "Resource claimed successfully",
    }


async def release_claim(
    swarm_id: str,
    agent_id: str,
    claim_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
) -> dict[str, Any]:
    """Release a resource claim.

    Args:
        swarm_id: The swarm ID
        agent_id: The agent's unique identifier
        claim_id: Specific claim ID to release
        resource_type: Type of resource (alternative to claim_id)
        resource_id: Resource identifier (alternative to claim_id)

    Returns:
        Dict with release status
    """
    db = await get_db()

    # Find agent in swarm
    agent = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    if not agent:
        return {
            "success": False,
            "error": "Agent not in swarm",
        }

    # Build query
    where: dict[str, Any] = {
        "swarmId": swarm_id,
        "agentId": agent.id,
        "status": "ACTIVE",
    }

    if claim_id:
        where["id"] = claim_id
    elif resource_type and resource_id:
        where["resourceType"] = resource_type
        where["resourceId"] = resource_id
    else:
        return {
            "success": False,
            "error": "Must provide claim_id or resource_type+resource_id",
        }

    # Find and release claim
    claim = await db.resourceclaim.find_first(where=where)

    if not claim:
        return {
            "success": False,
            "error": "Claim not found or not owned by agent",
        }

    await db.resourceclaim.update(
        where={"id": claim.id},
        data={
            "status": "RELEASED",
            "releasedAt": datetime.now(timezone.utc),
        },
    )

    logger.info(f"Released claim {claim.id}")

    return {
        "success": True,
        "claim_id": claim.id,
        "message": "Claim released successfully",
    }


async def check_claim(
    swarm_id: str,
    resource_type: str,
    resource_id: str,
) -> dict[str, Any]:
    """Check if a resource is claimed.

    Args:
        swarm_id: The swarm ID
        resource_type: Type of resource
        resource_id: Resource identifier

    Returns:
        Dict with claim status
    """
    db = await get_db()

    claim = await db.resourceclaim.find_first(
        where={
            "swarmId": swarm_id,
            "resourceType": resource_type,
            "resourceId": resource_id,
            "status": "ACTIVE",
        },
        include={"agent": True},
    )

    if not claim:
        return {
            "claimed": False,
            "resource_type": resource_type,
            "resource_id": resource_id,
        }

    # Lazy expiration
    if claim.expiresAt and claim.expiresAt < datetime.now(timezone.utc):
        await db.resourceclaim.update(
            where={"id": claim.id},
            data={"status": "EXPIRED"},
        )
        return {
            "claimed": False,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "note": "Previous claim expired",
        }

    return {
        "claimed": True,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "claim_id": claim.id,
        "claimed_by": claim.agent.agentId if claim.agent else None,
        "expires_at": claim.expiresAt.isoformat() if claim.expiresAt else None,
    }


# =============================================================================
# SHARED STATE
# =============================================================================


async def get_state(
    swarm_id: str,
    key: str,
) -> dict[str, Any]:
    """Get shared state value.

    Args:
        swarm_id: The swarm ID
        key: State key

    Returns:
        Dict with state value and metadata
    """
    db = await get_db()

    state = await db.sharedstate.find_first(
        where={
            "swarmId": swarm_id,
            "key": key,
        }
    )

    if not state:
        return {
            "found": False,
            "key": key,
            "value": None,
        }

    # Parse JSON value
    try:
        value = json.loads(state.value) if state.value else None
    except json.JSONDecodeError:
        value = state.value

    return {
        "found": True,
        "key": key,
        "value": value,
        "version": state.version,
        "updated_at": state.updatedAt.isoformat() if state.updatedAt else None,
        "updated_by": state.updatedBy,
    }


async def set_state(
    swarm_id: str,
    agent_id: str,
    key: str,
    value: Any,
    expected_version: int | None = None,
) -> dict[str, Any]:
    """Set shared state value with optimistic locking.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent setting the value
        key: State key
        value: Value to set (will be JSON serialized)
        expected_version: If provided, only update if version matches (optimistic lock)

    Returns:
        Dict with new version and status
    """
    db = await get_db()

    # Ensure value is JSON-serializable - Prisma expects dict/list for Json fields
    if isinstance(value, str):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = {"raw": value}
    elif isinstance(value, (dict, list)):
        parsed_value = value
    else:
        parsed_value = {"value": value}

    # Check existing state
    existing = await db.sharedstate.find_first(
        where={
            "swarmId": swarm_id,
            "key": key,
        }
    )

    if existing:
        # Version check (optimistic locking)
        if expected_version is not None and existing.version != expected_version:
            return {
                "success": False,
                "error": "Version mismatch (concurrent update)",
                "current_version": existing.version,
                "expected_version": expected_version,
            }

        # Update existing
        new_version = existing.version + 1
        await db.sharedstate.update(
            where={"id": existing.id},
            data={
                "value": Json(parsed_value),
                "version": new_version,
                "updatedBy": agent_id,
            },
        )

        return {
            "success": True,
            "key": key,
            "version": new_version,
            "message": "State updated",
        }
    else:
        # Create new state
        await db.sharedstate.create(
            data={
                "swarmId": swarm_id,
                "key": key,
                "value": Json(parsed_value),
                "version": 1,
                "updatedBy": agent_id,
            }
        )

        return {
            "success": True,
            "key": key,
            "version": 1,
            "message": "State created",
        }


# =============================================================================
# TASK QUEUE
# =============================================================================


async def create_task(
    swarm_id: str,
    agent_id: str,
    title: str,
    description: str | None = None,
    priority: int = 0,
    depends_on: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a task in the swarm's queue.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent creating the task
        title: Task title
        description: Task description
        priority: Priority (higher = more urgent)
        depends_on: List of task IDs this task depends on
        metadata: Additional task metadata

    Returns:
        Dict with task info
    """
    db = await get_db()

    task = await db.swarmtask.create(
        data={
            "swarm": {"connect": {"id": swarm_id}},
            "title": title,
            "description": description,
            "status": "PENDING",
            "priority": priority,
            "dependsOn": depends_on or [],
        }
    )

    logger.info(f"Created task {task.id} in swarm {swarm_id}")

    return {
        "success": True,
        "task_id": task.id,
        "title": title,
        "priority": priority,
        "depends_on": depends_on or [],
        "message": "Task created",
    }


async def claim_task(
    swarm_id: str,
    agent_id: str,
    task_id: str | None = None,
    timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Claim a task from the queue.

    If task_id is not provided, claims the highest priority available task
    (one whose dependencies are all completed).

    Args:
        swarm_id: The swarm ID
        agent_id: Agent claiming the task
        task_id: Specific task to claim (optional)
        timeout_seconds: Timeout for task completion

    Returns:
        Dict with task info or error
    """
    db = await get_db()

    # Find agent
    agent = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    if not agent:
        return {
            "success": False,
            "error": "Agent not in swarm",
            "task_id": None,
        }

    if task_id:
        # Claim specific task
        task = await db.swarmtask.find_first(
            where={
                "id": task_id,
                "swarmId": swarm_id,
                "status": "PENDING",
            }
        )

        if not task:
            return {
                "success": False,
                "error": "Task not found or not available",
                "task_id": None,
            }
    else:
        # Find available task (dependencies completed)
        task = await _get_available_task(swarm_id)

        if not task:
            return {
                "success": False,
                "error": "No available tasks",
                "task_id": None,
            }

    # Check dependencies
    if task.dependsOn:
        deps_complete = await _check_dependencies_complete(swarm_id, task.dependsOn)
        if not deps_complete:
            return {
                "success": False,
                "error": "Task dependencies not yet completed",
                "task_id": task.id,
                "depends_on": task.dependsOn,
            }

    # Claim the task
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)

    await db.swarmtask.update(
        where={"id": task.id},
        data={
            "status": "IN_PROGRESS",
            "agent": {"connect": {"id": agent.id}},  # Sets assignedTo via relation
            "startedAt": datetime.now(timezone.utc),
            "claimedAt": datetime.now(timezone.utc),
        },
    )

    logger.info(f"Agent {agent_id} claimed task {task.id}")

    return {
        "success": True,
        "task_id": task.id,
        "title": task.title,
        "description": task.description,
        "priority": task.priority,
        "deadline": deadline.isoformat(),
        "message": "Task claimed",
    }


async def complete_task(
    swarm_id: str,
    agent_id: str,
    task_id: str,
    result: Any | None = None,
    success: bool = True,
) -> dict[str, Any]:
    """Complete a claimed task.

    Args:
        swarm_id: The swarm ID
        agent_id: Agent completing the task
        task_id: Task to complete
        result: Task result data
        success: Whether task completed successfully

    Returns:
        Dict with completion status
    """
    db = await get_db()

    # Find agent
    agent = await db.swarmagent.find_first(
        where={
            "swarmId": swarm_id,
            "agentId": agent_id,
            "isActive": True,
        }
    )

    if not agent:
        return {
            "success": False,
            "error": "Agent not in swarm",
        }

    # Find task
    task = await db.swarmtask.find_first(
        where={
            "id": task_id,
            "swarmId": swarm_id,
            "assignedTo": agent.id,
            "status": "IN_PROGRESS",
        }
    )

    if not task:
        return {
            "success": False,
            "error": "Task not found or not assigned to agent",
        }

    # Update task
    status = "COMPLETED" if success else "FAILED"

    # Handle result - parse if string, use directly if dict/list
    parsed_result = result
    if isinstance(result, str):
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            parsed_result = {"raw": result}

    # Build update data - only include result if not None
    update_data: dict[str, Any] = {
        "status": status,
        "completedAt": datetime.now(timezone.utc),
    }
    if parsed_result is not None:
        update_data["result"] = Json(parsed_result)

    await db.swarmtask.update(
        where={"id": task.id},
        data=update_data,
    )

    logger.info(f"Task {task_id} completed with status {status}")

    return {
        "success": True,
        "task_id": task_id,
        "status": status.lower(),
        "message": f"Task marked as {status.lower()}",
    }


async def list_tasks(
    swarm_id: str,
    status: str | None = None,
    assigned_to: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """List tasks in a swarm.

    Args:
        swarm_id: The swarm ID
        status: Filter by status
        assigned_to: Filter by assigned agent
        limit: Maximum tasks to return

    Returns:
        Dict with tasks list
    """
    db = await get_db()

    where: dict[str, Any] = {"swarmId": swarm_id}

    if status:
        where["status"] = status.upper()

    if assigned_to:
        agent = await db.swarmagent.find_first(
            where={"swarmId": swarm_id, "agentId": assigned_to}
        )
        if agent:
            where["assignedTo"] = agent.id

    tasks = await db.swarmtask.find_many(
        where=where,
        order=[{"priority": "desc"}, {"createdAt": "asc"}],
        take=limit,
    )

    return {
        "tasks": [
            {
                "task_id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status.lower(),
                "priority": t.priority,
                "depends_on": t.dependsOn,
                "created_at": t.createdAt.isoformat() if t.createdAt else None,
                "deadline": t.deadline.isoformat() if t.deadline else None,
            }
            for t in tasks
        ],
        "total": len(tasks),
    }


async def _get_available_task(swarm_id: str):
    """Get highest priority task with all dependencies completed."""
    db = await get_db()

    # Get all pending tasks ordered by priority
    pending_tasks = await db.swarmtask.find_many(
        where={
            "swarmId": swarm_id,
            "status": "PENDING",
        },
        order=[{"priority": "desc"}, {"createdAt": "asc"}],
    )

    for task in pending_tasks:
        if not task.dependsOn:
            return task

        # Check if all dependencies are completed
        deps_complete = await _check_dependencies_complete(swarm_id, task.dependsOn)
        if deps_complete:
            return task

    return None


async def _check_dependencies_complete(swarm_id: str, dep_ids: list[str]) -> bool:
    """Check if all dependency tasks are completed."""
    db = await get_db()

    completed_count = await db.swarmtask.count(
        where={
            "swarmId": swarm_id,
            "id": {"in": dep_ids},
            "status": "COMPLETED",
        }
    )

    return completed_count == len(dep_ids)
