"""Swarm tool handlers for multi-agent coordination.

Handles:
- rlm_swarm_create: Create a new agent swarm
- rlm_swarm_join: Join an existing swarm
- rlm_claim: Claim exclusive access to a resource
- rlm_release: Release a claimed resource
- rlm_state_get: Read shared swarm state
- rlm_state_set: Write shared swarm state
- rlm_broadcast: Broadcast event to all agents
- rlm_task_create: Create a task in the queue
- rlm_task_claim: Claim a task from the queue
- rlm_task_complete: Mark a task as complete
"""

from typing import Any

from ...models import ToolResult
from ...services.swarm import (
    acquire_claim,
    broadcast_event,
    claim_task,
    complete_task,
    create_swarm,
    create_task,
    get_state,
    join_swarm,
    release_claim,
    set_state,
)
from .base import HandlerContext, count_tokens


async def handle_swarm_create(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Create a new agent swarm.

    Args:
        params: Dict containing:
            - name: Swarm name
            - description: Optional description
            - max_agents: Maximum agents allowed
            - config: Optional swarm configuration

    Returns:
        ToolResult with swarm ID and info
    """
    name = params.get("name", "")
    description = params.get("description")
    max_agents = params.get("max_agents", 10)
    config = params.get("config")

    if not name:
        return ToolResult(
            data={"error": "name is required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await create_swarm(
        project_id=ctx.project_id,
        name=name,
        description=description,
        max_agents=max_agents,
        config=config,
        user_id=ctx.user_id,
    )

    return ToolResult(
        data=result,
        input_tokens=count_tokens(name + (description or "")),
        output_tokens=count_tokens(str(result)),
    )


async def handle_swarm_join(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Join an existing swarm as an agent.

    Args:
        params: Dict containing:
            - swarm_id: Swarm to join
            - agent_id: Unique agent identifier
            - role: Agent role (coordinator, worker, observer)
            - capabilities: List of capabilities

    Returns:
        ToolResult with join status
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    role = params.get("role", "worker")
    capabilities = params.get("capabilities")

    if not swarm_id or not agent_id:
        return ToolResult(
            data={"error": "swarm_id and agent_id are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await join_swarm(
        swarm_id=swarm_id,
        agent_id=agent_id,
        role=role,
        capabilities=capabilities,
    )

    return ToolResult(
        data=result,
        input_tokens=0,
        output_tokens=count_tokens(str(result)),
    )


async def handle_claim(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Claim exclusive access to a resource.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Agent identifier
            - resource_type: Type of resource
            - resource_id: Resource identifier
            - timeout_seconds: Claim timeout

    Returns:
        ToolResult with claim status
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    resource_type = params.get("resource_type", "")
    resource_id = params.get("resource_id", "")
    timeout_seconds = params.get("timeout_seconds", 300)

    if not all([swarm_id, agent_id, resource_type, resource_id]):
        return ToolResult(
            data={"error": "swarm_id, agent_id, resource_type, and resource_id are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await acquire_claim(
        swarm_id=swarm_id,
        agent_id=agent_id,
        resource_type=resource_type,
        resource_id=resource_id,
        timeout_seconds=timeout_seconds,
    )

    return ToolResult(
        data=result,
        input_tokens=0,
        output_tokens=count_tokens(str(result)),
    )


async def handle_release(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Release a claimed resource.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Agent identifier
            - claim_id: Claim ID (optional)
            - resource_type: Resource type (alternative)
            - resource_id: Resource ID (alternative)

    Returns:
        ToolResult with release status
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    claim_id = params.get("claim_id")
    resource_type = params.get("resource_type")
    resource_id = params.get("resource_id")

    if not swarm_id or not agent_id:
        return ToolResult(
            data={"error": "swarm_id and agent_id are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await release_claim(
        swarm_id=swarm_id,
        agent_id=agent_id,
        claim_id=claim_id,
        resource_type=resource_type,
        resource_id=resource_id,
    )

    return ToolResult(
        data=result,
        input_tokens=0,
        output_tokens=count_tokens(str(result)),
    )


async def handle_state_get(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Read shared swarm state.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - key: State key to read

    Returns:
        ToolResult with state value and version
    """
    swarm_id = params.get("swarm_id", "")
    key = params.get("key", "")

    if not swarm_id or not key:
        return ToolResult(
            data={"error": "swarm_id and key are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await get_state(swarm_id=swarm_id, key=key)

    return ToolResult(
        data=result,
        input_tokens=0,
        output_tokens=count_tokens(str(result)),
    )


async def handle_state_set(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Write shared swarm state with optimistic locking.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Agent identifier
            - key: State key
            - value: Value to set
            - expected_version: Optional version for optimistic locking

    Returns:
        ToolResult with new version
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    key = params.get("key", "")
    value = params.get("value")
    expected_version = params.get("expected_version")

    if not swarm_id or not agent_id or not key:
        return ToolResult(
            data={"error": "swarm_id, agent_id, and key are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await set_state(
        swarm_id=swarm_id,
        agent_id=agent_id,
        key=key,
        value=value,
        expected_version=expected_version,
    )

    return ToolResult(
        data=result,
        input_tokens=count_tokens(str(value) if value else ""),
        output_tokens=count_tokens(str(result)),
    )


async def handle_broadcast(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Broadcast an event to all agents in the swarm.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Sending agent identifier
            - event_type: Type of event
            - payload: Optional event data

    Returns:
        ToolResult with delivery count
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    event_type = params.get("event_type", "")
    payload = params.get("payload")

    if not swarm_id or not agent_id or not event_type:
        return ToolResult(
            data={"error": "swarm_id, agent_id, and event_type are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await broadcast_event(
        swarm_id=swarm_id,
        agent_id=agent_id,
        event_type=event_type,
        payload=payload,
    )

    return ToolResult(
        data=result,
        input_tokens=count_tokens(str(payload) if payload else ""),
        output_tokens=count_tokens(str(result)),
    )


async def handle_task_create(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Create a task in the swarm's distributed task queue.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Creating agent identifier
            - title: Task title
            - description: Optional description
            - priority: Task priority (higher = more urgent)
            - depends_on: Task IDs this depends on
            - metadata: Additional task data

    Returns:
        ToolResult with task ID
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    title = params.get("title", "")
    description = params.get("description")
    priority = params.get("priority", 0)
    depends_on = params.get("depends_on")
    metadata = params.get("metadata")

    if not swarm_id or not agent_id or not title:
        return ToolResult(
            data={"error": "swarm_id, agent_id, and title are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await create_task(
        swarm_id=swarm_id,
        agent_id=agent_id,
        title=title,
        description=description,
        priority=priority,
        depends_on=depends_on,
        metadata=metadata,
    )

    return ToolResult(
        data=result,
        input_tokens=count_tokens(title + (description or "")),
        output_tokens=count_tokens(str(result)),
    )


async def handle_task_claim(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Claim a task from the queue.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Claiming agent identifier
            - task_id: Optional specific task to claim
            - timeout_seconds: Task timeout

    Returns:
        ToolResult with claimed task details
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    task_id = params.get("task_id")
    timeout_seconds = params.get("timeout_seconds", 600)

    if not swarm_id or not agent_id:
        return ToolResult(
            data={"error": "swarm_id and agent_id are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await claim_task(
        swarm_id=swarm_id,
        agent_id=agent_id,
        task_id=task_id,
        timeout_seconds=timeout_seconds,
    )

    return ToolResult(
        data=result,
        input_tokens=0,
        output_tokens=count_tokens(str(result)),
    )


async def handle_task_complete(
    params: dict[str, Any],
    ctx: HandlerContext,
) -> ToolResult:
    """Mark a claimed task as completed or failed.

    Args:
        params: Dict containing:
            - swarm_id: Swarm ID
            - agent_id: Completing agent identifier
            - task_id: Task to complete
            - success: Whether task succeeded
            - result: Optional result data

    Returns:
        ToolResult with completion confirmation
    """
    swarm_id = params.get("swarm_id", "")
    agent_id = params.get("agent_id", "")
    task_id = params.get("task_id", "")
    success = params.get("success", True)
    result_data = params.get("result")

    if not swarm_id or not agent_id or not task_id:
        return ToolResult(
            data={"error": "swarm_id, agent_id, and task_id are required"},
            input_tokens=0,
            output_tokens=0,
        )

    result = await complete_task(
        swarm_id=swarm_id,
        agent_id=agent_id,
        task_id=task_id,
        success=success,
        result=result_data,
    )

    return ToolResult(
        data=result,
        input_tokens=count_tokens(str(result_data) if result_data else ""),
        output_tokens=count_tokens(str(result)),
    )
