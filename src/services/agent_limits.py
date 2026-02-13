"""Agent Limits Service for Phase 8-9.

Enforces plan limits for memories, swarms, and agents.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from ..db import get_db

logger = logging.getLogger(__name__)

# Context plan hierarchy for validation
CONTEXT_PLAN_HIERARCHY = {
    "FREE": 0,
    "PRO": 1,
    "TEAM": 2,
    "ENTERPRISE": 3,
}

# Required Context plan for Agents plans
REQUIRED_CONTEXT_PLAN = {
    "STARTER": None,  # No requirement
    "PRO": None,  # No requirement
    "TEAM": "TEAM",  # Requires Context TEAM or higher
    "ENTERPRISE": "ENTERPRISE",  # Requires Context ENTERPRISE only
}

# Plan limits by tier
PLAN_LIMITS = {
    "STARTER": {
        "memories": 1000,
        "retention_days": 7,
        "swarms": 1,
        "agents_per_swarm": 2,
        "cache_ttl_seconds": 300,  # 5 min
        "real_time_events": False,
    },
    "PRO": {
        "memories": 5000,
        "retention_days": 30,
        "swarms": 5,
        "agents_per_swarm": 5,
        "cache_ttl_seconds": 1800,  # 30 min
        "real_time_events": False,
    },
    "TEAM": {
        "memories": 25000,
        "retention_days": 90,
        "swarms": 20,
        "agents_per_swarm": 15,
        "cache_ttl_seconds": 7200,  # 2 hr
        "real_time_events": True,
    },
    "ENTERPRISE": {
        "memories": -1,  # Unlimited
        "retention_days": -1,  # Unlimited
        "swarms": -1,  # Unlimited
        "agents_per_swarm": 50,
        "cache_ttl_seconds": 86400,  # 24 hr
        "real_time_events": True,
    },
}

# Default limits if no subscription
DEFAULT_LIMITS = PLAN_LIMITS["STARTER"]


async def get_agents_subscription(
    project_id: str, user_id: str | None = None
) -> dict[str, Any] | None:
    """Get the agents subscription for a project.

    First tries to find by team (if project belongs to team),
    then falls back to user's personal subscription.

    Args:
        project_id: The project ID
        user_id: The authenticated user's ID (used for personal subscription lookup)

    Returns:
        Subscription dict or None if no subscription
    """
    db = await get_db()

    # Check if AgentsSubscription model is available in the Prisma client
    if not hasattr(db, "agentssubscription"):
        logger.warning("AgentsSubscription model not available in Prisma client - needs regeneration")
        return None

    # Get project with team info
    project = await db.project.find_unique(
        where={"id": project_id},
        include={"team": True},
    )

    if not project:
        logger.warning(f"Project not found: {project_id}")
        return None

    # Try team subscription first
    if project.teamId:
        team_sub = await db.agentssubscription.find_first(
            where={
                "teamId": project.teamId,
                "status": "active",
            },
        )
        if team_sub:
            return _subscription_to_dict(team_sub)

        # Fall back to team owner's personal subscription
        team_member = await db.teammember.find_first(
            where={
                "teamId": project.teamId,
                "role": "OWNER",
            },
        )
        if team_member:
            owner_sub = await db.agentssubscription.find_first(
                where={
                    "userId": team_member.userId,
                    "teamId": None,  # Personal subscription
                    "status": "active",
                },
            )
            if owner_sub:
                return _subscription_to_dict(owner_sub)

    # Fall back to authenticated user's personal subscription
    # This handles projects without teams (most common case for individual users)
    if user_id:
        user_sub = await db.agentssubscription.find_first(
            where={
                "userId": user_id,
                "teamId": None,  # Personal subscription
                "status": "active",
            },
        )
        if user_sub:
            return _subscription_to_dict(user_sub)

    return None


def _subscription_to_dict(sub: Any) -> dict[str, Any]:
    """Convert subscription model to dict."""
    return {
        "id": sub.id,
        "plan": sub.plan,
        "status": sub.status,
        "team_id": sub.teamId,
        "memory_limit": sub.memoryLimit,
        "swarm_limit": sub.swarmLimit,
        "agents_per_swarm_limit": sub.agentsPerSwarmLimit,
        "cache_ttl_seconds": sub.cacheTtlSeconds,
        "grace_period_end": getattr(sub, "gracePeriodEnd", None),
        "grace_period_reason": getattr(sub, "gracePeriodReason", None),
    }


def get_plan_limits(plan: str) -> dict[str, Any]:
    """Get limits for a plan.

    Args:
        plan: Plan name (STARTER, PRO, TEAM, ENTERPRISE)

    Returns:
        Dict with plan limits
    """
    return PLAN_LIMITS.get(plan.upper(), DEFAULT_LIMITS)


async def check_memory_limits(
    project_id: str, user_id: str | None = None
) -> tuple[bool, str | None]:
    """Check if project can create new memories.

    Args:
        project_id: The project ID
        user_id: The authenticated user's ID (used for personal subscription lookup)

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()
    subscription = await get_agents_subscription(project_id, user_id)

    # Get limit from subscription or use default
    if subscription:
        memory_limit = subscription["memory_limit"]
    else:
        memory_limit = DEFAULT_LIMITS["memories"]

    # -1 means unlimited
    if memory_limit == -1:
        return True, None

    # Count current memories
    current_count = await db.agentmemory.count(
        where={"projectId": project_id}
    )

    if current_count >= memory_limit:
        return False, f"Memory limit reached ({current_count}/{memory_limit}). Upgrade your plan for more memories."

    return True, None


async def check_swarm_limits(
    project_id: str, user_id: str | None = None
) -> tuple[bool, str | None]:
    """Check if project can create new swarms.

    Args:
        project_id: The project ID
        user_id: The authenticated user's ID (used for personal subscription lookup)

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()
    subscription = await get_agents_subscription(project_id, user_id)

    # Get limit from subscription or use default
    if subscription:
        swarm_limit = subscription["swarm_limit"]
    else:
        swarm_limit = DEFAULT_LIMITS["swarms"]

    # -1 means unlimited
    if swarm_limit == -1:
        return True, None

    # Count current active swarms
    current_count = await db.agentswarm.count(
        where={
            "projectId": project_id,
            "isActive": True,
        }
    )

    if current_count >= swarm_limit:
        return False, f"Swarm limit reached ({current_count}/{swarm_limit}). Upgrade your plan for more swarms."

    return True, None


async def check_swarm_agent_limits(swarm_id: str) -> tuple[bool, str | None]:
    """Check if swarm can accept more agents.

    Args:
        swarm_id: The swarm ID

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()

    # Get swarm with project
    swarm = await db.agentswarm.find_unique(
        where={"id": swarm_id},
        include={"project": True},
    )

    if not swarm:
        return False, "Swarm not found"

    subscription = await get_agents_subscription(swarm.projectId)

    # Get limit from subscription or use default
    if subscription:
        agents_limit = subscription["agents_per_swarm_limit"]
    else:
        agents_limit = DEFAULT_LIMITS["agents_per_swarm"]

    # Also check swarm's own maxAgents setting
    swarm_max = swarm.maxAgents
    effective_limit = min(agents_limit, swarm_max)

    # Count current agents
    current_count = await db.swarmagent.count(
        where={
            "swarmId": swarm_id,
            "isActive": True,
        }
    )

    if current_count >= effective_limit:
        return False, f"Agent limit reached ({current_count}/{effective_limit}). Upgrade your plan or increase swarm capacity."

    return True, None


async def get_memory_retention_limit(project_id: str) -> int:
    """Get max retention days for memories.

    Args:
        project_id: The project ID

    Returns:
        Max retention days (-1 = unlimited)
    """
    subscription = await get_agents_subscription(project_id)

    if subscription:
        plan = subscription["plan"]
        limits = get_plan_limits(plan)
        return limits["retention_days"]

    return DEFAULT_LIMITS["retention_days"]


async def get_cache_ttl(project_id: str) -> int:
    """Get cache TTL for project.

    Args:
        project_id: The project ID

    Returns:
        Cache TTL in seconds
    """
    subscription = await get_agents_subscription(project_id)

    if subscription:
        return subscription["cache_ttl_seconds"]

    return DEFAULT_LIMITS["cache_ttl_seconds"]


async def has_real_time_events(project_id: str) -> bool:
    """Check if project has real-time events enabled.

    Args:
        project_id: The project ID

    Returns:
        True if real-time events are enabled
    """
    subscription = await get_agents_subscription(project_id)

    if subscription:
        plan = subscription["plan"]
        limits = get_plan_limits(plan)
        return limits["real_time_events"]

    return DEFAULT_LIMITS["real_time_events"]


async def get_usage_stats(project_id: str) -> dict[str, Any]:
    """Get current usage statistics for a project.

    Args:
        project_id: The project ID

    Returns:
        Dict with current usage and limits
    """
    db = await get_db()
    subscription = await get_agents_subscription(project_id)

    # Get current counts
    memory_count = await db.agentmemory.count(where={"projectId": project_id})
    swarm_count = await db.agentswarm.count(
        where={"projectId": project_id, "isActive": True}
    )

    # Get limits
    if subscription:
        plan = subscription["plan"]
        memory_limit = subscription["memory_limit"]
        swarm_limit = subscription["swarm_limit"]
    else:
        plan = "STARTER"
        memory_limit = DEFAULT_LIMITS["memories"]
        swarm_limit = DEFAULT_LIMITS["swarms"]

    return {
        "plan": plan,
        "subscription_active": subscription is not None,
        "memories": {
            "current": memory_count,
            "limit": memory_limit,
            "unlimited": memory_limit == -1,
        },
        "swarms": {
            "current": swarm_count,
            "limit": swarm_limit,
            "unlimited": swarm_limit == -1,
        },
    }


async def validate_agents_context_plan(
    project_id: str,
    agents_subscription: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Validate that team has required Context plan for Agents TEAM/ENTERPRISE.

    Args:
        project_id: The project ID
        agents_subscription: Optional pre-fetched agents subscription

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()

    # Get agents subscription if not provided
    if agents_subscription is None:
        agents_subscription = await get_agents_subscription(project_id)

    if not agents_subscription:
        return True, None  # No agents subscription = no validation needed

    agents_plan = agents_subscription["plan"]
    required_context = REQUIRED_CONTEXT_PLAN.get(agents_plan)

    if not required_context:
        return True, None  # STARTER/PRO = no Context requirement

    team_id = agents_subscription.get("team_id")
    if not team_id:
        # Agents TEAM/ENTERPRISE requires a team context
        return False, f"Agents {agents_plan} requires a team subscription context."

    # Get team's Context subscription
    context_sub = await db.subscription.find_first(
        where={"teamId": team_id}
    )

    if not context_sub:
        return False, f"Agents {agents_plan} requires Context {required_context} plan. No Context subscription found."

    context_plan = context_sub.plan

    # Check if Context plan is sufficient
    required_level = CONTEXT_PLAN_HIERARCHY.get(required_context, 0)
    current_level = CONTEXT_PLAN_HIERARCHY.get(context_plan, 0)

    if current_level < required_level:
        return False, (
            f"Agents {agents_plan} requires Context {required_context} plan or higher. "
            f"Current Context plan: {context_plan}. Please upgrade your Context plan."
        )

    return True, None


async def check_grace_period(
    agents_subscription: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Check if agents subscription is within grace period.

    If grace period is set and expired, access is denied.
    If grace period is set but not expired, access is allowed with a warning.

    Args:
        agents_subscription: The agents subscription dict

    Returns:
        Tuple of (allowed, warning_message)
        - (True, None): No grace period or not in grace period
        - (True, "Warning: X days left..."): In grace period but not expired
        - (False, "Grace period expired..."): Grace period has expired
    """
    if not agents_subscription:
        return True, None

    grace_period_end = agents_subscription.get("grace_period_end")
    if not grace_period_end:
        return True, None

    # Ensure grace_period_end is timezone-aware
    if isinstance(grace_period_end, datetime):
        if grace_period_end.tzinfo is None:
            grace_period_end = grace_period_end.replace(tzinfo=UTC)

    now = datetime.now(UTC)

    if now > grace_period_end:
        agents_plan = agents_subscription.get("plan", "UNKNOWN")
        reason = agents_subscription.get("grace_period_reason", "Context plan downgraded")
        return False, (
            f"Grace period expired for Agents {agents_plan}. "
            f"Reason: {reason}. "
            f"Please upgrade your Context plan to continue using Agents features."
        )

    # Grace period not yet expired - allow with warning
    days_left = (grace_period_end - now).days
    agents_plan = agents_subscription.get("plan", "UNKNOWN")
    return True, (
        f"Warning: {days_left} day(s) left before Agents {agents_plan} access is restricted. "
        f"Please upgrade your Context plan to avoid interruption."
    )


async def validate_agents_access(
    project_id: str,
    user_id: str | None = None,
) -> tuple[bool, str | None, str | None]:
    """Full validation for agents access including Context plan and grace period.

    Args:
        project_id: The project ID
        user_id: The authenticated user's ID (used for personal subscription lookup)

    Returns:
        Tuple of (allowed, error_message, warning_message)
        - error_message is set if access is denied
        - warning_message is set if in grace period (access allowed but expiring)
    """
    agents_subscription = await get_agents_subscription(project_id, user_id)

    if not agents_subscription:
        return True, None, None  # No agents subscription = use defaults

    # Check grace period first
    grace_allowed, grace_message = await check_grace_period(agents_subscription)
    if not grace_allowed:
        return False, grace_message, None

    # Check Context plan requirement
    context_allowed, context_error = await validate_agents_context_plan(
        project_id, agents_subscription
    )
    if not context_allowed:
        return False, context_error, None

    # Access allowed, return any grace period warning
    return True, None, grace_message
