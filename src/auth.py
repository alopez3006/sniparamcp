"""
API key validation module.

This module handles authentication for the MCP server, supporting both API keys
and OAuth tokens. All functions that accept a project identifier support both:
- Database ID (e.g., "cm5xyz123abc...")
- Project slug (e.g., "snipara", "my-project")

This flexibility allows users to use human-readable slugs in their MCP
configuration URLs instead of opaque database IDs. For example:
    https://api.snipara.com/mcp/snipara  (using slug)
    https://api.snipara.com/mcp/cm5xyz...  (using ID)

Both formats are resolved in a single database query using OR conditions
to avoid extra round-trips.
"""

import hashlib
from datetime import datetime, timezone

from .db import get_db


def hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(key.encode()).hexdigest()


async def validate_api_key(api_key: str, project_id_or_slug: str) -> dict | None:
    """
    Validate an API key and check project access.

    Tries project-specific API keys first, then falls back to team API keys.
    This allows users to use either:
    - A project-specific key (rlm_...) for a single project
    - A team API key that works for all projects in the team

    Args:
        api_key: The API key from the request header
        project_id_or_slug: The project ID or slug being accessed

    Returns:
        API key record if valid, None otherwise
    """
    db = await get_db()

    # Hash the provided key to compare with stored hash
    key_hash = hash_api_key(api_key)

    # First, try to find a project-specific API key
    api_key_record = await db.apikey.find_first(
        where={
            "keyHash": key_hash,
            "project": {
                "OR": [
                    {"id": project_id_or_slug},
                    {"slug": project_id_or_slug},
                ]
            },
        },
        include={
            "project": {
                "include": {
                    "team": {
                        "include": {
                            "subscription": True,
                        }
                    }
                }
            },
            "user": True,
        },
    )

    if api_key_record:
        # Check if key is revoked
        if api_key_record.revokedAt:
            return None

        # Check if key is expired
        if api_key_record.expiresAt and api_key_record.expiresAt < datetime.now(timezone.utc):
            return None

        # Update last used timestamp
        await db.apikey.update(
            where={"id": api_key_record.id},
            data={"lastUsedAt": datetime.now(timezone.utc)},
        )

        return {
            "id": api_key_record.id,
            "name": api_key_record.name,
            "user_id": api_key_record.userId,
            "project_id": api_key_record.projectId,
            "project": api_key_record.project,
        }

    # If no project key found, try team API key
    # First get the project to find its team
    project = await db.project.find_first(
        where={
            "OR": [
                {"id": project_id_or_slug},
                {"slug": project_id_or_slug},
            ]
        },
        include={
            "team": {
                "include": {
                    "subscription": True,
                }
            }
        },
    )

    if not project:
        return None

    # Now try to find a team API key for this project's team
    team_key_record = await db.teamapikey.find_first(
        where={
            "keyHash": key_hash,
            "teamId": project.teamId,
        },
        include={
            "team": {
                "include": {
                    "subscription": True,
                }
            },
            "user": True,
        },
    )

    if not team_key_record:
        return None

    # Check if team key is revoked
    if team_key_record.revokedAt:
        return None

    # Check if team key is expired
    if team_key_record.expiresAt and team_key_record.expiresAt < datetime.now(timezone.utc):
        return None

    # Update last used timestamp for team key
    await db.teamapikey.update(
        where={"id": team_key_record.id},
        data={"lastUsedAt": datetime.now(timezone.utc)},
    )

    # Return team key info with project attached
    return {
        "id": team_key_record.id,
        "name": team_key_record.name,
        "user_id": team_key_record.userId,
        "project_id": project.id,
        "project": project,
        "auth_type": "team_key",
    }


async def validate_team_api_key(api_key: str, team_id: str) -> dict | None:
    """
    Validate a team API key and check team access.

    Args:
        api_key: The API key from the request header
        team_id: The team being accessed

    Returns:
        API key record if valid, None otherwise
    """
    db = await get_db()

    key_hash = hash_api_key(api_key)

    api_key_record = await db.teamapikey.find_first(
        where={
            "keyHash": key_hash,
            "teamId": team_id,
        },
        include={
            "team": {
                "include": {
                    "subscription": True,
                    "projects": True,
                }
            },
            "user": True,
        },
    )

    if not api_key_record:
        return None

    if api_key_record.expiresAt and api_key_record.expiresAt < datetime.now(timezone.utc):
        return None

    if api_key_record.revokedAt:
        return None

    await db.teamapikey.update(
        where={"id": api_key_record.id},
        data={"lastUsedAt": datetime.now(timezone.utc)},
    )

    return {
        "id": api_key_record.id,
        "name": api_key_record.name,
        "user_id": api_key_record.userId,
        "team_id": api_key_record.teamId,
        "team": api_key_record.team,
    }


async def get_project_with_team(project_id_or_slug: str) -> dict | None:
    """
    Get project details including team and subscription info.

    Args:
        project_id_or_slug: The project ID or slug

    Returns:
        Project record with team and subscription, or None
    """
    db = await get_db()

    # Try by ID first, then by slug
    project = await db.project.find_first(
        where={
            "OR": [
                {"id": project_id_or_slug},
                {"slug": project_id_or_slug},
            ]
        },
        include={
            "team": {
                "include": {
                    "subscription": True,
                }
            },
            "documents": True,
        },
    )

    return project


async def get_team_by_slug_or_id(team_slug_or_id: str) -> dict | None:
    """
    Get team by slug or ID, including subscription and projects.

    Args:
        team_slug_or_id: Team slug or ID

    Returns:
        Team record with subscription and projects, or None
    """
    db = await get_db()

    team = await db.team.find_first(
        where={
            "OR": [
                {"id": team_slug_or_id},
                {"slug": team_slug_or_id},
            ]
        },
        include={
            "subscription": True,
            "projects": True,
        },
    )

    return team


async def validate_oauth_token(token: str, project_id_or_slug: str) -> dict | None:
    """
    Validate an OAuth access token and check project access.

    Args:
        token: The OAuth access token (snipara_at_...)
        project_id_or_slug: The project ID or slug being accessed

    Returns:
        Token record if valid, None otherwise
    """
    db = await get_db()

    # Hash the token to compare with stored hash
    token_hash = hash_api_key(token)

    # Find the OAuth token by hash and project (matching either id or slug)
    oauth_token = await db.oauthtoken.find_first(
        where={
            "accessTokenHash": token_hash,
            "project": {
                "OR": [
                    {"id": project_id_or_slug},
                    {"slug": project_id_or_slug},
                ]
            },
        },
        include={
            "project": {
                "include": {
                    "team": {
                        "include": {
                            "subscription": True,
                        }
                    }
                }
            },
            "user": True,
        },
    )

    if not oauth_token:
        return None

    # Check if revoked
    if oauth_token.revokedAt:
        return None

    # Check if expired
    if oauth_token.accessExpiresAt < datetime.now(timezone.utc):
        return None

    # Update last used timestamp
    await db.oauthtoken.update(
        where={"id": oauth_token.id},
        data={"lastUsedAt": datetime.now(timezone.utc)},
    )

    return {
        "id": oauth_token.id,
        "user_id": oauth_token.userId,
        "project_id": oauth_token.projectId,
        "project": oauth_token.project,
        "scope": oauth_token.scope,
        "auth_type": "oauth",
    }


async def get_project_settings(project_id_or_slug: str) -> dict | None:
    """
    Get project automation settings from database.

    These settings are configured in the dashboard and used by the MCP server
    to customize query behavior (max tokens, search mode, etc.).

    Args:
        project_id_or_slug: The project ID or slug

    Returns:
        Dict with automation settings or None if project not found
    """
    db = await get_db()

    # Find by ID or slug
    project = await db.project.find_first(
        where={
            "OR": [
                {"id": project_id_or_slug},
                {"slug": project_id_or_slug},
            ]
        },
        include=None,  # No relations needed, just the project fields
    )

    if not project:
        return None

    return {
        "automation_client": project.automationClient,
        "auto_inject_context": project.autoInjectContext,
        "track_accessed_files": project.trackAccessedFiles,
        "preserve_on_compaction": project.preserveOnCompaction,
        "restore_on_session_start": project.restoreOnSessionStart,
        "enrich_prompts": project.enrichPrompts,
        "max_tokens_per_query": project.maxTokensPerQuery,
        "search_mode": project.searchMode,
        "include_summaries": project.includeSummaries,
    }
