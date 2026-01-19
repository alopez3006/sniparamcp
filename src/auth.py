"""API key validation module."""

import hashlib
from datetime import datetime

from .db import get_db


def hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(key.encode()).hexdigest()


async def validate_api_key(api_key: str, project_id: str) -> dict | None:
    """
    Validate an API key and check project access.

    Args:
        api_key: The API key from the request header
        project_id: The project being accessed

    Returns:
        API key record if valid, None otherwise
    """
    db = await get_db()

    # Hash the provided key to compare with stored hash
    key_hash = hash_api_key(api_key)

    # Find the API key by hash
    api_key_record = await db.apikey.find_first(
        where={
            "keyHash": key_hash,
            "OR": [
                {"projectId": project_id},
                {"projectId": None},  # Global API key (no project restriction)
            ],
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

    if not api_key_record:
        return None

    # Check if key is expired
    if api_key_record.expiresAt and api_key_record.expiresAt < datetime.utcnow():
        return None

    # Update last used timestamp
    await db.apikey.update(
        where={"id": api_key_record.id},
        data={"lastUsedAt": datetime.utcnow()},
    )

    return {
        "id": api_key_record.id,
        "name": api_key_record.name,
        "user_id": api_key_record.userId,
        "project_id": api_key_record.projectId,
        "project": api_key_record.project,
    }


async def get_project_with_team(project_id: str) -> dict | None:
    """
    Get project details including team and subscription info.

    Args:
        project_id: The project ID

    Returns:
        Project record with team and subscription, or None
    """
    db = await get_db()

    project = await db.project.find_unique(
        where={"id": project_id},
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
