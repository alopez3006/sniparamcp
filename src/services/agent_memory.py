"""Agent Memory Service for Phase 8.2.

Provides semantic memory storage and recall for AI agents.
Memories can have types (FACT, DECISION, LEARNING, etc.), scopes,
and TTL with confidence decay over time.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ..db import get_db
from .cache import get_redis
from .embeddings import EMBEDDING_DIMENSION, get_embeddings_service

logger = logging.getLogger(__name__)

# Cache key prefixes
MEMORY_EMBEDDING_PREFIX = "rlm:mem_emb:"  # Memory embedding storage
MEMORY_EMBEDDING_TTL = 60 * 60 * 24 * 7  # 7 days default

# Confidence decay settings
CONFIDENCE_DECAY_RATE = 0.01  # 1% decay per day
MIN_CONFIDENCE = 0.1  # Minimum confidence after decay


def calculate_confidence_decay(
    initial_confidence: float,
    created_at: datetime,
    last_accessed_at: datetime | None = None,
) -> float:
    """Calculate decayed confidence based on age and access patterns.

    Args:
        initial_confidence: Original confidence (0-1)
        created_at: When memory was created
        last_accessed_at: Last time memory was accessed (boosts confidence)

    Returns:
        Decayed confidence value (0-1)
    """
    now = datetime.now(UTC)

    # Use last access time if available, otherwise creation time
    reference_time = last_accessed_at or created_at

    # Ensure reference_time is timezone-aware (database may return naive datetimes)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)

    days_since_reference = (now - reference_time).days

    # Apply exponential decay
    decay_factor = (1 - CONFIDENCE_DECAY_RATE) ** days_since_reference
    decayed = initial_confidence * decay_factor

    return max(decayed, MIN_CONFIDENCE)


def _is_valid_embedding(embedding: Any) -> bool:
    """Validate that an embedding has the correct structure and dimensions.

    Args:
        embedding: The embedding to validate

    Returns:
        True if embedding is valid (list of EMBEDDING_DIMENSION floats)
    """
    if not isinstance(embedding, list):
        return False
    if len(embedding) != EMBEDDING_DIMENSION:
        return False
    # Check that all elements are numbers (int or float)
    return all(isinstance(x, (int, float)) for x in embedding)


async def _get_memory_embedding(memory_id: str) -> list[float] | None:
    """Get cached embedding for a memory from Redis.

    Args:
        memory_id: The memory ID

    Returns:
        Embedding vector or None if not cached
    """
    redis = await get_redis()
    if redis is None:
        return None

    try:
        key = f"{MEMORY_EMBEDDING_PREFIX}{memory_id}"
        cached = await redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        logger.warning(f"Error getting memory embedding: {e}")
        return None


async def _get_memory_embeddings_batch(memory_ids: list[str]) -> dict[str, list[float]]:
    """Get cached embeddings for multiple memories from Redis using MGET.

    Args:
        memory_ids: List of memory IDs

    Returns:
        Dict mapping memory_id to embedding vector (only for cached entries)
    """
    if not memory_ids:
        return {}

    redis = await get_redis()
    if redis is None:
        return {}

    try:
        keys = [f"{MEMORY_EMBEDDING_PREFIX}{mid}" for mid in memory_ids]
        values = await redis.mget(keys)

        result = {}
        for mid, value in zip(memory_ids, values):
            if value:
                try:
                    embedding = json.loads(value)
                    # Validate embedding structure and dimensions
                    if _is_valid_embedding(embedding):
                        result[mid] = embedding
                    else:
                        logger.warning(
                            f"Invalid embedding for memory {mid}: "
                            f"expected {EMBEDDING_DIMENSION} dimensions, "
                            f"got {len(embedding) if isinstance(embedding, list) else 'non-list'}"
                        )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse embedding JSON for memory {mid}")
        return result
    except Exception as e:
        logger.warning(f"Error getting memory embeddings batch: {e}")
        return {}


async def _store_memory_embedding(
    memory_id: str,
    embedding: list[float],
    ttl: int = MEMORY_EMBEDDING_TTL,
) -> bool:
    """Store embedding for a memory in Redis.

    Args:
        memory_id: The memory ID
        embedding: The embedding vector
        ttl: Time-to-live in seconds

    Returns:
        True if stored successfully
    """
    redis = await get_redis()
    if redis is None:
        return False

    try:
        key = f"{MEMORY_EMBEDDING_PREFIX}{memory_id}"
        await redis.setex(key, ttl, json.dumps(embedding))
        return True
    except Exception as e:
        logger.warning(f"Error storing memory embedding: {e}")
        return False


async def _delete_memory_embedding(memory_id: str) -> bool:
    """Delete embedding for a memory from Redis.

    Args:
        memory_id: The memory ID

    Returns:
        True if deleted
    """
    redis = await get_redis()
    if redis is None:
        return False

    try:
        key = f"{MEMORY_EMBEDDING_PREFIX}{memory_id}"
        await redis.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Error deleting memory embedding: {e}")
        return False


async def store_memory(
    project_id: str,
    content: str,
    memory_type: str = "fact",
    scope: str = "project",
    category: str | None = None,
    ttl_days: int | None = None,
    related_to: list[str] | None = None,
    document_refs: list[str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Store a new memory with semantic embedding.

    Args:
        project_id: The project ID
        content: Memory content
        memory_type: Type of memory (fact, decision, learning, preference, todo, context)
        scope: Visibility scope (agent, project, team, user)
        category: Optional grouping category
        ttl_days: Days until expiration (null = permanent)
        related_to: IDs of related memories
        document_refs: Referenced document paths
        source: What created this memory

    Returns:
        Dict with memory_id, created status, and message
    """
    db = await get_db()

    # Calculate expiration
    expires_at = None
    if ttl_days:
        expires_at = datetime.now(UTC) + timedelta(days=ttl_days)

    # Map string types to enum values (Prisma expects uppercase)
    memory_type_upper = memory_type.upper()
    scope_upper = scope.upper()

    # Create memory in database
    memory = await db.agentmemory.create(
        data={
            "projectId": project_id,
            "content": content,
            "type": memory_type_upper,
            "scope": scope_upper,
            "category": category,
            "expiresAt": expires_at,
            "relatedMemoryIds": related_to or [],
            "documentRefs": document_refs or [],
            "source": source,
            "confidence": 1.0,
            "accessCount": 0,
        }
    )

    # Generate and store embedding
    try:
        embeddings_service = get_embeddings_service()
        embedding = await embeddings_service.embed_text_async(content)

        # TTL for embedding based on memory TTL
        embedding_ttl = MEMORY_EMBEDDING_TTL
        if ttl_days:
            embedding_ttl = min(ttl_days * 24 * 60 * 60, MEMORY_EMBEDDING_TTL)

        await _store_memory_embedding(memory.id, embedding, embedding_ttl)
        logger.info(f"Stored memory {memory.id} with embedding")
    except Exception as e:
        logger.warning(f"Failed to generate embedding for memory {memory.id}: {e}")
        # Memory is still created, just without embedding

    return {
        "memory_id": memory.id,
        "content": memory.content,
        "type": memory_type,
        "scope": scope,
        "category": category,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "created": True,
        "message": f"Memory stored successfully (ID: {memory.id})",
    }


async def semantic_recall(
    project_id: str,
    query: str,
    memory_type: str | None = None,
    scope: str | None = None,
    category: str | None = None,
    limit: int = 5,
    min_relevance: float = 0.5,
    include_expired: bool = False,
) -> dict[str, Any]:
    """Semantically recall relevant memories based on a query.

    Args:
        project_id: The project ID
        query: Search query
        memory_type: Filter by type
        scope: Filter by scope
        category: Filter by category
        limit: Maximum memories to return
        min_relevance: Minimum relevance score (0-1)
        include_expired: Include expired memories

    Returns:
        Dict with recalled memories and metadata
    """
    import time
    start_time = time.time()

    db = await get_db()
    embeddings_service = get_embeddings_service()

    # Build filter
    where: dict[str, Any] = {"projectId": project_id}
    if memory_type:
        where["type"] = memory_type.upper()
    if scope:
        where["scope"] = scope.upper()
    if category:
        where["category"] = category
    if not include_expired:
        where["OR"] = [
            {"expiresAt": None},
            {"expiresAt": {"gt": datetime.now(UTC)}},
        ]

    # Get all matching memories
    memories = await db.agentmemory.find_many(
        where=where,
        order={"createdAt": "desc"},
        take=500,  # Limit to prevent huge queries
    )

    if not memories:
        return {
            "memories": [],
            "total_searched": 0,
            "query": query,
            "timing_ms": int((time.time() - start_time) * 1000),
        }

    # Generate query embedding
    try:
        query_embedding = await embeddings_service.embed_text_async(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        # Fallback to text search if embedding fails
        return await _text_search_fallback(
            memories, query, limit, min_relevance, start_time
        )

    # Batch fetch all cached embeddings
    memory_ids = [m.id for m in memories]
    cached_embeddings = await _get_memory_embeddings_batch(memory_ids)
    logger.debug(f"Cache hit: {len(cached_embeddings)}/{len(memories)} embeddings")

    # Identify memories needing embedding generation
    memory_embeddings: list[tuple[Any, list[float]]] = []
    memories_to_embed: list[Any] = []

    for memory in memories:
        if memory.id in cached_embeddings:
            memory_embeddings.append((memory, cached_embeddings[memory.id]))
        else:
            memories_to_embed.append(memory)

    # Batch generate embeddings for cache misses (limit to prevent timeout)
    if memories_to_embed:
        # Limit on-the-fly generation to prevent long delays
        max_to_embed = min(len(memories_to_embed), 10)
        for memory in memories_to_embed[:max_to_embed]:
            try:
                embedding = await embeddings_service.embed_text_async(memory.content)
                await _store_memory_embedding(memory.id, embedding)
                memory_embeddings.append((memory, embedding))
            except Exception as e:
                logger.warning(f"Failed to embed memory {memory.id}: {e}")
                continue
        if len(memories_to_embed) > max_to_embed:
            logger.info(f"Skipped embedding {len(memories_to_embed) - max_to_embed} memories to prevent timeout")

    if not memory_embeddings:
        return {
            "memories": [],
            "total_searched": len(memories),
            "query": query,
            "timing_ms": int((time.time() - start_time) * 1000),
        }

    # Calculate similarities
    doc_embeddings = [emb for _, emb in memory_embeddings]
    try:
        similarities = embeddings_service.cosine_similarity(query_embedding, doc_embeddings)
    except ValueError as e:
        logger.error(
            f"Failed to calculate similarities due to dimension mismatch: {e}. "
            "This indicates corrupted embeddings in cache. Falling back to text search."
        )
        # Fallback to text search if embeddings are corrupted
        return await _text_search_fallback(
            memories, query, limit, min_relevance, start_time
        )

    # Score and rank
    results = []
    for (memory, _), similarity in zip(memory_embeddings, similarities):
        # Apply confidence decay
        decayed_confidence = calculate_confidence_decay(
            memory.confidence,
            memory.createdAt,
            memory.lastAccessedAt,
        )

        # Combined relevance = similarity * confidence
        relevance = similarity * decayed_confidence

        if relevance >= min_relevance:
            results.append({
                "memory_id": memory.id,
                "content": memory.content,
                "type": memory.type.lower(),
                "scope": memory.scope.lower(),
                "category": memory.category,
                "relevance": round(relevance, 4),
                "confidence": round(decayed_confidence, 4),
                "created_at": memory.createdAt.isoformat(),
                "last_accessed_at": memory.lastAccessedAt.isoformat() if memory.lastAccessedAt else None,
                "access_count": memory.accessCount,
            })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    results = results[:limit]

    # Batch update access counts for returned memories
    if results:
        result_ids = [r["memory_id"] for r in results]
        try:
            await db.agentmemory.update_many(
                where={"id": {"in": result_ids}},
                data={"lastAccessedAt": datetime.now(UTC)},
            )
            # Note: update_many doesn't support increment, so we do a raw query
            # For now, skip accessCount increment to optimize latency
            # TODO: Use raw SQL for atomic increment if needed
        except Exception as e:
            logger.warning(f"Failed to batch update access counts: {e}")

    return {
        "memories": results,
        "total_searched": len(memories),
        "query": query,
        "timing_ms": int((time.time() - start_time) * 1000),
    }


async def _text_search_fallback(
    memories: list,
    query: str,
    limit: int,
    min_relevance: float,
    start_time: float,
) -> dict[str, Any]:
    """Fallback to text search if embedding fails.

    Uses simple keyword matching as a degraded mode.
    """
    import time

    query_terms = set(query.lower().split())
    results = []

    for memory in memories:
        content_terms = set(memory.content.lower().split())
        overlap = len(query_terms & content_terms)

        if overlap > 0:
            # Simple relevance based on term overlap
            relevance = overlap / max(len(query_terms), 1)

            if relevance >= min_relevance:
                decayed_confidence = calculate_confidence_decay(
                    memory.confidence,
                    memory.createdAt,
                    memory.lastAccessedAt,
                )

                results.append({
                    "memory_id": memory.id,
                    "content": memory.content,
                    "type": memory.type.lower(),
                    "scope": memory.scope.lower(),
                    "category": memory.category,
                    "relevance": round(relevance * decayed_confidence, 4),
                    "confidence": round(decayed_confidence, 4),
                    "created_at": memory.createdAt.isoformat(),
                    "last_accessed_at": memory.lastAccessedAt.isoformat() if memory.lastAccessedAt else None,
                    "access_count": memory.accessCount,
                })

    results.sort(key=lambda x: x["relevance"], reverse=True)
    results = results[:limit]

    return {
        "memories": results,
        "total_searched": len(memories),
        "query": query,
        "timing_ms": int((time.time() - start_time) * 1000),
    }


async def list_memories(
    project_id: str,
    memory_type: str | None = None,
    scope: str | None = None,
    category: str | None = None,
    search: str | None = None,
    limit: int = 20,
    offset: int = 0,
    include_expired: bool = False,
) -> dict[str, Any]:
    """List memories with optional filters.

    Args:
        project_id: The project ID
        memory_type: Filter by type
        scope: Filter by scope
        category: Filter by category
        search: Text search in content
        limit: Maximum memories to return
        offset: Pagination offset
        include_expired: Include expired memories

    Returns:
        Dict with memories list and pagination info
    """
    db = await get_db()

    # Build filter
    where: dict[str, Any] = {"projectId": project_id}
    if memory_type:
        where["type"] = memory_type.upper()
    if scope:
        where["scope"] = scope.upper()
    if category:
        where["category"] = category
    if search:
        where["content"] = {"contains": search, "mode": "insensitive"}
    if not include_expired:
        where["OR"] = [
            {"expiresAt": None},
            {"expiresAt": {"gt": datetime.now(UTC)}},
        ]

    # Count total
    total_count = await db.agentmemory.count(where=where)

    # Get memories
    memories = await db.agentmemory.find_many(
        where=where,
        order={"createdAt": "desc"},
        skip=offset,
        take=limit,
    )

    results = []
    for memory in memories:
        decayed_confidence = calculate_confidence_decay(
            memory.confidence,
            memory.createdAt,
            memory.lastAccessedAt,
        )

        results.append({
            "memory_id": memory.id,
            "content": memory.content,
            "type": memory.type.lower(),
            "scope": memory.scope.lower(),
            "category": memory.category,
            "confidence": round(decayed_confidence, 4),
            "source": memory.source,
            "created_at": memory.createdAt.isoformat(),
            "expires_at": memory.expiresAt.isoformat() if memory.expiresAt else None,
            "access_count": memory.accessCount,
        })

    return {
        "memories": results,
        "total_count": total_count,
        "has_more": (offset + limit) < total_count,
    }


async def delete_memories(
    project_id: str,
    memory_id: str | None = None,
    memory_type: str | None = None,
    category: str | None = None,
    older_than_days: int | None = None,
) -> dict[str, Any]:
    """Delete memories matching criteria.

    Args:
        project_id: The project ID
        memory_id: Specific memory to delete
        memory_type: Delete all of this type
        category: Delete all in this category
        older_than_days: Delete memories older than N days

    Returns:
        Dict with deleted count and message
    """
    db = await get_db()

    # Build filter
    where: dict[str, Any] = {"projectId": project_id}

    if memory_id:
        where["id"] = memory_id
    if memory_type:
        where["type"] = memory_type.upper()
    if category:
        where["category"] = category
    if older_than_days:
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        where["createdAt"] = {"lt": cutoff}

    # Get IDs to delete embeddings
    to_delete = await db.agentmemory.find_many(where=where)
    memory_ids = [m.id for m in to_delete]

    # Delete memories
    result = await db.agentmemory.delete_many(where=where)
    deleted_count = result

    # Delete embeddings from Redis
    for mid in memory_ids:
        await _delete_memory_embedding(mid)

    message = f"Deleted {deleted_count} memories"
    if memory_id:
        message = f"Memory {memory_id} deleted" if deleted_count > 0 else f"Memory {memory_id} not found"

    return {
        "deleted_count": deleted_count,
        "message": message,
    }
