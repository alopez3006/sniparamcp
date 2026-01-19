"""Query result caching service.

Caches context query results for Team+ plans to improve response times
for similar queries.
"""

import hashlib
import json
import logging
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

# Redis client - lazy initialized
_redis: Any | None = None


async def get_redis():
    """Get or create Redis connection."""
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis

            _redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return None
    return _redis


def _generate_cache_key(project_id: str, query: str, max_tokens: int) -> str:
    """Generate a cache key for a query."""
    # Normalize query (lowercase, strip whitespace)
    normalized_query = query.lower().strip()

    # Create hash of normalized query
    query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()[:16]

    return f"rlm:context:{project_id}:{query_hash}:{max_tokens}"


class QueryCache:
    """Cache for context query results."""

    # Cache TTL in seconds (1 hour)
    DEFAULT_TTL = 3600

    def __init__(self, project_id: str):
        """Initialize cache for a project.

        Args:
            project_id: The project ID
        """
        self.project_id = project_id

    async def get(
        self, query: str, max_tokens: int
    ) -> dict[str, Any] | None:
        """Get cached result for a query.

        Args:
            query: The query string
            max_tokens: Token budget

        Returns:
            Cached result dict or None if not found
        """
        redis = await get_redis()
        if redis is None:
            return None

        try:
            cache_key = _generate_cache_key(self.project_id, query, max_tokens)
            cached = await redis.get(cache_key)

            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached)

            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(
        self,
        query: str,
        max_tokens: int,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a query result.

        Args:
            query: The query string
            max_tokens: Token budget
            result: The result to cache
            ttl: Time-to-live in seconds (default: 1 hour)

        Returns:
            True if cached successfully
        """
        redis = await get_redis()
        if redis is None:
            return False

        try:
            cache_key = _generate_cache_key(self.project_id, query, max_tokens)
            ttl = ttl or self.DEFAULT_TTL

            await redis.setex(
                cache_key,
                ttl,
                json.dumps(result),
            )

            logger.debug(f"Cached result for query: {query[:50]}... (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            pattern: Optional pattern to match (e.g., for specific queries).
                    If None, invalidates all cache for this project.

        Returns:
            Number of keys deleted
        """
        redis = await get_redis()
        if redis is None:
            return 0

        try:
            if pattern:
                cache_pattern = f"rlm:context:{self.project_id}:{pattern}*"
            else:
                cache_pattern = f"rlm:context:{self.project_id}:*"

            # Find all matching keys
            keys = []
            async for key in redis.scan_iter(match=cache_pattern):
                keys.append(key)

            if keys:
                deleted = await redis.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for project {self.project_id}")
                return deleted

            return 0

        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for this project.

        Returns:
            Dict with cache statistics
        """
        redis = await get_redis()
        if redis is None:
            return {"available": False}

        try:
            cache_pattern = f"rlm:context:{self.project_id}:*"

            # Count keys
            count = 0
            total_size = 0
            async for key in redis.scan_iter(match=cache_pattern):
                count += 1
                # Get size of each key's value
                value = await redis.get(key)
                if value:
                    total_size += len(value)

            return {
                "available": True,
                "project_id": self.project_id,
                "cached_queries": count,
                "total_size_bytes": total_size,
                "ttl_seconds": self.DEFAULT_TTL,
            }

        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"available": False, "error": str(e)}


class SimilarQueryCache:
    """Cache that also matches semantically similar queries.

    For Team+ plans, this provides fuzzy matching for similar queries
    to maximize cache hits.
    """

    # Similarity threshold (0-1, higher = more similar required)
    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, project_id: str):
        """Initialize similar query cache.

        Args:
            project_id: The project ID
        """
        self.project_id = project_id
        self.base_cache = QueryCache(project_id)

    async def get(
        self, query: str, max_tokens: int
    ) -> tuple[dict[str, Any] | None, bool]:
        """Get cached result, including similar query matches.

        Args:
            query: The query string
            max_tokens: Token budget

        Returns:
            Tuple of (cached result, is_exact_match)
        """
        # First try exact match
        exact_result = await self.base_cache.get(query, max_tokens)
        if exact_result:
            return exact_result, True

        # For similar query matching, we'd need embeddings
        # This is a placeholder for future semantic matching
        # TODO: Implement semantic similarity matching using embeddings

        return None, False

    async def set(
        self,
        query: str,
        max_tokens: int,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a query result.

        Args:
            query: The query string
            max_tokens: Token budget
            result: The result to cache
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        return await self.base_cache.set(query, max_tokens, result, ttl)


# Factory function
def get_cache(project_id: str, use_similarity: bool = False) -> QueryCache:
    """Get a cache instance for a project.

    Args:
        project_id: The project ID
        use_similarity: Whether to use similar query matching (Team+ only)

    Returns:
        QueryCache or SimilarQueryCache instance
    """
    if use_similarity:
        return SimilarQueryCache(project_id)
    return QueryCache(project_id)
