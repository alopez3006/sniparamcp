"""Database connection module using Prisma."""

from prisma import Prisma

# Global Prisma client instance
_client: Prisma | None = None


async def get_db() -> Prisma:
    """Get or create the Prisma client instance."""
    global _client
    if _client is None:
        _client = Prisma()
        await _client.connect()
    return _client


async def close_db() -> None:
    """Close the database connection."""
    global _client
    if _client is not None:
        await _client.disconnect()
        _client = None
