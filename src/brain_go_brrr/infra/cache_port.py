"""Cache port (interface) following SOLID's Interface Segregation Principle.

This defines the contract for cache implementations without depending on
any specific infrastructure (Redis, Memory, etc).
"""

from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class CachePort(Protocol, Generic[T]):
    """Interface for cache operations.

    This protocol defines the minimal interface that any cache
    implementation must provide. Following ISP, we keep it small
    and focused on essential operations only.
    """

    def get(self, key: str) -> T | None:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        ...

    def set(self, key: str, value: T, ttl_seconds: int | None = None) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (None = no expiry)
        """
        ...

    def delete(self, key: str) -> bool:
        """Remove value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if key didn't exist
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and hasn't expired
        """
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


class AsyncCachePort(Protocol, Generic[T]):
    """Async version of cache interface for high-performance scenarios."""

    async def get(self, key: str) -> T | None: ...
    async def set(self, key: str, value: T, ttl_seconds: int | None = None) -> None: ...
    async def delete(self, key: str) -> bool: ...
    async def exists(self, key: str) -> bool: ...
    async def clear(self) -> None: ...


__all__ = ["AsyncCachePort", "CachePort"]
