"""Cache ports for Clean Architecture.

Domain defines the cache abstractions that infrastructure implements.
"""

from typing import Any, Protocol


class CachePort(Protocol):
    """Port for synchronous cache operations."""

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


class AsyncCachePort(Protocol):
    """Port for asynchronous cache operations."""

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...