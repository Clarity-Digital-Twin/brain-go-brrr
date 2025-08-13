"""Application-layer ports (hexagonal)."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class CachePort(Protocol):
    """Cache port interface."""

    def get(self, key: str) -> Any:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ...

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        ...

    def close(self) -> None:
        """Close cache connection."""
        ...


@runtime_checkable
class AsyncCachePort(Protocol):
    """Async cache port interface."""

    async def aget(self, key: str) -> Any:
        """Get value from cache."""
        ...

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ...

    async def adelete(self, key: str) -> None:
        """Delete key from cache."""
        ...

    async def aclose(self) -> None:
        """Close cache connection."""
        ...