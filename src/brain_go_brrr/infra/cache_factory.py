"""Cache factory following SOLID's Dependency Inversion and Factory patterns.

This module provides cache implementations that conform to the CachePort protocol,
allowing the application layer to depend on abstractions rather than concrete
implementations.
"""

import os
from typing import Any

from brain_go_brrr.core.cache_port import CachePort
from brain_go_brrr.infra.cache import RedisCache as InfraRedisCache


class MemoryCache:
    """Simple in-memory cache implementation.

    Implements the CachePort protocol for testing and development.
    """

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        self._store: dict[str, Any] = {}
        self._ttls: dict[str, float] = {}

    def get(self, key: str) -> Any | None:
        """Get value from memory cache."""
        # TODO: Check TTL expiry
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set value in memory cache."""
        self._store[key] = value
        if ttl_seconds:
            import time

            self._ttls[key] = time.time() + ttl_seconds

    def delete(self, key: str) -> bool:
        """Delete from memory cache."""
        existed = key in self._store
        self._store.pop(key, None)
        self._ttls.pop(key, None)
        return existed

    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        return key in self._store

    def clear(self) -> None:
        """Clear memory cache."""
        self._store.clear()
        self._ttls.clear()


def get_cache(backend: str | None = None) -> CachePort:
    """Factory function to get appropriate cache implementation.

    This follows the Factory pattern and Dependency Inversion Principle:
    - The application depends on CachePort (abstraction)
    - This factory decides which concrete implementation to provide
    - Configuration is handled here, not in the application layer

    Args:
        backend: Cache backend to use ("redis", "memory", or None for env-based)

    Returns:
        Cache implementation conforming to CachePort protocol
    """
    if backend is None:
        backend = os.getenv("CACHE_BACKEND", "memory").lower()

    if backend == "redis":
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return InfraRedisCache(url=redis_url)  # type: ignore
    elif backend == "memory":
        return MemoryCache()
    else:
        raise ValueError(f"Unknown cache backend: {backend}")


__all__ = ["MemoryCache", "get_cache"]
