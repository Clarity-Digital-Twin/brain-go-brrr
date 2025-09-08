"""Cache protocol and implementations.

Unified cache module combining protocol definition, Redis implementation,
and in-memory cache for testing.
"""

import logging
import os
import time
from typing import Any, Protocol, runtime_checkable

from brain_go_brrr.infra.redis import RedisConnectionPool, get_redis_pool
from brain_go_brrr.infra.serialization import deserialize_value, serialize_value

logger = logging.getLogger(__name__)


class CacheBackendError(Exception):
    """Base exception for cache backend errors."""

    pass


class CacheConnectionError(CacheBackendError):
    """Raised when cache backend connection fails."""

    pass


class CacheTimeoutError(CacheBackendError):
    """Raised when cache operation times out."""

    pass


@runtime_checkable
class RedisCacheProtocol(Protocol):
    """Protocol for Redis cache operations."""

    @property
    def connected(self) -> bool:
        """Check if cache is connected."""
        ...

    def get(self, key: str) -> Any:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """Set value in cache with optional expiry."""
        ...

    def delete(self, key: str) -> int:
        """Delete key from cache."""
        ...

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Check cache health."""
        ...


class RedisCache:
    """Redis cache implementation."""

    def __init__(self, pool: RedisConnectionPool | None = None) -> None:
        """Initialize Redis cache with optional pool."""
        self.pool = pool or get_redis_pool()
        self._connected = False
        self._check_connection()

    def _check_connection(self) -> None:
        """Check Redis connection."""
        try:
            with self.pool.get_client() as client:
                client.ping()
                self._connected = True
        except Exception:
            self._connected = False

    @property
    def connected(self) -> bool:
        """Check if cache is connected."""
        return self._connected

    def get(self, key: str) -> Any:
        """Get value from cache."""
        if not self.connected:
            return None

        try:
            value = self.pool.execute("get", key)
            if value is None:
                return None

            # Use the serialization registry to deserialize
            return deserialize_value(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """Set value in cache with optional expiry."""
        if not self.connected:
            return False

        try:
            # Use the serialization registry to serialize
            serialized = serialize_value(value)

            if expiry:
                return bool(self.pool.execute("setex", key, expiry, serialized))
            else:
                return bool(self.pool.execute("set", key, serialized))
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> int:
        """Delete key from cache."""
        if not self.connected:
            return 0

        try:
            return int(self.pool.execute("delete", key) or 0)
        except Exception:
            return 0

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        if not self.connected:
            return 0

        try:
            with self.pool.get_client() as client:
                # Redis keys() returns List[bytes] but typing is inconsistent
                keys_result = client.keys(pattern)
                if not keys_result:
                    return 0

                # Ensure we have a list of keys
                key_list = keys_result if isinstance(keys_result, list) else []

                if not key_list:
                    return 0

                # Delete all matching keys with separate error handling
                try:
                    delete_count = client.delete(*key_list)
                    # Redis delete returns int of deleted keys
                    return delete_count if isinstance(delete_count, int) else 0
                except ConnectionError as e:
                    # Log and translate to cache-specific error
                    logger.error(f"Failed to delete {len(key_list)} keys: {e}")
                    raise CacheConnectionError(f"Connection failed while deleting keys: {e}") from e
                except TimeoutError as e:
                    logger.error(f"Timeout deleting {len(key_list)} keys: {e}")
                    raise CacheTimeoutError(f"Operation timed out: {e}") from e
        except ConnectionError as e:
            # Translate to cache-specific error
            raise CacheConnectionError(f"Cache connection failed: {e}") from e
        except TimeoutError as e:
            raise CacheTimeoutError(f"Cache operation timed out: {e}") from e
        except Exception as e:
            # For other unexpected errors, log and re-raise
            logger.error(f"Unexpected error clearing pattern '{pattern}': {e}")
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.pool.get_stats()

    def health_check(self) -> dict[str, Any]:
        """Check cache health."""
        return self.pool.health_check()


class InMemoryCache:
    """In-memory cache implementation for testing and development."""

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        self._store: dict[str, Any] = {}
        self._ttls: dict[str, float] = {}
        self._connected = True

    @property
    def connected(self) -> bool:
        """Always connected for memory cache."""
        return True

    def get(self, key: str) -> Any:
        """Get value from memory cache."""
        # Check TTL expiry
        if key in self._ttls and time.time() > self._ttls[key]:
            # Expired
            del self._store[key]
            del self._ttls[key]
            return None
        return self._store.get(key)

    def set(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """Set value in memory cache."""
        self._store[key] = value
        if expiry:
            self._ttls[key] = time.time() + expiry
        return True

    def delete(self, key: str) -> int:
        """Delete from memory cache."""
        deleted = 1 if key in self._store else 0
        self._store.pop(key, None)
        self._ttls.pop(key, None)
        return deleted

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern using shell-style wildcards.
        
        Args:
            pattern: Shell-style pattern (e.g., 'eeg_*', 'analysis:*')
            
        Returns:
            Number of keys deleted
        """
        import fnmatch

        # P1 FIX: Use fnmatch correctly without regex conversion
        # fnmatch expects shell patterns where * matches any chars
        keys_to_delete = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
        for key in keys_to_delete:
            self._store.pop(key, None)
            self._ttls.pop(key, None)
        return len(keys_to_delete)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "backend": "memory",
            "keys": len(self._store),
            "expired_keys": sum(1 for k, t in self._ttls.items() if time.time() > t),
        }

    def health_check(self) -> dict[str, Any]:
        """Check cache health."""
        return {"healthy": True, "backend": "memory", "keys": len(self._store)}


# Global cache instance
_cache: RedisCache | InMemoryCache | None = None


def create_cache(backend: str | None = None) -> RedisCache | InMemoryCache:
    """Factory function to create appropriate cache implementation.

    Args:
        backend: Cache backend ("redis", "memory", or None for env-based)

    Returns:
        Cache implementation
    """
    if backend is None:
        backend = os.getenv("CACHE_BACKEND", "memory").lower()

    if backend == "redis":
        return RedisCache()
    elif backend == "memory":
        return InMemoryCache()
    else:
        raise ValueError(f"Unknown cache backend: {backend}")


def get_cache() -> RedisCache | InMemoryCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = create_cache()
    return _cache


def close_cache() -> None:
    """Close global cache instance."""
    global _cache
    if _cache is not None:
        if hasattr(_cache, 'pool'):
            _cache.pool.close()
        _cache = None
