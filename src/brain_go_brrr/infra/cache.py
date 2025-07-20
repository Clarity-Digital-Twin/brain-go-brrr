"""Cache protocol and Redis implementation."""

import logging
from typing import Any, Protocol, runtime_checkable

from brain_go_brrr.infra.redis import RedisConnectionPool, get_redis_pool

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
            return self.pool.execute("get", key)
        except Exception:
            return None

    def set(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """Set value in cache with optional expiry."""
        if not self.connected:
            return False

        try:
            if expiry:
                return bool(self.pool.execute("setex", key, expiry, value))
            else:
                return bool(self.pool.execute("set", key, value))
        except Exception:
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
                key_list = keys_result if isinstance(keys_result, list) else list(keys_result)

                if not key_list:
                    return 0

                # Delete all matching keys with separate error handling
                try:
                    delete_count = client.delete(*key_list)
                    # Redis delete returns int of deleted keys
                    return int(delete_count) if delete_count is not None else 0
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


# Global cache instance
_cache: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache


def close_cache() -> None:
    """Close global cache instance."""
    global _cache
    if _cache is not None:
        _cache.pool.close()
        _cache = None
