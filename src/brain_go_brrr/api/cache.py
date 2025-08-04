"""Redis caching for EEG analysis results - delegating to infra.cache."""

import hashlib
import json
import logging
import os
import time
from typing import Any, cast

from brain_go_brrr.infra.cache import get_cache as get_infra_cache

logger = logging.getLogger(__name__)

# Cache versioning - bump when incompatible changes are made
CACHE_VERSION = os.getenv("BRAIN_GO_BRRR_CACHE_VERSION", "v1.0.0")
APP_VERSION = os.getenv("BRAIN_GO_BRRR_APP_VERSION", "0.1.0")


class RedisCache:
    """Redis cache wrapper for EEG analysis results - delegates to infra.cache."""

    def __init__(
        self, *args: Any, **kwargs: Any
    ) -> None:  # noqa: ARG002 - Accepts args for drop-in compatibility, delegates to infra.cache
        """Initialize using infra cache."""
        self._cache = get_infra_cache()
        self.connected = self._cache.connected

    def generate_cache_key(self, file_content: bytes, analysis_type: str = "standard") -> str:
        """Generate cache key from file content hash.

        Includes version info to prevent stale cache hits after updates.
        """
        file_hash = hashlib.sha256(file_content).hexdigest()
        return f"eeg_analysis:{CACHE_VERSION}:{file_hash}:{analysis_type}"

    def get(self, key: str) -> dict | None:
        """Get cached result."""
        try:
            cached = self._cache.get(key)
            if cached:
                logger.info(f"Cache hit for key: {key}")
                data = json.loads(cached) if isinstance(cached, str) else cached
                # Remove metadata before returning
                if isinstance(data, dict) and "_cache_meta" in data:
                    del data["_cache_meta"]
                return cast("dict[Any, Any]", data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: dict, ttl: int = 3600) -> bool:
        """Set cache entry with TTL."""
        try:
            # Add version metadata
            value_with_meta = {
                **value,
                "_cache_meta": {
                    "app_version": APP_VERSION,
                    "cache_version": CACHE_VERSION,
                    "cached_at": time.time(),
                },
            }
            json_value = json.dumps(value_with_meta)
            result = self._cache.set(key, json_value, expiry=ttl)
            if result:
                logger.info(f"Cached result for key: {key} with TTL: {ttl}s")
            return result
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return self._cache.get(key) is not None
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        return self._cache.delete(key) > 0

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        return self._cache.clear_pattern(pattern)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.get_stats()

    def health_check(self) -> dict:
        """Perform health check on Redis connection."""
        return self._cache.health_check()


# Global cache instance
_cache_instance: RedisCache | None = None


def get_cache() -> RedisCache | None:
    """Get or create cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance if _cache_instance.connected else None
