"""Redis infrastructure for caching and job management."""

from .pool import RedisConnectionPool, close_redis_pool, get_redis_pool

# Only expose essential public API
__all__ = [
    "RedisConnectionPool",
    "close_redis_pool",
    "get_redis_pool",
]
