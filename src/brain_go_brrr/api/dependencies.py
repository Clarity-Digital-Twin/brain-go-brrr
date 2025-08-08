"""Shared dependencies for API endpoints."""

from typing import Any

from brain_go_brrr.api.cache import RedisCache

# Global instances (to be replaced with proper DI)
cache_client: RedisCache | None = None
job_store: dict[str, dict[str, Any]] = {}


async def get_cache() -> RedisCache | None:
    """Get cache client dependency."""
    return cache_client


async def get_job_store() -> dict[str, dict[str, Any]]:
    """Get job store dependency."""
    return job_store
