"""Shared dependencies for API endpoints."""

from typing import Any

from brain_go_brrr.api.cache import APIRedisCache  # P1 FIX: Use renamed class

# Global instances (to be replaced with proper DI)
cache_client: APIRedisCache | None = None  # P1 FIX: Updated type
job_store: dict[str, dict[str, Any]] = {}


async def get_cache() -> APIRedisCache | None:  # P1 FIX: Updated return type
    """Get cache client dependency."""
    return cache_client


async def get_job_store() -> dict[str, dict[str, Any]]:
    """Get job store dependency."""
    return job_store
