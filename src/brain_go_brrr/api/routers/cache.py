"""Cache management endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from brain_go_brrr.api.auth import verify_cache_clear_permission
from brain_go_brrr.api.cache import get_cache
from brain_go_brrr.api.schemas import CacheWarmupRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats")  # type: ignore[misc]
async def get_cache_stats(cache_client: Any = Depends(get_cache)) -> dict[str, Any]:
    """Get Redis cache statistics."""
    if not cache_client or not cache_client.connected:
        return {"status": "unavailable", "message": "Cache not available"}

    try:
        stats = cache_client.get_stats()
        return dict(stats)
    except (ConnectionError, TimeoutError) as e:
        # Redis connection issues - expected and recoverable
        logger.warning(f"Redis connection issue getting stats: {e}")
        return {"status": "disconnected", "error": str(e)}
    except AttributeError as e:
        # Cache client method missing
        logger.error(f"Cache client missing get_stats method: {e}")
        return {"status": "error", "error": "Invalid cache client"}


@router.delete("/clear")  # type: ignore[misc]
async def clear_cache(
    pattern: str = "eeg_analysis:*",
    cache_client: Any = Depends(get_cache),
    _authorized: bool = Depends(verify_cache_clear_permission),
) -> dict[str, Any]:
    """Clear cache entries matching pattern.

    Args:
        pattern: Redis key pattern to match
        cache_client: Redis cache client

    Returns:
        Number of keys deleted
    """
    if not cache_client or not cache_client.connected:
        return {"status": "unavailable", "message": "Cache not available", "deleted": 0}

    try:
        deleted_count = cache_client.clear_pattern(pattern)
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "pattern": pattern,
        }
    except (ConnectionError, TimeoutError) as e:
        # Redis connection issues
        logger.error(f"Redis connection error clearing cache: {e}")
        return {"status": "unavailable", "message": "Cache temporarily unavailable", "deleted": 0}
    except ValueError as e:
        # Invalid pattern
        logger.error(f"Invalid cache pattern: {e}")
        raise HTTPException(status_code=400, detail="Invalid cache pattern") from e


@router.post("/warmup")  # type: ignore[misc]
async def warmup_cache(
    request: CacheWarmupRequest,  # noqa: ARG001
    cache_client: Any = Depends(get_cache),
) -> dict[str, Any]:
    """Pre-warm cache with analysis results.

    Useful for demo/testing to ensure fast responses.
    """
    if not cache_client or not cache_client.connected:
        return {"status": "unavailable", "message": "Cache not available", "warmed": 0}

    # TODO: Implement cache warmup logic
    return {
        "status": "not_implemented",
        "message": "Cache warmup not yet implemented",
    }
