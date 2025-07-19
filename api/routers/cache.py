"""Cache management endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.cache import get_cache
from api.schemas import CacheWarmupRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats")
async def get_cache_stats(cache_client=Depends(get_cache)):
    """Get Redis cache statistics."""
    if not cache_client or not cache_client.connected:
        return {"status": "unavailable", "message": "Cache not available"}

    try:
        stats = cache_client.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"status": "error", "error": str(e)}


@router.delete("/clear")
async def clear_cache(
    pattern: str = "eeg_analysis:*",
    cache_client=Depends(get_cache),
):
    """Clear cache entries matching pattern.

    Args:
        pattern: Redis key pattern to match
        cache_client: Redis cache client

    Returns:
        Number of keys deleted
    """
    if not cache_client or not cache_client.connected:
        raise HTTPException(status_code=503, detail="Cache not available")

    try:
        deleted_count = cache_client.clear_pattern(pattern)
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "pattern": pattern,
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warmup")
async def warmup_cache(
    request: CacheWarmupRequest,
    cache_client=Depends(get_cache),
):
    """Pre-warm cache with analysis results.

    Useful for demo/testing to ensure fast responses.
    """
    if not cache_client or not cache_client.connected:
        raise HTTPException(status_code=503, detail="Cache not available")

    # TODO: Implement cache warmup logic
    return {
        "status": "not_implemented",
        "message": "Cache warmup not yet implemented",
    }
