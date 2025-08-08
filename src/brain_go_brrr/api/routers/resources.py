"""Resource monitoring endpoints."""

import logging
from typing import Any

from fastapi import APIRouter

# Try importing GPUtil
try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    GPUtil = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("/gpu")
async def get_gpu_resources() -> dict[str, Any]:
    """Get GPU resource utilization."""
    if not HAS_GPUTIL or GPUtil is None:
        return {"gpus": [], "message": "GPUtil not installed"}

    try:
        gpus = GPUtil.getGPUs()
        return {
            "gpus": [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "gpu_load": gpu.load * 100,
                    "temperature": gpu.temperature,
                }
                for gpu in gpus
            ]
        }
    except (AttributeError, RuntimeError) as e:
        # GPUtil access errors - expected when no GPU or drivers missing
        logger.warning(f"GPU access error (likely no GPU available): {e}")
        return {"gpus": [], "error": f"GPU not available: {e!s}"}


@router.get("/memory")
async def get_memory_resources() -> dict[str, Any]:
    """Get system memory utilization."""
    import psutil

    memory = psutil.virtual_memory()
    return {
        "used": memory.used,
        "available": memory.available,
        "percent": memory.percent,
        "total": memory.total,
        "free": memory.free,
    }
