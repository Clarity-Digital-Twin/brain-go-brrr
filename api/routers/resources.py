"""Resource monitoring endpoints."""

import importlib
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

if TYPE_CHECKING:
    import GPUtil  # type: ignore[import-not-found]
else:
    GPUtil = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("/gpu")
async def get_gpu_resources() -> dict[str, Any]:
    """Get GPU resource utilization."""
    global GPUtil

    # Try to import GPUtil dynamically if not already loaded
    if GPUtil is None:
        try:
            GPUtil = importlib.import_module("GPUtil")
        except ImportError:
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
async def get_memory_resources():
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
