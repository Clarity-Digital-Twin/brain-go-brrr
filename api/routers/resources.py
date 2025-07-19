"""Resource monitoring endpoints."""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resources", tags=["resources"])


@router.get("/gpu")
async def get_gpu_resources():
    """Get GPU resource utilization."""
    try:
        import GPUtil

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
    except ImportError:
        return {"gpus": [], "message": "GPUtil not installed"}
    except Exception as e:
        return {"gpus": [], "error": str(e)}


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
