"""Health check endpoints."""

import logging

from fastapi import APIRouter

from brain_go_brrr.utils.time import utc_now

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", name="health_check")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": utc_now().isoformat(),
        "service": "brain-go-brrr-api",
        "version": "0.1.0",
    }


@router.get("/ready", name="readiness_check")
async def readiness_check() -> dict[str, str]:
    """Readiness check for Kubernetes."""
    # TODO: Check dependencies (Redis, model loading, etc.)
    return {
        "status": "ready",
        "timestamp": utc_now().isoformat(),
    }
