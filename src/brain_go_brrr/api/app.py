"""Application factory for Brain-Go-Brrr API."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from brain_go_brrr.api.routers import cache, eegpt, health, jobs, qc, queue, resources, sleep
from brain_go_brrr.core.logger import get_logger

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        """Encode numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Brain-Go-Brrr API...")
    # TODO: Initialize services (Redis, model loading, etc.)

    yield

    # Shutdown
    logger.info("Shutting down Brain-Go-Brrr API...")
    # TODO: Cleanup (close connections, etc.)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create FastAPI app with custom JSON encoder
    app = FastAPI(
        title="Brain-Go-Brrr EEG Analysis API",
        description="Production-ready API for EEG analysis using EEGPT and YASA",
        version="0.4.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
        json_encoders={
            np.integer: int,
            np.floating: float,
            np.ndarray: lambda x: x.tolist(),
        },
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root endpoint - defined before routers are included
    @app.get("/", tags=["root"], name="root")  # type: ignore[misc]
    async def root(request: Request) -> dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "message": "Welcome to Brain-Go-Brrr API",
            "version": "0.1.0",
            "endpoints": {
                "docs": str(request.app.docs_url),
                "redoc": str(request.app.redoc_url),
                "health": request.app.url_path_for("health_check"),
                "ready": request.app.url_path_for("readiness_check"),
                "queue_status": request.app.url_path_for("get_queue_status"),
                "jobs": request.app.url_path_for("list_jobs"),
            },
        }

    # Include routers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(jobs.router, prefix="/api/v1")
    app.include_router(queue.router, prefix="/api/v1")
    app.include_router(resources.router, prefix="/api/v1")
    app.include_router(sleep.router, prefix="/api/v1")
    app.include_router(qc.router, prefix="/api/v1")
    app.include_router(cache.router, prefix="/api/v1")
    app.include_router(eegpt.router, prefix="/api/v1")

    # Dual-mount for backward compatibility - mount eegpt router at both locations
    # The router already has /eeg/eegpt prefix, so we need to modify it
    # Create a new router instance with different prefix for backward compatibility
    # Import the router creation logic
    eegpt_compat_router = APIRouter(prefix="/eegpt", tags=["eegpt (deprecated)"])

    # Copy all routes from the original router
    for route in eegpt.router.routes:
        if hasattr(route, "path"):
            # Remove the /eeg/eegpt prefix and add to compat router
            new_path = route.path.replace("/eeg/eegpt", "")
            eegpt_compat_router.add_api_route(
                new_path,
                route.endpoint,
                methods=route.methods,
                name=f"{route.name}_compat" if hasattr(route, "name") else None,
                deprecated=True,
            )

    app.include_router(eegpt_compat_router, prefix="/api/v1")

    # Note: Startup/shutdown logic moved to lifespan context manager above

    return app
