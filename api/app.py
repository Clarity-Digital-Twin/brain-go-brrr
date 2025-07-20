"""Application factory for Brain-Go-Brrr API."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import cache, health, jobs, qc, queue, resources, sleep
from brain_go_brrr.core.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create FastAPI app
    app = FastAPI(
        title="Brain-Go-Brrr EEG Analysis API",
        description="Production-ready API for EEG analysis using EEGPT and YASA",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
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
    @app.get("/", tags=["root"], name="root")
    async def root(request: Request):
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

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Brain-Go-Brrr API...")
        # TODO: Initialize services (Redis, model loading, etc.)

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Brain-Go-Brrr API...")
        # TODO: Cleanup (close connections, etc.)

    return app
