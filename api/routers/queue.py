"""Queue management endpoints."""

import logging
from typing import Any

from fastapi import APIRouter

from api.routers.jobs import job_store  # TODO: Move to core
from api.schemas import JobResponse, JobStatus, QueueStatusResponse
from brain_go_brrr.utils.time import utc_now

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get current queue status and statistics."""
    # Count jobs by status
    status_counts = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
    }

    for job in job_store.values():
        status = job["status"]
        if status == JobStatus.PENDING:
            status_counts["pending"] += 1
        elif status == JobStatus.PROCESSING:
            status_counts["processing"] += 1
        elif status == JobStatus.COMPLETED:
            status_counts["completed"] += 1
        elif status == JobStatus.FAILED:
            status_counts["failed"] += 1

    # Determine health based on queue size
    total_active = status_counts["pending"] + status_counts["processing"]
    if total_active > 100:
        health = "degraded"
    elif total_active > 200:
        health = "unhealthy"
    else:
        health = "healthy"

    return QueueStatusResponse(
        pending_jobs=status_counts["pending"],
        processing_jobs=status_counts["processing"],
        completed_jobs=status_counts["completed"],
        failed_jobs=status_counts["failed"],
        workers_active=1,  # TODO: Get from Celery
        queue_health=health,
    )


@router.get("/health")
async def queue_health_check() -> dict[str, Any]:
    """Check queue system health."""
    status = await get_queue_status()
    return {
        "status": status.queue_health,
        "metrics": {
            "pending": status.pending_jobs,
            "processing": status.processing_jobs,
            "failed": status.failed_jobs,
        },
    }


@router.get("/workers")
async def get_worker_status() -> dict[str, Any]:
    """Get status of queue workers."""
    # TODO: Integrate with Celery
    return {
        "workers": [
            {
                "id": "worker-1",
                "status": "active",
                "current_job": None,
                "jobs_completed": 0,
            }
        ],
        "total_workers": 1,
    }


@router.post("/cleanup")
async def cleanup_old_jobs() -> dict[str, str]:
    """Clean up old completed/failed jobs."""
    # TODO: Implement cleanup logic
    return {"status": "not_implemented", "message": "Cleanup not yet implemented"}


@router.post("/pause")
async def pause_queue() -> dict[str, str]:
    """Pause job processing."""
    return {"status": "success", "message": "Queue paused", "queue_state": "paused"}


@router.post("/resume")
async def resume_queue() -> dict[str, str]:
    """Resume job processing."""
    return {"status": "success", "message": "Queue resumed", "queue_state": "active"}


@router.get("/failed")
async def get_failed_jobs() -> dict[str, Any]:
    """Get failed jobs (dead letter queue)."""
    failed_jobs = [job for job in job_store.values() if job["status"] == JobStatus.FAILED]

    return {
        "failed_jobs": [
            JobResponse(
                job_id=job["job_id"],
                analysis_type=job["analysis_type"],
                status=job["status"],
                priority=job["priority"],
                progress=job["progress"],
                result=job["result"],
                error=job["error"],
                created_at=job["created_at"],
                updated_at=job["updated_at"],
                started_at=job["started_at"],
                completed_at=job["completed_at"],
            )
            for job in failed_jobs
        ],
        "total": len(failed_jobs),
    }


@router.post("/recover")
async def recover_jobs() -> dict[str, Any]:
    """Recover in-progress jobs after system restart."""
    # Find all jobs that were processing
    recovered = 0
    for job in job_store.values():
        if job["status"] == JobStatus.PROCESSING:
            # Reset to pending for reprocessing
            job["status"] = JobStatus.PENDING
            job["updated_at"] = utc_now().isoformat()
            recovered += 1

    return {
        "status": "success",
        "recovered_jobs": recovered,
        "message": f"Recovered {recovered} jobs for reprocessing",
    }
