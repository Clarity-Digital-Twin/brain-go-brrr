"""Job management endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import (
    JobCreateRequest,
    JobListResponse,
    JobResponse,
    JobStatus,
)
from brain_go_brrr.utils.time import utc_now

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# TODO: Move this to core/jobs/store.py
job_store: dict[str, dict] = {}


@router.post("/create", response_model=JobResponse, status_code=201)
async def create_job(request: JobCreateRequest) -> JobResponse:
    """Create a new analysis job."""
    job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "analysis_type": request.analysis_type,
        "file_path": request.file_path,
        "options": request.options,
        "status": JobStatus.PENDING,
        "priority": request.priority,
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": utc_now().isoformat(),
        "updated_at": utc_now().isoformat(),
        "started_at": None,
        "completed_at": None,
    }

    job_store[job_id] = job

    return JobResponse(**job)


@router.get("/{job_id}/status", response_model=JobResponse)
async def get_job_status(job_id: str) -> JobResponse:
    """Get the status of a specific job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobResponse(**job_store[job_id])


@router.get("/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store[job_id]

    if job["status"] != JobStatus.COMPLETED:
        return {
            "status": job["status"],
            "message": "Job not yet completed",
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "result": job.get("result", {}),
        "metadata": {
            "created_at": job["created_at"],
            "completed_at": job["completed_at"],
        },
    }


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store[job_id]

    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(status_code=409, detail="Cannot cancel completed job")

    job["status"] = JobStatus.CANCELLED
    job["updated_at"] = utc_now().isoformat()

    return {"message": f"Job {job_id} cancelled"}


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: JobStatus | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """List all jobs with optional filtering."""
    all_jobs = list(job_store.values())

    # Filter by status if provided
    if status:
        all_jobs = [job for job in all_jobs if job["status"] == status]

    # Sort by creation time (newest first)
    all_jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply pagination
    total = len(all_jobs)
    paginated_jobs = all_jobs[offset : offset + limit]

    return JobListResponse(
        jobs=[JobResponse(**job) for job in paginated_jobs],
        total=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_next=(offset + limit) < total,
    )


@router.get("/{job_id}/progress")
async def get_job_progress(job_id: str):
    """Get detailed progress for a specific job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store[job_id]
    return {
        "job_id": job_id,
        "percent_complete": job.get("progress", 0.0) * 100,
        "current_step": job.get("current_step", "Unknown"),
        "estimated_remaining": job.get("estimated_remaining", None),
        "status": job["status"],
    }


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get execution logs for a specific job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # In production, this would fetch from a logging service
    return {"job_id": job_id, "logs": job_store[job_id].get("logs", [])}


@router.get("/{job_id}/stream")
async def stream_job_updates(job_id: str):
    """Stream real-time job updates (placeholder)."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Placeholder - in production would use WebSocket or SSE
    raise HTTPException(
        status_code=501,
        detail="Real-time streaming not implemented. Use polling on /status endpoint.",
    )


@router.get("/history")
async def get_job_history(limit: int = 100, offset: int = 0):
    """Get historical job data."""
    all_jobs = list(job_store.values())

    # Sort by creation time (newest first)
    all_jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply pagination
    paginated_jobs = all_jobs[offset : offset + limit]

    return {
        "jobs": [JobResponse(**job) for job in paginated_jobs],
        "total": len(all_jobs),
        "limit": limit,
        "offset": offset,
    }
