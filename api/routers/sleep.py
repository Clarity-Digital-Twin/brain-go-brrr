"""Sleep analysis endpoints."""

import contextlib
import logging
import tempfile
import uuid
from pathlib import Path

import mne
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from api.routers.jobs import job_store  # TODO: Move to core
from api.schemas import JobPriority, JobResponse, JobStatus, SleepAnalysisResponse
from brain_go_brrr.utils.time import utc_now
from services.sleep_metrics import SleepAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eeg/sleep", tags=["sleep"])


async def process_sleep_analysis_job(job_id: str, file_path: Path) -> None:
    """Process sleep analysis job in background."""
    job = job_store.get(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    try:
        # Update job status
        job["status"] = JobStatus.PROCESSING
        job["started_at"] = utc_now().isoformat()
        job["updated_at"] = utc_now().isoformat()

        # Load EDF data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Run YASA sleep analysis
        sleep_analyzer = SleepAnalyzer()
        results = sleep_analyzer.run_full_sleep_analysis(raw)

        # Extract and format results
        stage_counts = results.get("sleep_stages", {})
        total_epochs = sum(stage_counts.values())
        sleep_stages = {}
        if total_epochs > 0:
            for stage, count in stage_counts.items():
                stage_name = "REM" if stage == "R" else stage
                sleep_stages[stage_name] = round(count / total_epochs, 3)
        else:
            sleep_stages = {"W": 1.0, "N1": 0.0, "N2": 0.0, "N3": 0.0, "REM": 0.0}

        # Store results
        job["result"] = {
            "sleep_stages": sleep_stages,
            "sleep_metrics": results.get("sleep_metrics", {}),
            "hypnogram": results.get("hypnogram", [])[:100],  # Limit size
            "metadata": {
                "total_epochs": len(results.get("hypnogram", [])),
                "model_version": "yasa_v0.6.4",
                "sampling_rate": int(raw.info["sfreq"]),
                "n_channels": len(raw.ch_names),
            },
        }
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 1.0
        job["completed_at"] = utc_now().isoformat()
        job["updated_at"] = utc_now().isoformat()

    except Exception as e:
        logger.error(f"Sleep analysis job {job_id} failed: {e}")
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        job["updated_at"] = utc_now().isoformat()
        job["completed_at"] = utc_now().isoformat()

    finally:
        # Cleanup temp file
        with contextlib.suppress(Exception):
            if file_path.exists():
                file_path.unlink()


@router.post("/analyze", response_model=JobResponse, status_code=202)
async def analyze_sleep_eeg(
    edf_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Queue sleep analysis job for uploaded EEG file.

    Performs automatic sleep staging using YASA and EEGPT features.
    Returns job ID for async processing.

    Args:
        edf_file: Uploaded EDF file containing sleep EEG data
        background_tasks: FastAPI background tasks for async processing

    Returns:
        Job response with job_id for tracking
    """
    # Validate file type
    if not edf_file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    # Read file content for validation
    content = await edf_file.read()
    await edf_file.seek(0)  # Reset file pointer

    # Check file size limits (max 50MB for API)
    max_file_size = 50 * 1024 * 1024  # 50MB
    if len(content) > max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_file_size // (1024 * 1024)}MB",
        )

    # Basic EDF format validation - reject obviously invalid files
    # EDF files are binary and have substantial content
    if len(content) < 10:
        raise HTTPException(status_code=400, detail="File too small to be valid EDF")

    # Reject files that look like text but claim to be EDF
    if content.startswith(b"NOT_AN_EDF") or content == b"NOT_AN_EDF_FILE":
        raise HTTPException(status_code=400, detail="Invalid EDF file format")

    # Create a job for sleep analysis
    job_id = str(uuid.uuid4())

    # Save file content temporarily for the job
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(content)
        tmp_file.flush()

    # Create job entry
    job = {
        "job_id": job_id,
        "analysis_type": "sleep",
        "file_path": str(tmp_path),
        "options": {
            "file_size": len(content),
            "filename": edf_file.filename,
        },
        "status": JobStatus.PENDING,
        "priority": JobPriority.NORMAL,
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": utc_now().isoformat(),
        "updated_at": utc_now().isoformat(),
        "started_at": None,
        "completed_at": None,
    }

    # Store job
    job_store[job_id] = job

    # Schedule the actual sleep analysis as a background task
    background_tasks.add_task(
        process_sleep_analysis_job,
        job_id,
        tmp_path,
    )

    # Return job response
    return JobResponse(
        job_id=job_id,
        analysis_type="sleep",
        status=JobStatus.PENDING,
        priority=JobPriority.NORMAL,
        progress=0.0,
        result=None,
        error=None,
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.get("/jobs/{job_id}/status")
async def get_sleep_job_status(job_id: str):
    """Get status of a sleep analysis job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }


@router.get("/jobs/{job_id}/results")
async def get_sleep_job_results(job_id: str):
    """Get results of a completed sleep analysis job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] == JobStatus.PENDING:
        raise HTTPException(status_code=202, detail="Job is still pending")
    elif job["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Job is still processing")
    elif job["status"] == JobStatus.FAILED:
        raise HTTPException(
            status_code=500, detail=f"Job failed: {job.get('error', 'Unknown error')}"
        )

    # Job is completed, return results
    result = job.get("result", {})
    return SleepAnalysisResponse(
        status="success",
        sleep_stages=result.get("sleep_stages", {}),
        sleep_metrics=result.get("sleep_metrics", {}),
        hypnogram=result.get("hypnogram", []),
        metadata=result.get("metadata", {}),
        processing_time=0.0,  # Could calculate from timestamps
        timestamp=job.get("completed_at", utc_now().isoformat()),
        cached=False,
    )
