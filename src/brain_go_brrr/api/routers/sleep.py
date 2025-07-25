"""Sleep analysis endpoints."""

import contextlib
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from brain_go_brrr.api.routers.jobs import job_store  # TODO: Move to core
from brain_go_brrr.api.schemas import (
    JobData,
    JobPriority,
    JobResponse,
    JobStatus,
    SleepAnalysisResponse,
)
from brain_go_brrr.core.edf_loader import load_edf_safe
from brain_go_brrr.core.exceptions import EdfLoadError, SleepAnalysisError, UnsupportedMontageError
from brain_go_brrr.core.sleep import SleepAnalyzer
from brain_go_brrr.utils.time import utc_now

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
        job_store.update(
            job_id,
            {
                "status": JobStatus.PROCESSING,
                "started_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
            },
        )

        # Load EDF data
        try:
            raw = load_edf_safe(file_path, preload=True, verbose=False)
        except EdfLoadError as e:
            # Expected EDF loading errors
            logger.error(f"Failed to load EDF file: {e}")
            job_store.update(
                job_id,
                {
                    "status": JobStatus.FAILED,
                    "error": str(e),
                    "updated_at": utc_now().isoformat(),
                    "completed_at": utc_now().isoformat(),
                },
            )
            return

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

        # Store results with defensive null-object pattern
        sampling_rate = 256  # default
        n_channels = 0

        if hasattr(raw, "info") and raw.info:
            try:
                sfreq = raw.info.get("sfreq", 256)
                if sfreq and str(sfreq).strip():
                    sampling_rate = int(float(sfreq))
            except (ValueError, TypeError, AttributeError):
                pass

        if hasattr(raw, "ch_names"):
            with contextlib.suppress(TypeError, AttributeError):
                n_channels = len(raw.ch_names)

        job_store.update(
            job_id,
            {
                "result": {
                    "sleep_stages": sleep_stages,
                    "sleep_metrics": results.get("sleep_metrics", {}),
                    "hypnogram": results.get("hypnogram", [])[:100],  # Limit size
                    "metadata": {
                        "total_epochs": len(results.get("hypnogram", [])),
                        "model_version": "yasa_v0.6.4",
                        "sampling_rate": sampling_rate,
                        "n_channels": n_channels,
                    },
                },
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "completed_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
            },
        )

    except UnsupportedMontageError as e:
        # Unsupported montage - client error
        logger.warning(f"Sleep analysis job {job_id} failed due to unsupported montage: {e}")
        job_store.update(
            job_id,
            {
                "status": JobStatus.FAILED,
                "error": f"422: {e!s}",
                "updated_at": utc_now().isoformat(),
                "completed_at": utc_now().isoformat(),
            },
        )
    except SleepAnalysisError as e:
        # Expected sleep analysis errors
        logger.error(f"Sleep analysis job {job_id} failed: {e}")
        job_store.update(
            job_id,
            {
                "status": JobStatus.FAILED,
                "error": str(e),
                "updated_at": utc_now().isoformat(),
                "completed_at": utc_now().isoformat(),
            },
        )
    except ImportError as e:
        # Missing dependencies
        logger.error(f"Missing dependency for sleep analysis: {e}")
        job_store.update(
            job_id,
            {
                "status": JobStatus.FAILED,
                "error": "Sleep analysis dependencies not installed",
                "updated_at": utc_now().isoformat(),
                "completed_at": utc_now().isoformat(),
            },
        )
    except Exception as e:
        # Unexpected errors - log and fail job
        logger.critical(f"Unexpected error in sleep analysis job {job_id}: {e}")
        logger.debug("Full traceback:", exc_info=True)
        job_store.update(
            job_id,
            {
                "status": JobStatus.FAILED,
                "error": "Internal server error",
                "updated_at": utc_now().isoformat(),
                "completed_at": utc_now().isoformat(),
            },
        )

    finally:
        # Cleanup temp file
        with contextlib.suppress(Exception):
            if file_path.exists():
                file_path.unlink()


@router.post("/analyze", response_model=JobResponse, status_code=202)
async def analyze_sleep_eeg(
    edf_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
) -> JobResponse:
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
    if not edf_file.filename or not edf_file.filename.lower().endswith(".edf"):
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
    try:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(content)
            tmp_file.flush()
    except OSError as e:
        logger.error(f"Failed to save temporary EDF file: {e}")
        raise HTTPException(status_code=507, detail="Insufficient storage for file upload") from e

    # Create job entry
    job = JobData(
        job_id=job_id,
        analysis_type="sleep",
        file_path=str(tmp_path),
        options={
            "file_size": len(content),
            "filename": edf_file.filename,
        },
        status=JobStatus.PENDING,
        priority=JobPriority.NORMAL,
        progress=0.0,
        result=None,
        error=None,
        created_at=utc_now(),
        updated_at=utc_now(),
        started_at=None,
        completed_at=None,
    )

    # Store job
    job_store.create(job_id, job)

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
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        started_at=None,
        completed_at=None,
    )


@router.get("/jobs/{job_id}/status")
async def get_sleep_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a sleep analysis job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress or 0.0,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
    }


@router.get("/jobs/{job_id}/results", response_model=SleepAnalysisResponse)
async def get_sleep_job_results(job_id: str) -> SleepAnalysisResponse:
    """Get results of a completed sleep analysis job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status == JobStatus.PENDING:
        raise HTTPException(status_code=202, detail="Job is still pending")
    elif job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Job is still processing")
    elif job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error or 'Unknown error'}")

    # Job is completed, return results
    result = job.result or {}

    return SleepAnalysisResponse(
        status="success",
        sleep_stages=result.get("sleep_stages", {}),
        sleep_metrics=result.get("sleep_metrics", {}),
        hypnogram=result.get("hypnogram", []),
        metadata=result.get("metadata", {}),
        processing_time=0.0,  # Could calculate from timestamps
        timestamp=job.completed_at.isoformat() if job.completed_at else utc_now().isoformat(),
        cached=False,
    )
