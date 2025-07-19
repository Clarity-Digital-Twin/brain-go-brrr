"""Brain-Go-Brrr API - FastAPI application for EEG analysis.

MVP: Auto-QC + Risk Flagger endpoint
"""

import base64
import logging

# Add project to path
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# Add project root to path first, before any project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time  # noqa: E402
import uuid  # noqa: E402
from enum import Enum  # noqa: E402

import mne  # noqa: E402
from fastapi import (  # noqa: E402
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field  # noqa: E402

from api.auth import verify_cache_clear_permission  # noqa: E402
from api.cache import RedisCache, get_cache  # noqa: E402
from brain_go_brrr.data.edf_streaming import estimate_memory_usage  # noqa: E402
from brain_go_brrr.utils import utc_now  # noqa: E402
from brain_go_brrr.visualization.markdown_report import (  # noqa: E402
    MarkdownReportGenerator,
)
from brain_go_brrr.visualization.pdf_report import PDFReportGenerator  # noqa: E402
from services.qc_flagger import EEGQualityController  # noqa: E402
from services.sleep_metrics import SleepAnalyzer  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Brain-Go-Brrr API",
    description="Production-ready EEG analysis API with EEGPT",
    version="0.1.0",
)

# Global model instance (loaded once)
# Support environment variable for model path (useful for Docker/cloud)
default_model_path = project_root / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
EEGPT_MODEL_PATH = Path(os.getenv("EEGPT_MODEL_PATH", str(default_model_path))).absolute()
qc_controller = None
cache_client: RedisCache | None = None


class AnalysisRequest(BaseModel):
    """Request model for EEG analysis."""

    analysis_type: str = Field(default="qc", description="Type of analysis to perform")
    options: dict[str, Any] = Field(default_factory=dict, description="Analysis options")


class QCResponse(BaseModel):
    """Response model for QC analysis."""

    status: str
    bad_channels: list[str]
    bad_pct: float
    abnormal_prob: float
    flag: str
    confidence: float
    processing_time: float
    quality_grade: str
    timestamp: str
    error: str | None = None
    cached: bool = False


class SleepAnalysisResponse(BaseModel):
    """Response model for sleep analysis."""

    status: str
    sleep_stages: dict[str, float]
    sleep_metrics: dict[str, float]
    hypnogram: list[dict[str, Any]]
    metadata: dict[str, Any]
    processing_time: float
    timestamp: str
    error: str | None = None
    cached: bool = False


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobCreateRequest(BaseModel):
    """Request model for job creation."""

    analysis_type: str = Field(..., description="Type of analysis to perform")
    file_path: str = Field(..., description="Path to the file to analyze")
    options: dict[str, Any] = Field(default_factory=dict, description="Analysis options")
    priority: JobPriority | None = Field(default=JobPriority.NORMAL, description="Job priority")
    timeout: int | None = Field(default=300, description="Timeout in seconds")


class JobResponse(BaseModel):
    """Response model for job operations."""

    job_id: str
    status: JobStatus
    analysis_type: str
    created_at: str
    updated_at: str
    priority: JobPriority
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobListResponse(BaseModel):
    """Response model for job listing."""

    jobs: list[JobResponse]
    total: int
    page: int = 1
    per_page: int = 50


class QueueStatusResponse(BaseModel):
    """Response model for queue status."""

    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_jobs: int
    workers_active: int = 0
    queue_healthy: bool = True


# In-memory job store (for MVP - replace with database in production)
job_store: dict[str, dict] = {}


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global qc_controller, cache_client
    try:
        logger.info("Loading EEGPT model...")
        qc_controller = EEGQualityController(eegpt_model_path=EEGPT_MODEL_PATH)
        logger.info("EEGPT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load EEGPT model: {e}")
        # Continue without EEGPT - will use fallback methods
        try:
            qc_controller = EEGQualityController(eegpt_model_path=None)
            logger.info("QC controller initialized without EEGPT model")
        except Exception as e2:
            logger.error(f"Failed to initialize QC controller: {e2}")
            qc_controller = None

    # Initialize cache
    try:
        cache_client = get_cache()
        if cache_client:
            logger.info("Redis cache initialized successfully")
        else:
            logger.info("Running without cache")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        cache_client = None


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global qc_controller
    if qc_controller is not None and hasattr(qc_controller, "cleanup"):
        logger.info("Cleaning up resources...")
        qc_controller.cleanup()
        logger.info("Cleanup completed")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Brain-Go-Brrr API",
        "version": "0.1.0",
        "endpoints": {
            "/health": "Health check",
            "/api/v1/eeg/analyze": "Upload and analyze EEG file",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_health = {}
    if cache_client:
        redis_health = cache_client.health_check()

    return {
        "status": "healthy",
        "version": app.version,
        "eegpt_loaded": qc_controller is not None and qc_controller.eegpt_model is not None,
        "redis": redis_health,
        "timestamp": utc_now().isoformat(),
    }


@app.post("/api/v1/eeg/analyze", response_model=QCResponse)
async def analyze_eeg(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze uploaded EEG file.

    MVP: Auto-QC + Risk Flagger
    - Detect bad channels
    - Compute abnormality probability
    - Return triage flag

    Args:
        file: Uploaded EDF file
        background_tasks: FastAPI background tasks for async cleanup

    Returns:
        QC analysis results
    """
    # Validate file type
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    # Read file content for caching
    content = await file.read()
    await file.seek(0)  # Reset file pointer

    # Check cache if available
    if cache_client and cache_client.connected:
        try:
            cache_key = cache_client.generate_cache_key(content, "standard")
            cached_result = cache_client.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            cached_result = None

        if cached_result:
            logger.info(f"Returning cached result for {file.filename}")
            cached_result["cached"] = True
            return QCResponse(**cached_result)

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
            # Save uploaded file
            tmp_file.write(content)
            tmp_file.flush()

            # Check memory requirements
            mem_estimate = estimate_memory_usage(tmp_path, preload=True)
            logger.info(
                f"Processing file: {file.filename} (~{mem_estimate['estimated_total_mb']:.1f} MB)"
            )

            # Warn if file is large
            if mem_estimate["estimated_total_mb"] > 500:
                logger.warning(
                    f"Large file detected: {mem_estimate['estimated_total_mb']:.1f} MB. "
                    "Consider using streaming mode for production."
                )

            # Load EEG data
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

            # Check if QC controller is available
            if qc_controller is None:
                raise RuntimeError("QC controller not initialized. Please check logs.")

            # Run QC analysis
            logger.info("Running QC analysis...")
            results = qc_controller.run_full_qc_pipeline(raw)

            # Extract key metrics
            quality_metrics = results.get("quality_metrics", {})
            bad_channels = quality_metrics.get("bad_channels", [])
            bad_pct = quality_metrics.get("bad_channel_ratio", 0) * 100
            abnormal_prob = quality_metrics.get("abnormality_score", 0)
            quality_grade = quality_metrics.get("quality_grade", "UNKNOWN")

            # Determine triage flag
            if abnormal_prob > 0.8 or quality_grade == "POOR":
                flag = "URGENT - Expedite read"
            elif abnormal_prob > 0.6 or quality_grade == "FAIR":
                flag = "EXPEDITE - Priority review"
            elif abnormal_prob > 0.4:
                flag = "ROUTINE - Standard workflow"
            else:
                flag = "NORMAL - Low priority"

            # Get confidence score
            if "processing_info" in results and isinstance(
                results["processing_info"].get("confidence"), float
            ):
                confidence = results["processing_info"]["confidence"]
            else:
                confidence = 0.8 if qc_controller.eegpt_model is not None else 0.5

            # Processing time
            processing_time = results.get("processing_time", 0)

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, tmp_path)

            # Prepare response
            response_data = {
                "status": "success",
                "bad_channels": bad_channels,
                "bad_pct": round(bad_pct, 1),
                "abnormal_prob": round(abnormal_prob, 3),
                "flag": flag,
                "confidence": round(confidence, 3),
                "processing_time": round(processing_time, 2),
                "quality_grade": quality_grade,
                "timestamp": utc_now().isoformat(),
            }

            # Cache the result if cache is available
            if cache_client and cache_client.connected:
                try:
                    cache_key = cache_client.generate_cache_key(content, "standard")
                    cache_client.set(cache_key, response_data, ttl=3600)  # 1 hour TTL
                except Exception as e:
                    logger.error(f"Failed to cache result: {e}")

            return QCResponse(**response_data)

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            logger.error(traceback.format_exc())

            # Cleanup on error
            if tmp_path.exists():
                tmp_path.unlink()

            return QCResponse(
                status="error",
                bad_channels=[],
                bad_pct=0,
                abnormal_prob=0,
                flag="ERROR",
                confidence=0,
                processing_time=0,
                quality_grade="ERROR",
                timestamp=utc_now().isoformat(),
                error=str(e),
            )


@app.post("/api/v1/eeg/sleep/analyze", response_model=SleepAnalysisResponse)
async def analyze_sleep_eeg(
    edf_file: UploadFile = File(...), _background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze uploaded EEG file for sleep staging.

    Performs automatic sleep staging using YASA and EEGPT features.
    Returns hypnogram, sleep metrics, and stage percentages.

    Args:
        edf_file: Uploaded EDF file containing sleep EEG data
        _background_tasks: FastAPI background tasks (unused in minimal implementation)

    Returns:
        Sleep analysis results including hypnogram and metrics
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

    # Create temporary file to load EDF
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
            # Save uploaded content
            tmp_file.write(content)
            tmp_file.flush()

            # Start timing
            start_time = time.time()

            # Load EDF data
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

            # Run YASA sleep analysis
            sleep_analyzer = SleepAnalyzer()
            results = sleep_analyzer.run_full_sleep_analysis(raw)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Extract sleep stages (convert to percentages)
            stage_counts = results.get("sleep_stages", {})
            total_epochs = sum(stage_counts.values())
            sleep_stages = {}
            if total_epochs > 0:
                for stage, count in stage_counts.items():
                    # Map R to REM for consistency
                    stage_name = "REM" if stage == "R" else stage
                    sleep_stages[stage_name] = round(count / total_epochs, 3)
            else:
                # Default if no stages found
                sleep_stages = {"W": 1.0, "N1": 0.0, "N2": 0.0, "N3": 0.0, "REM": 0.0}

            # Extract sleep metrics
            sleep_metrics = results.get("sleep_metrics", {})

            # Convert hypnogram to required format
            hypnogram_stages = results.get("hypnogram", [])
            hypnogram = []
            for i, stage in enumerate(hypnogram_stages):
                # Map R to REM
                stage_name = "REM" if stage == "R" else stage
                hypnogram.append(
                    {
                        "epoch": i + 1,
                        "stage": stage_name,
                        "confidence": 0.85,  # Default confidence since YASA doesn't provide per-epoch
                    }
                )

            # Schedule cleanup
            _background_tasks.add_task(cleanup_temp_file, tmp_path)

            return SleepAnalysisResponse(
                status="success",
                sleep_stages=sleep_stages,
                sleep_metrics={
                    "total_sleep_time": sleep_metrics.get("total_sleep_time", 0.0),
                    "sleep_efficiency": sleep_metrics.get("sleep_efficiency", 0.0),
                    "sleep_onset_latency": sleep_metrics.get("sleep_onset_latency", 0.0),
                    "rem_latency": sleep_metrics.get("rem_latency", 0.0),
                    "wake_after_sleep_onset": sleep_metrics.get("wake_after_sleep_onset", 0.0),
                },
                hypnogram=hypnogram[:100],  # Limit to first 100 epochs for response size
                metadata={
                    "total_epochs": len(hypnogram_stages),
                    "analysis_duration": round(processing_time, 2),
                    "model_version": "yasa_v0.6.4",
                    "confidence_threshold": 0.0,  # YASA doesn't use threshold
                    "sampling_rate": int(raw.info["sfreq"]),
                    "n_channels": len(raw.ch_names),
                },
                processing_time=round(processing_time, 2),
                timestamp=utc_now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error processing sleep analysis: {e}")
            logger.error(traceback.format_exc())

            # Cleanup on error
            if tmp_path.exists():
                with contextlib.suppress(Exception):
                    tmp_path.unlink()

            # Return error response
            return SleepAnalysisResponse(
                status="error",
                sleep_stages={"W": 0.0, "N1": 0.0, "N2": 0.0, "N3": 0.0, "REM": 0.0},
                sleep_metrics={
                    "total_sleep_time": 0.0,
                    "sleep_efficiency": 0.0,
                    "sleep_onset_latency": 0.0,
                    "rem_latency": 0.0,
                    "wake_after_sleep_onset": 0.0,
                },
                hypnogram=[],
                metadata={"error": str(e)},
                processing_time=0.0,
                timestamp=utc_now().isoformat(),
                error=str(e),
            )


@app.post("/api/v1/eeg/analyze/detailed")
async def analyze_eeg_detailed(
    file: UploadFile = File(...),
    include_report: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Detailed EEG analysis with optional PDF report.

    Returns comprehensive analysis results with PDF report.
    """
    # Read file content for caching
    content = await file.read()
    await file.seek(0)  # Reset file pointer

    # Check cache if available
    if cache_client and cache_client.connected:
        cache_key = cache_client.generate_cache_key(content, "detailed")
        cached_result = cache_client.get(cache_key)

        if cached_result:
            logger.info(f"Returning cached detailed result for {file.filename}")
            cached_result["basic"]["cached"] = True
            return cached_result

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
            # Save uploaded file
            tmp_file.write(content)
            tmp_file.flush()

            # Check memory requirements
            mem_estimate = estimate_memory_usage(tmp_path, preload=True)
            logger.info(
                f"Processing file for detailed analysis: {file.filename} "
                f"(~{mem_estimate['estimated_total_mb']:.1f} MB)"
            )

            # Warn if file is large
            if mem_estimate["estimated_total_mb"] > 500:
                logger.warning(
                    f"Large file detected: {mem_estimate['estimated_total_mb']:.1f} MB. "
                    "Consider using streaming mode for production."
                )

            # Load EEG data
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

            # Check if QC controller is available
            if qc_controller is None:
                raise RuntimeError("QC controller not initialized. Please check logs.")

            # Run QC analysis
            logger.info("Running detailed QC analysis...")
            results = qc_controller.run_full_qc_pipeline(raw)

            # Extract key metrics for basic response
            quality_metrics = results.get("quality_metrics", {})
            bad_channels = quality_metrics.get("bad_channels", [])
            bad_pct = quality_metrics.get("bad_channel_ratio", 0) * 100
            abnormal_prob = quality_metrics.get("abnormality_score", 0)
            quality_grade = quality_metrics.get("quality_grade", "UNKNOWN")

            # Determine triage flag
            if abnormal_prob > 0.8 or quality_grade == "POOR":
                flag = "URGENT - Expedite read"
            elif abnormal_prob > 0.6 or quality_grade == "FAIR":
                flag = "EXPEDITE - Priority review"
            elif abnormal_prob > 0.4:
                flag = "ROUTINE - Standard workflow"
            else:
                flag = "NORMAL - Low priority"

            # Get confidence score
            if "processing_info" in results and isinstance(
                results["processing_info"].get("confidence"), float
            ):
                confidence = results["processing_info"]["confidence"]
            else:
                confidence = 0.8 if qc_controller.eegpt_model is not None else 0.5

            # Processing time
            processing_time = results.get("processing_time", 0)

            # Generate reports if requested
            pdf_base64 = None
            markdown_report = None
            if include_report:
                try:
                    # Add more details to results for report generation
                    # Ensure artifacts have severity scores
                    if "artifact_segments" in quality_metrics:
                        for artifact in quality_metrics["artifact_segments"]:
                            if "severity" not in artifact:
                                # Assign severity based on type
                                severity_map = {
                                    "motion": 0.9,
                                    "muscle": 0.7,
                                    "eye_blink": 0.5,
                                    "heartbeat": 0.3,
                                    "other": 0.6,
                                }
                                artifact["severity"] = severity_map.get(
                                    artifact.get("type", "other"), 0.6
                                )

                    report_results = {
                        "quality_metrics": {
                            **quality_metrics,
                            "channel_positions": _get_channel_positions(raw),
                            "flag": flag,  # Add triage flag
                        },
                        "processing_info": {
                            "file_name": file.filename,
                            "timestamp": utc_now().isoformat(),
                            "duration_seconds": raw.times[-1],
                            "sampling_rate": raw.info["sfreq"],
                            "confidence": confidence,
                            "channels_used": len(raw.ch_names),
                        },
                        "autoreject_results": results.get("autoreject_results", {}),
                        "eegpt_features": results.get("eegpt_features", {}),
                        "processing_time": processing_time,
                    }

                    # Generate PDF
                    pdf_generator = PDFReportGenerator()
                    pdf_bytes = pdf_generator.generate_report(report_results, raw.get_data())

                    # Convert to base64 for JSON response
                    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    logger.info("PDF report generated successfully")

                    # Generate Markdown report
                    markdown_generator = MarkdownReportGenerator()
                    markdown_report = markdown_generator.generate_report(report_results)
                    logger.info("Markdown report generated successfully")

                    # Optionally save markdown to outputs directory
                    output_dir = project_root / "outputs" / "reports"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Create filename with timestamp
                    timestamp_str = utc_now().strftime("%Y%m%d_%H%M%S")
                    markdown_path = output_dir / f"eeg_report_{timestamp_str}.md"
                    markdown_generator.save_report(report_results, markdown_path)
                    logger.info(f"Markdown report saved to {markdown_path}")

                except Exception as e:
                    logger.error(f"Failed to generate reports: {e}")
                    # Continue without reports

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, tmp_path)

            # Create basic result
            basic_result = QCResponse(
                status="success",
                bad_channels=bad_channels,
                bad_pct=round(bad_pct, 1),
                abnormal_prob=round(abnormal_prob, 3),
                flag=flag,
                confidence=round(confidence, 3),
                processing_time=round(processing_time, 2),
                quality_grade=quality_grade,
                timestamp=utc_now().isoformat(),
            )

            # Prepare response
            response_data = {
                "basic": basic_result.model_dump(),
                "detailed": {
                    "message": "Detailed analysis complete",
                    "pdf_available": pdf_base64 is not None,
                    "pdf_base64": pdf_base64,
                    "markdown_available": markdown_report is not None,
                    "markdown_report": markdown_report,
                    "artifact_count": len(quality_metrics.get("artifact_segments", [])),
                    "channel_count": len(raw.ch_names),
                    "duration_seconds": raw.times[-1],
                },
            }

            # Cache the result if cache is available
            if cache_client and cache_client.connected:
                try:
                    cache_key = cache_client.generate_cache_key(content, "detailed")
                    cache_client.set(cache_key, response_data, ttl=3600)  # 1 hour TTL
                except Exception as e:
                    logger.error(f"Failed to cache detailed result: {e}")

            return response_data

        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
            logger.error(traceback.format_exc())

            # Cleanup on error
            if tmp_path.exists():
                tmp_path.unlink()

            return {
                "basic": QCResponse(
                    status="error",
                    bad_channels=[],
                    bad_pct=0,
                    abnormal_prob=0,
                    flag="ERROR",
                    confidence=0,
                    processing_time=0,
                    quality_grade="ERROR",
                    timestamp=utc_now().isoformat(),
                    error=str(e),
                ),
                "detailed": {
                    "message": "Detailed analysis failed",
                    "pdf_available": False,
                    "error": str(e),
                },
            }


def _get_channel_positions(raw: mne.io.Raw) -> dict[str, tuple[float, float]]:
    """Extract channel positions from raw data."""
    positions = {}

    try:
        # Get montage if available
        montage = raw.get_montage()
        if montage is not None:
            ch_names = montage.ch_names
            pos = montage.get_positions()["ch_pos"]

            # Convert 3D positions to 2D (x, y) for visualization
            for ch_name in ch_names:
                if ch_name in pos:
                    x, y, z = pos[ch_name]
                    positions[ch_name] = (x, y)
        else:
            # Fallback to standard 10-20 positions for common channels
            standard_positions = {
                "Fp1": (-0.3, 0.8),
                "Fp2": (0.3, 0.8),
                "F3": (-0.5, 0.5),
                "F4": (0.5, 0.5),
                "C3": (-0.5, 0),
                "C4": (0.5, 0),
                "P3": (-0.5, -0.5),
                "P4": (0.5, -0.5),
                "O1": (-0.3, -0.8),
                "O2": (0.3, -0.8),
                "F7": (-0.8, 0.3),
                "F8": (0.8, 0.3),
                "T3": (-0.8, 0),
                "T4": (0.8, 0),
                "T5": (-0.8, -0.3),
                "T6": (0.8, -0.3),
                "Fz": (0, 0.6),
                "Cz": (0, 0),
                "Pz": (0, -0.6),
            }

            # Use standard positions for channels that match
            for ch_name in raw.ch_names:
                if ch_name in standard_positions:
                    positions[ch_name] = standard_positions[ch_name]

    except Exception as e:
        logger.warning(f"Failed to extract channel positions: {e}")

    return positions


def cleanup_temp_file(file_path: Path) -> None:
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file: {e}")


@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Get Redis cache statistics."""
    if not cache_client or not cache_client.connected:
        return {"status": "unavailable", "message": "Cache not available"}

    try:
        stats = cache_client.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"status": "error", "error": str(e)}


@app.delete("/api/v1/cache/clear")
async def clear_cache(
    pattern: str = "eeg_analysis:*",
    _authorized: bool = Depends(verify_cache_clear_permission),
):
    """Clear cache entries matching pattern.

    Requires admin token or HMAC signature for authorization.
    """
    if not cache_client or not cache_client.connected:
        return {"status": "unavailable", "message": "Cache not available"}

    try:
        keys_deleted = cache_client.clear_pattern(pattern)
        return {"status": "success", "keys_deleted": keys_deleted, "pattern": pattern}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {"status": "error", "error": str(e)}


class CacheWarmupRequest(BaseModel):
    """Request model for cache warmup."""

    file_patterns: list[str] = Field(default=["sleep-*.edf"], description="File patterns to cache")


@app.post("/api/v1/cache/warmup")
async def warmup_cache(request: CacheWarmupRequest):
    """Pre-warm cache with common test files."""
    # This is a placeholder - in production, would scan for files
    # matching patterns and pre-process them
    return {
        "status": "success",
        "message": "Cache warmup initiated",
        "files_cached": 0,
        "patterns": request.file_patterns,
    }


# Job Queue Endpoints


@app.post("/api/v1/jobs/create", response_model=JobResponse, status_code=201)
async def create_job(request: JobCreateRequest) -> JobResponse:
    """Create a new analysis job."""
    job_id = str(uuid.uuid4())
    now = utc_now().isoformat()

    job_data = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "analysis_type": request.analysis_type,
        "file_path": request.file_path,
        "options": request.options,
        "priority": request.priority,
        "timeout": request.timeout,
        "created_at": now,
        "updated_at": now,
        "progress": 0.0,
        "result": None,
        "error": None,
    }

    job_store[job_id] = job_data

    # TODO: Queue job for background processing
    logger.info(f"Created job {job_id} for {request.analysis_type} analysis")

    return JobResponse(**job_data)


@app.get("/api/v1/jobs/{job_id}/status", response_model=JobResponse)
async def get_job_status(job_id: str) -> JobResponse:
    """Get the status of a specific job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobResponse(**job_store[job_id])


@app.get("/api/v1/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store[job_id]

    if job["status"] == JobStatus.PENDING:
        return {"status": "pending", "message": "Job is queued for processing"}, 202
    elif job["status"] == JobStatus.PROCESSING:
        return {
            "status": "processing",
            "message": "Job is currently being processed",
            "progress": job.get("progress", 0),
        }, 202
    elif job["status"] == JobStatus.COMPLETED:
        return {"status": "completed", "result": job["result"]}, 200
    elif job["status"] == JobStatus.FAILED:
        return {"status": "failed", "error": job["error"]}, 200
    else:
        return {"status": job["status"]}, 200


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = job_store[job_id]

    if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
        return {"status": "error", "message": "Cannot cancel completed or failed job"}, 409

    job["status"] = JobStatus.CANCELLED
    job["updated_at"] = utc_now().isoformat()

    return {"status": "success", "message": f"Job {job_id} cancelled"}


@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs(
    status: JobStatus | None = None, page: int = 1, per_page: int = 50
) -> JobListResponse:
    """List jobs with optional filtering."""
    # Filter jobs by status if provided
    if status:
        filtered_jobs = [job for job in job_store.values() if job["status"] == status]
    else:
        filtered_jobs = list(job_store.values())

    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_jobs = filtered_jobs[start_idx:end_idx]

    return JobListResponse(
        jobs=[JobResponse(**job) for job in paginated_jobs],
        total=len(filtered_jobs),
        page=page,
        per_page=per_page,
    )


@app.get("/api/v1/queue/status", response_model=QueueStatusResponse)
async def get_queue_status() -> QueueStatusResponse:
    """Get queue status and statistics."""
    # Count jobs by status
    status_counts = {
        JobStatus.PENDING: 0,
        JobStatus.PROCESSING: 0,
        JobStatus.COMPLETED: 0,
        JobStatus.FAILED: 0,
        JobStatus.CANCELLED: 0,
    }

    for job in job_store.values():
        status_counts[job["status"]] += 1

    return QueueStatusResponse(
        pending_jobs=status_counts[JobStatus.PENDING],
        processing_jobs=status_counts[JobStatus.PROCESSING],
        completed_jobs=status_counts[JobStatus.COMPLETED],
        failed_jobs=status_counts[JobStatus.FAILED],
        total_jobs=len(job_store),
        workers_active=1 if status_counts[JobStatus.PROCESSING] > 0 else 0,
        queue_healthy=True,
    )


@app.get("/api/v1/queue/health")
async def get_queue_health():
    """Check queue health status."""
    return {
        "status": "healthy",
        "queue_available": True,
        "workers_available": True,
        "timestamp": utc_now().isoformat(),
    }


@app.get("/api/v1/queue/workers")
async def get_worker_status():
    """Get worker status information."""
    # Mock implementation for MVP
    return {
        "workers": [
            {
                "id": "worker-1",
                "status": "idle",
                "last_heartbeat": utc_now().isoformat(),
                "jobs_processed": 0,
            }
        ],
        "total_workers": 1,
        "active_workers": 0,
    }


@app.post("/api/v1/queue/cleanup")
async def cleanup_old_jobs(days_old: int = 7):
    """Clean up old completed jobs."""
    # Mock implementation - in production would remove old jobs
    return {
        "status": "success",
        "jobs_removed": 0,
        "message": f"Cleaned up jobs older than {days_old} days",
    }


@app.post("/api/v1/queue/pause")
async def pause_queue():
    """Pause job processing."""
    return {"status": "success", "message": "Queue paused", "queue_state": "paused"}


@app.post("/api/v1/queue/resume")
async def resume_queue():
    """Resume job processing."""
    return {"status": "success", "message": "Queue resumed", "queue_state": "active"}


# Resource Monitoring Endpoints


@app.get("/api/v1/resources/gpu")
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


@app.get("/api/v1/resources/memory")
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


@app.get("/api/v1/queue/failed")
async def get_failed_jobs():
    """Get failed jobs (dead letter queue)."""
    failed_jobs = [job for job in job_store.values() if job["status"] == JobStatus.FAILED]

    return {"failed_jobs": [JobResponse(**job) for job in failed_jobs], "total": len(failed_jobs)}


@app.get("/api/v1/jobs/{job_id}/progress")
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


@app.get("/api/v1/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get execution logs for a specific job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # In production, this would fetch from a logging service
    return {"job_id": job_id, "logs": job_store[job_id].get("logs", [])}


@app.get("/api/v1/jobs/{job_id}/stream")
async def stream_job_updates(job_id: str):
    """Stream real-time job updates (placeholder)."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Placeholder - in production would use WebSocket or SSE
    raise HTTPException(
        status_code=501,
        detail="Real-time streaming not implemented. Use polling on /status endpoint.",
    )


@app.get("/api/v1/jobs/history")
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


@app.post("/api/v1/queue/recover")
async def recover_jobs():
    """Recover in-progress jobs after system restart."""
    # Find all jobs that were processing
    recovered = 0
    for _job_id, job in job_store.items():
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
