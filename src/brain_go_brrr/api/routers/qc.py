"""Quality control and EEG analysis endpoints."""

import base64
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from brain_go_brrr.api.cache import get_cache
from brain_go_brrr.api.schemas import QCResponse
from brain_go_brrr.core.edf_loader import load_edf_safe
from brain_go_brrr.core.exceptions import EdfLoadError, QualityCheckError
from brain_go_brrr.core.quality import EEGQualityController
from brain_go_brrr.utils.time import utc_now

logger = logging.getLogger(__name__)


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


router = APIRouter(prefix="/eeg", tags=["qc", "analysis"])

# Initialize QC controller
qc_controller = None
try:
    qc_controller = EEGQualityController()
    logger.info("QC controller initialized successfully")
except (ImportError, RuntimeError, AttributeError, ValueError) as e:
    logger.error(f"Failed to initialize QC controller: {e}")
    qc_controller = None  # Set to None to handle gracefully


def cleanup_temp_file(file_path: Path) -> None:
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file: {file_path}")
    except (OSError, PermissionError, FileNotFoundError) as e:
        logger.warning(
            f"Failed to cleanup temp file {file_path}: {e}"
        )  # Warning, not error - cleanup is non-critical


@router.post("/analyze", response_model=QCResponse)
async def analyze_eeg(
    edf_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    cache_client: Any = None,  # Dependency injection
) -> QCResponse:
    """Analyze uploaded EEG file for quality control and abnormality detection.

    This endpoint performs comprehensive EEG analysis including:
    - Quality control checks (bad channels, artifacts)
    - Abnormality detection using EEGPT
    - Triage recommendations

    Args:
        edf_file: Uploaded EDF file containing EEG data
        background_tasks: FastAPI background tasks (for cleanup)
        cache_client: Redis cache client (optional)

    Returns:
        QCResponse with quality metrics and recommendations
    """
    # Validate file type
    if not edf_file.filename or not edf_file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    # Read file content for caching
    content = await edf_file.read()
    await edf_file.seek(0)  # Reset file pointer

    # Check cache if available
    if cache_client and cache_client.connected:
        cache_key = cache_client.generate_cache_key(content, "basic")
        cached_result = cache_client.get(cache_key)

        if cached_result:
            logger.info(f"Returning cached result for {edf_file.filename}")
            # Convert cached dict back to QCResponse
            cached_result["cached"] = True
            return QCResponse(**cached_result)

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
            raw = load_edf_safe(tmp_path, preload=True, verbose=False)

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
                flag = "URGENT"
                recommendation = "Immediate clinical review required"
            elif abnormal_prob > 0.5 or quality_grade == "FAIR":
                flag = "EXPEDITE"
                recommendation = "Priority clinical review recommended"
            else:
                flag = "ROUTINE"
                recommendation = "Standard clinical workflow"

            # Calculate processing time
            processing_time = time.time() - start_time

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, tmp_path)

            # Prepare response
            response_data = {
                "flag": flag,
                "confidence": 1.0 - abnormal_prob,  # Convert to normalcy confidence
                "bad_channels": bad_channels,
                "quality_metrics": {
                    "bad_channel_percentage": bad_pct,
                    "abnormality_score": abnormal_prob,
                    "quality_grade": quality_grade,
                    "total_channels": quality_metrics.get("total_channels", 0),
                    "artifact_percentage": quality_metrics.get("artifact_ratio", 0) * 100,
                },
                "recommendation": recommendation,
                "processing_time": round(processing_time, 2),
                "quality_grade": quality_grade,
                "timestamp": utc_now().isoformat(),
            }

            # Cache the result if cache is available
            if cache_client and cache_client.connected:
                cache_client.set(cache_key, response_data, expiry=3600)  # 1 hour cache

            return QCResponse(**response_data)

        except EdfLoadError as e:
            # EDF loading errors
            logger.error(f"Failed to load EDF file: {e}")
            logger.debug("Full traceback:", exc_info=True)
            error_msg = str(e)
        except QualityCheckError as e:
            # Expected QC processing errors
            logger.error(f"Quality check failed: {e}")
            error_msg = str(e)
        except Exception as e:
            # Unexpected error - re-raise after logging
            logger.critical(f"Unexpected error processing EEG file: {e}")
            logger.debug("Full traceback:", exc_info=True)
            raise

        # Cleanup on error
        if tmp_path.exists():
            tmp_path.unlink()

        # Return error response
        return QCResponse(
            flag="ERROR",
            confidence=0,
            bad_channels=[],
            quality_metrics={},
            recommendation="Analysis failed. Please check file format and try again.",
            processing_time=0,
            quality_grade="ERROR",
            timestamp=utc_now().isoformat(),
            error=error_msg,
        )


@router.post("/analyze/detailed", response_class=JSONResponse)
async def analyze_eeg_detailed(
    edf_file: UploadFile = File(...),
    include_report: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> JSONResponse:
    """Detailed EEG analysis with optional PDF report.

    Returns comprehensive analysis results with PDF report.
    """
    # Read file content for caching
    content = await edf_file.read()
    await edf_file.seek(0)  # Reset file pointer

    cache_client = get_cache()  # Get cache through dependency

    # Check cache if available
    if cache_client and cache_client.connected:
        cache_key = cache_client.generate_cache_key(content, "detailed")
        cached_result = cache_client.get(cache_key)

        if cached_result:
            logger.info(f"Returning cached detailed result for {edf_file.filename}")
            cached_result["basic"]["cached"] = True
            return JSONResponse(content=json.loads(json.dumps(cached_result, cls=NumpyEncoder)))

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
            # Save uploaded file
            tmp_file.write(content)
            tmp_file.flush()

            # Check memory requirements
            import psutil

            available_memory = psutil.virtual_memory().available
            required_memory = len(content) * 10  # Rough estimate

            if required_memory > available_memory:
                raise HTTPException(
                    status_code=507,
                    detail="Insufficient memory for analysis. Try a smaller file.",
                )

            # Load EDF data
            raw = load_edf_safe(tmp_path, preload=True, verbose=False)

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
                flag = "URGENT"
                recommendation = "Immediate clinical review required"
            elif abnormal_prob > 0.5 or quality_grade == "FAIR":
                flag = "EXPEDITE"
                recommendation = "Priority clinical review recommended"
            else:
                flag = "ROUTINE"
                recommendation = "Standard clinical workflow"

            # Generate report if requested
            report_base64 = None
            if include_report:
                from brain_go_brrr.visualization.pdf_report import PDFReportGenerator

                report_gen = PDFReportGenerator()
                report_bytes = report_gen.generate_report(results)
                report_base64 = base64.b64encode(report_bytes).decode("utf-8")

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, tmp_path)

            # Prepare detailed response
            detailed_response = {
                "basic": {
                    "flag": flag,
                    "confidence": 1.0 - abnormal_prob,
                    "bad_channels": bad_channels,
                    "quality_metrics": {
                        "bad_channel_percentage": bad_pct,
                        "abnormality_score": abnormal_prob,
                        "quality_grade": quality_grade,
                        "total_channels": quality_metrics.get("total_channels", 0),
                        "artifact_percentage": quality_metrics.get("artifact_ratio", 0) * 100,
                    },
                    "recommendation": recommendation,
                    "processing_time": quality_metrics.get("processing_time", 0),
                    "quality_grade": quality_grade,
                    "timestamp": utc_now().isoformat(),
                },
                "detailed_metrics": results,
                "report": report_base64,
            }

            # Cache the result
            if cache_client and cache_client.connected:
                cache_client.set(cache_key, detailed_response, ttl=3600)

            # Return with custom encoder to handle numpy types
            return JSONResponse(content=json.loads(json.dumps(detailed_response, cls=NumpyEncoder)))

        except EdfLoadError as e:
            logger.error(f"Failed to load EDF file: {e}")
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(status_code=400, detail=str(e)) from e
        except QualityCheckError as e:
            logger.error(f"Error in detailed analysis: {e}")
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Analysis error: {e!s}") from e
        except ImportError as e:
            logger.error(f"Missing dependency for detailed analysis: {e}")
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(status_code=501, detail="PDF generation not available") from e
        except Exception as e:
            # Unexpected error - re-raise
            logger.critical(f"Unexpected error in detailed analysis: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise
