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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.qc_flagger import EEGQualityController
from src.brain_go_brrr.visualization.markdown_report import MarkdownReportGenerator
from src.brain_go_brrr.visualization.pdf_report import PDFReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Brain-Go-Brrr API",
    description="Production-ready EEG analysis API with EEGPT",
    version="0.1.0"
)

# Global model instance (loaded once)
# Support environment variable for model path (useful for Docker/cloud)
default_model_path = project_root / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
EEGPT_MODEL_PATH = Path(os.getenv("EEGPT_MODEL_PATH", str(default_model_path))).absolute()
qc_controller = None


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


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global qc_controller
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


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global qc_controller
    if qc_controller is not None and hasattr(qc_controller, 'cleanup'):
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
            "/api/v1/eeg/analyze": "Upload and analyze EEG file"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "eegpt_loaded": qc_controller is not None and qc_controller.eegpt_model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/v1/eeg/analyze", response_model=QCResponse)
async def analyze_eeg(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze uploaded EEG file.

    MVP: Auto-QC + Risk Flagger
    - Detect bad channels
    - Compute abnormality probability
    - Return triage flag

    Args:
        file: Uploaded EDF file

    Returns:
        QC analysis results
    """
    # Validate file type
    if not file.filename.lower().endswith('.edf'):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
            # Save uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

            # Load EEG data
            logger.info(f"Processing file: {file.filename}")
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

            # Check if QC controller is available
            if qc_controller is None:
                raise RuntimeError("QC controller not initialized. Please check logs.")

            # Run QC analysis
            logger.info("Running QC analysis...")
            results = qc_controller.run_full_qc_pipeline(raw)

            # Extract key metrics
            quality_metrics = results.get('quality_metrics', {})
            bad_channels = quality_metrics.get('bad_channels', [])
            bad_pct = quality_metrics.get('bad_channel_ratio', 0) * 100
            abnormal_prob = quality_metrics.get('abnormality_score', 0)
            quality_grade = quality_metrics.get('quality_grade', 'UNKNOWN')

            # Determine triage flag
            if abnormal_prob > 0.8 or quality_grade == 'POOR':
                flag = "URGENT - Expedite read"
            elif abnormal_prob > 0.6 or quality_grade == 'FAIR':
                flag = "EXPEDITE - Priority review"
            elif abnormal_prob > 0.4:
                flag = "ROUTINE - Standard workflow"
            else:
                flag = "NORMAL - Low priority"

            # Get confidence score
            if 'processing_info' in results and isinstance(results['processing_info'].get('confidence'), float):
                confidence = results['processing_info']['confidence']
            else:
                confidence = 0.8 if qc_controller.eegpt_model is not None else 0.5

            # Processing time
            processing_time = results.get('processing_time', 0)

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, tmp_path)

            return QCResponse(
                status="success",
                bad_channels=bad_channels,
                bad_pct=round(bad_pct, 1),
                abnormal_prob=round(abnormal_prob, 3),
                flag=flag,
                confidence=round(confidence, 3),
                processing_time=round(processing_time, 2),
                quality_grade=quality_grade,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

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
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=str(e)
            )


@app.post("/api/v1/eeg/analyze/detailed")
async def analyze_eeg_detailed(
    file: UploadFile = File(...),
    include_report: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Detailed EEG analysis with optional PDF report.

    Returns comprehensive analysis results with PDF report.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

        try:
            # Save uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

            # Load EEG data
            logger.info(f"Processing file for detailed analysis: {file.filename}")
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

            # Check if QC controller is available
            if qc_controller is None:
                raise RuntimeError("QC controller not initialized. Please check logs.")

            # Run QC analysis
            logger.info("Running detailed QC analysis...")
            results = qc_controller.run_full_qc_pipeline(raw)

            # Extract key metrics for basic response
            quality_metrics = results.get('quality_metrics', {})
            bad_channels = quality_metrics.get('bad_channels', [])
            bad_pct = quality_metrics.get('bad_channel_ratio', 0) * 100
            abnormal_prob = quality_metrics.get('abnormality_score', 0)
            quality_grade = quality_metrics.get('quality_grade', 'UNKNOWN')

            # Determine triage flag
            if abnormal_prob > 0.8 or quality_grade == 'POOR':
                flag = "URGENT - Expedite read"
            elif abnormal_prob > 0.6 or quality_grade == 'FAIR':
                flag = "EXPEDITE - Priority review"
            elif abnormal_prob > 0.4:
                flag = "ROUTINE - Standard workflow"
            else:
                flag = "NORMAL - Low priority"

            # Get confidence score
            if 'processing_info' in results and isinstance(results['processing_info'].get('confidence'), float):
                confidence = results['processing_info']['confidence']
            else:
                confidence = 0.8 if qc_controller.eegpt_model is not None else 0.5

            # Processing time
            processing_time = results.get('processing_time', 0)

            # Generate reports if requested
            pdf_base64 = None
            markdown_report = None
            if include_report:
                try:
                    # Add more details to results for report generation
                    report_results = {
                        'quality_metrics': {
                            **quality_metrics,
                            'channel_positions': _get_channel_positions(raw)
                        },
                        'processing_info': {
                            'file_name': file.filename,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'duration_seconds': raw.times[-1],
                            'sampling_rate': raw.info['sfreq']
                        }
                    }

                    # Generate PDF
                    pdf_generator = PDFReportGenerator()
                    pdf_bytes = pdf_generator.generate_report(report_results, raw.get_data())

                    # Convert to base64 for JSON response
                    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                    logger.info("PDF report generated successfully")

                    # Generate Markdown report
                    markdown_generator = MarkdownReportGenerator()
                    markdown_report = markdown_generator.generate_report(report_results)
                    logger.info("Markdown report generated successfully")

                    # Optionally save markdown to outputs directory
                    output_dir = project_root / "outputs" / "reports"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Create filename with timestamp
                    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            return {
                "basic": basic_result,
                "detailed": {
                    "message": "Detailed analysis complete",
                    "pdf_available": pdf_base64 is not None,
                    "pdf_base64": pdf_base64,
                    "markdown_available": markdown_report is not None,
                    "markdown_report": markdown_report,
                    "artifact_count": len(quality_metrics.get('artifact_segments', [])),
                    "channel_count": len(raw.ch_names),
                    "duration_seconds": raw.times[-1]
                }
            }

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
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error=str(e)
                ),
                "detailed": {
                    "message": "Detailed analysis failed",
                    "pdf_available": False,
                    "error": str(e)
                }
            }


def _get_channel_positions(raw: mne.io.Raw) -> dict[str, tuple[float, float]]:
    """Extract channel positions from raw data."""
    positions = {}

    try:
        # Get montage if available
        montage = raw.get_montage()
        if montage is not None:
            ch_names = montage.ch_names
            pos = montage.get_positions()['ch_pos']

            # Convert 3D positions to 2D (x, y) for visualization
            for ch_name in ch_names:
                if ch_name in pos:
                    x, y, z = pos[ch_name]
                    positions[ch_name] = (x, y)
        else:
            # Fallback to standard 10-20 positions for common channels
            standard_positions = {
                'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8),
                'F3': (-0.5, 0.5), 'F4': (0.5, 0.5),
                'C3': (-0.5, 0), 'C4': (0.5, 0),
                'P3': (-0.5, -0.5), 'P4': (0.5, -0.5),
                'O1': (-0.3, -0.8), 'O2': (0.3, -0.8),
                'F7': (-0.8, 0.3), 'F8': (0.8, 0.3),
                'T3': (-0.8, 0), 'T4': (0.8, 0),
                'T5': (-0.8, -0.3), 'T6': (0.8, -0.3),
                'Fz': (0, 0.6), 'Cz': (0, 0), 'Pz': (0, -0.6)
            }

            # Use standard positions for channels that match
            for ch_name in raw.ch_names:
                if ch_name in standard_positions:
                    positions[ch_name] = standard_positions[ch_name]

    except Exception as e:
        logger.warning(f"Failed to extract channel positions: {e}")

    return positions


def cleanup_temp_file(file_path: Path):
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
