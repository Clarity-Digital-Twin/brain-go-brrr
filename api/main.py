"""
Brain-Go-Brrr API - FastAPI application for EEG analysis.

MVP: Auto-QC + Risk Flagger endpoint
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import traceback
import logging
from datetime import datetime
import json

import mne

# Add project to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.qc_flagger import EEGQualityController

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
EEGPT_MODEL_PATH = project_root / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
qc_controller = None


class AnalysisRequest(BaseModel):
    """Request model for EEG analysis."""
    analysis_type: str = Field(default="qc", description="Type of analysis to perform")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")


class QCResponse(BaseModel):
    """Response model for QC analysis."""
    status: str
    bad_channels: List[str]
    bad_pct: float
    abnormal_prob: float
    flag: str
    confidence: float
    processing_time: float
    quality_grade: str
    timestamp: str
    error: Optional[str] = None


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
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/eeg/analyze", response_model=QCResponse)
async def analyze_eeg(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Analyze uploaded EEG file.
    
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
                timestamp=datetime.utcnow().isoformat()
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
                timestamp=datetime.utcnow().isoformat(),
                error=str(e)
            )


@app.post("/api/v1/eeg/analyze/detailed")
async def analyze_eeg_detailed(
    file: UploadFile = File(...),
    include_report: bool = True
):
    """
    Detailed EEG analysis with optional PDF report.
    
    Returns comprehensive analysis results.
    """
    # Run basic analysis first
    basic_result = await analyze_eeg(file)
    
    # TODO: Add detailed analysis
    # - Generate PDF report
    # - Include visualization
    # - Add event timeline
    
    return {
        "basic": basic_result,
        "detailed": {
            "message": "Detailed analysis coming soon",
            "pdf_available": False
        }
    }


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