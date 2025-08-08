"""EEGPT-based analysis endpoints using linear probes."""

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from brain_go_brrr.core.edf_loader import load_edf_safe
from brain_go_brrr.core.exceptions import EdfLoadError
from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.models.linear_probe import (
    AbnormalityProbe,
    SleepStageProbe,
    create_probe_for_task,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eeg/eegpt", tags=["eegpt"])


class EEGPTAnalysisResponse(BaseModel):
    """Response for EEGPT-based analysis."""

    analysis_type: str
    result: dict[str, Any]
    confidence: float
    method: str = "linear_probe"
    metadata: dict[str, Any]


class ProbeInfoResponse(BaseModel):
    """Information about available linear probes."""

    available_probes: list[str]
    probe_info: dict[str, dict[str, Any]]


# Global instances (would be dependency injected in production)
_eegpt_model: EEGPTModel | None = None
_probes: dict[str, Any] = {}


def get_eegpt_model() -> EEGPTModel:
    """Get or initialize EEGPT model."""
    global _eegpt_model
    if _eegpt_model is None:
        _eegpt_model = EEGPTModel()
    return _eegpt_model


def get_probe(task: str) -> Any:
    """Get or initialize probe for specific task."""
    global _probes
    if task not in _probes:
        _probes[task] = create_probe_for_task(task)
    return _probes[task]


@router.post("/analyze", response_model=EEGPTAnalysisResponse)  # type: ignore[misc]
async def analyze_with_probe(
    edf_file: UploadFile = File(...),
    analysis_type: Literal[
        "abnormality_probe", "sleep_probe", "motor_imagery_probe"
    ] = "abnormality_probe",
) -> EEGPTAnalysisResponse:
    """Analyze EEG data using EEGPT with task-specific linear probe.

    This endpoint uses pretrained EEGPT features with linear probes
    for various downstream tasks. No fine-tuning required!

    Args:
        edf_file: Uploaded EDF file
        analysis_type: Type of analysis to perform

    Returns:
        Analysis results with confidence scores
    """
    # Validate file
    if not edf_file.filename or not edf_file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    # Read content
    content = await edf_file.read()

    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="File too small to be valid EDF")

    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(content)
        tmp_file.flush()

    try:
        # Load EDF
        raw = load_edf_safe(tmp_path, preload=True, verbose=False)
        data = raw.get_data()
        channel_names = raw.ch_names
        sfreq = int(raw.info["sfreq"])

        # Get model and probe
        eegpt_model = get_eegpt_model()

        # Map analysis type to probe
        probe_map = {
            "abnormality_probe": "abnormality",
            "sleep_probe": "sleep",
            "motor_imagery_probe": "motor_imagery",
        }
        probe_task = probe_map.get(analysis_type, "abnormality")
        probe = get_probe(probe_task)

        # Process windows
        window_duration = 4.0
        window_samples = int(window_duration * sfreq)
        n_windows = data.shape[1] // window_samples

        all_predictions: list[Any] = []
        all_confidences: list[float] = []

        for i in range(n_windows):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            window_data = data[:, start_idx:end_idx]

            # Extract EEGPT features
            features = eegpt_model.extract_features(window_data, channel_names)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Get prediction based on probe type
            with torch.no_grad():
                if isinstance(probe, AbnormalityProbe):
                    abnormal_prob = probe.predict_abnormal_probability(features_tensor)
                    all_predictions.append(abnormal_prob.item())
                    all_confidences.append(abnormal_prob.item())
                elif isinstance(probe, SleepStageProbe):
                    stages, confidences = probe.predict_stage(features_tensor)
                    all_predictions.extend(stages)
                    all_confidences.extend(confidences.tolist())
                else:
                    # Generic probe
                    probs = probe.predict_proba(features_tensor)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs.max().item()
                    all_predictions.append(pred_class)
                    all_confidences.append(confidence)

        # Aggregate results
        if analysis_type == "abnormality_probe":
            mean_abnormal = np.mean(all_predictions) if all_predictions else 0.0
            result = {
                "abnormal_probability": float(mean_abnormal),
                "window_scores": all_predictions[:10],  # First 10 windows
                "n_windows": len(all_predictions),
            }
        elif analysis_type == "sleep_probe":
            # Aggregate into hypnogram
            result = {
                "stages": all_predictions[:50],  # First 50 predictions
                "confidence_scores": all_confidences[:50],
                "n_windows": len(all_predictions),
            }
        else:
            result = {
                "predictions": all_predictions[:10],
                "confidences": all_confidences[:10],
                "n_windows": len(all_predictions),
            }

        mean_confidence = np.mean(all_confidences) if all_confidences else 0.0

        return EEGPTAnalysisResponse(
            analysis_type=analysis_type,
            result=result,
            confidence=float(mean_confidence),
            method="linear_probe",
            metadata={
                "n_channels": len(channel_names),
                "sampling_rate": sfreq,
                "duration_seconds": data.shape[1] / sfreq,
                "model": "eegpt_10m",
                "probe_type": probe_task,
            },
        )

    except EdfLoadError as e:
        raise HTTPException(status_code=400, detail=f"Failed to load EDF: {e}") from e
    except Exception as e:
        logger.error(f"Error in EEGPT analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        # Cleanup
        with contextlib.suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


@router.get("/probes/available", response_model=ProbeInfoResponse)  # type: ignore[misc]
async def get_available_probes() -> ProbeInfoResponse:
    """Get information about available linear probes."""
    probe_info = {
        "sleep": {
            "description": "5-stage sleep classification (W, N1, N2, N3, REM)",
            "num_classes": 5,
            "input_dim": 2048,
        },
        "abnormality": {
            "description": "Binary abnormality detection (normal/abnormal)",
            "num_classes": 2,
            "input_dim": 2048,
        },
        "motor_imagery": {
            "description": "Motor imagery classification (left/right hand, feet, tongue)",
            "num_classes": 4,
            "input_dim": 2048,
        },
    }

    return ProbeInfoResponse(
        available_probes=list(probe_info.keys()),
        probe_info=probe_info,
    )


@router.post("/sleep/stages")  # type: ignore[misc]
async def analyze_sleep_stages(
    edf_file: UploadFile = File(...),
) -> dict[str, Any]:
    """Analyze sleep stages from EEG file.

    Returns window-by-window sleep stage predictions.
    """
    # Validate file
    if not edf_file.filename or not edf_file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    content = await edf_file.read()

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(content)
        tmp_file.flush()

    try:
        # Load EDF
        raw = load_edf_safe(tmp_path, preload=True, verbose=False)
        data = raw.get_data()
        channel_names = raw.ch_names

        # Get model
        eegpt_model = get_eegpt_model()

        # Extract windows
        windows = eegpt_model.extract_windows(data, int(raw.info["sfreq"]))
        logger.info(f"Extracted {len(windows)} windows from {data.shape} data")

        # Get sleep probe
        probe = get_probe("sleep")

        stages = []
        confidence_scores = []

        for window in windows:
            # Extract features
            features = eegpt_model.extract_features(window, channel_names)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Predict stage
            with torch.no_grad():
                stage, confidence = probe.predict_stage(features_tensor)
                stages.extend(stage)
                confidence_scores.append(confidence.item())

        return {
            "stages": stages,
            "confidence_scores": confidence_scores,
            "total_windows": len(windows),
            "sampling_rate": int(raw.info["sfreq"]),
        }

    except Exception as e:
        logger.error(f"Error in sleep stage analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Sleep analysis failed") from e
    finally:
        with contextlib.suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


@router.post("/analyze/batch")  # type: ignore[misc]
async def analyze_batch(
    edf_file: UploadFile = File(...),
    analysis_type: str = "abnormality",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Batch process multiple windows from EEG file.

    Efficient batch processing for longer recordings.
    """
    # Validate file
    if not edf_file.filename or not edf_file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")

    content = await edf_file.read()

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(content)
        tmp_file.flush()

    try:
        # Load EDF
        raw = load_edf_safe(tmp_path, preload=True, verbose=False)
        data = raw.get_data()
        channel_names = raw.ch_names

        # Get model
        eegpt_model = get_eegpt_model()

        # Extract windows
        windows = eegpt_model.extract_windows(data, int(raw.info["sfreq"]))

        # Process in batches
        results = []
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_array = np.stack(batch_windows)

            # Extract features for batch
            batch_features = eegpt_model.extract_features_batch(batch_array, channel_names)

            # Get predictions
            probe = get_probe(analysis_type)
            features_tensor = torch.FloatTensor(batch_features)

            with torch.no_grad():
                if hasattr(probe, "predict_proba"):
                    probs = probe.predict_proba(features_tensor)
                    batch_results = probs.cpu().numpy().tolist()
                else:
                    batch_results = [[0.5, 0.5]] * len(batch_windows)

            results.extend(batch_results)

        return {
            "analysis_type": analysis_type,
            "results": results[:10],  # First 10 for response size
            "total_windows": len(windows),
            "batch_size": batch_size,
        }

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch processing failed") from e
    finally:
        with contextlib.suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


def _reset_state_for_tests() -> None:
    """Reset global state for testing. Only use in tests!"""
    global _eegpt_model, _probes
    _eegpt_model = None
    _probes = {}
