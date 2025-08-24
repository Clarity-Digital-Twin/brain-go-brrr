"""EEGPT orchestration functions extracted from eegpt_model.py.

High-level pipeline functions that coordinate EEGPT inference.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_go_brrr._typing import MNERaw
from brain_go_brrr.domain.preprocessing.eegpt_preprocessing import (
    extract_windows,
    prepare_batch_for_eegpt,
    preprocess_for_eegpt,
    validate_eeg_input,
)
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt

logger = logging.getLogger(__name__)


def predict_abnormality_with_eegpt(
    model_or_path: Any,
    raw: MNERaw,
    probe_path: Path | None = None,
    window_duration: float = 4.0,
    overlap: float = 0.5,
    device: str = "auto",
) -> dict[str, Any]:
    """Run abnormality detection using EEGPT.

    This is the high-level orchestration function that was previously
    part of EEGPTModel class.

    Args:
        model_or_path: EEGPT model instance or path to checkpoint
        raw: MNE Raw object with EEG data
        probe_path: Path to trained probe weights (optional)
        window_duration: Window duration in seconds
        overlap: Window overlap fraction
        device: Device for inference

    Returns:
        Dictionary with predictions and confidence scores
    """
    # Load model if path provided
    if isinstance(model_or_path, str | Path):
        model = create_normalized_eegpt(str(model_or_path))
    else:
        model = model_or_path

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load probe if provided
    probe = None
    if probe_path and Path(probe_path).exists():
        from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe

        probe = EEGPTProbe(backbone=model, n_classes=2, architecture="linear")
        checkpoint = torch.load(probe_path, map_location=device, weights_only=True)
        probe.load_state_dict(checkpoint['model_state_dict'])
        probe = probe.to(device)
        probe.eval()
        logger.info(f"Loaded probe from {probe_path}")

    # Preprocess data
    data = preprocess_for_eegpt(raw)

    # Validate
    is_valid, message = validate_eeg_input(
        data, expected_samples=int(raw.info['sfreq'] * window_duration)
    )
    if not is_valid:
        logger.warning(f"Input validation warning: {message}")

    # Extract windows
    windows = extract_windows(data, window_duration, int(raw.info['sfreq']), overlap)

    # Prepare batch
    batch = prepare_batch_for_eegpt(windows, device=device)

    # Run inference
    predictions: list[int] = []
    confidences: list[float] = []

    with torch.no_grad():
        for i in range(0, len(batch), 32):  # Process in mini-batches
            mini_batch = batch[i : i + 32]

            if probe:
                # Use trained probe with summary-token features (flattened to 2,048 internally)
                logits = probe(mini_batch, return_all_temporal=False)
                probs = torch.softmax(logits, dim=-1)
                abnormal_prob = probs[:, 1]  # Abnormal class probability
            else:
                # Use features directly (no trained probe)
                features = model.extract_features(mini_batch, summary=False)  # (B, 4, 512)
                # Simple heuristic: mean across summary tokens and embedding dims
                abnormal_prob = features.mean(dim=(1, 2))
                abnormal_prob = torch.sigmoid(abnormal_prob)  # Convert to probability

            predictions.extend((abnormal_prob > 0.5).cpu().numpy())
            confidences.extend(abnormal_prob.cpu().numpy())

    # Aggregate results
    overall_prediction = int(np.mean(predictions) > 0.5)
    overall_confidence = float(np.mean(confidences))

    return {
        "prediction": "abnormal" if overall_prediction else "normal",
        "confidence": overall_confidence,
        "window_predictions": predictions,
        "window_confidences": confidences,
        "n_windows": len(windows),
        "triage": _get_triage_level(overall_confidence, bool(overall_prediction)),
    }


def _get_triage_level(confidence: float, is_abnormal: bool) -> str:
    """Determine triage level based on prediction confidence."""
    if not is_abnormal:
        return "routine"
    elif confidence > 0.9:
        return "urgent"
    elif confidence > 0.7:
        return "expedite"
    else:
        return "review"
