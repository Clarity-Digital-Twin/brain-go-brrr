"""EEGPT inference pipelines.

High-level orchestration functions extracted from eegpt_model.py.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import numpy as np
import torch
import mne

from brain_go_brrr._typing import MNERaw
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.domain.preprocessing.eegpt_preprocessing import (
    preprocess_for_eegpt,
    extract_windows,
    prepare_batch_for_eegpt,
    validate_eeg_input
)

logger = logging.getLogger(__name__)


def predict_abnormality(
    model_or_path: Any,
    raw: MNERaw,
    probe_path: Optional[Path] = None,
    window_duration: float = 4.0,
    overlap: float = 0.5,
    device: str = "auto"
) -> Dict[str, Any]:
    """Run abnormality detection pipeline on EEG data.
    
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
    if isinstance(model_or_path, (str, Path)):
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
        probe = EEGPTProbe(
            backbone=model,
            n_classes=2,
            architecture="linear"
        )
        checkpoint = torch.load(probe_path, map_location=device)
        probe.load_state_dict(checkpoint['model_state_dict'])
        probe = probe.to(device)
        probe.eval()
        logger.info(f"Loaded probe from {probe_path}")
    
    # Preprocess data
    data = preprocess_for_eegpt(raw)
    
    # Validate
    is_valid, message = validate_eeg_input(
        data, 
        expected_samples=int(raw.info['sfreq'] * window_duration)
    )
    if not is_valid:
        logger.warning(f"Input validation warning: {message}")
    
    # Extract windows
    windows = extract_windows(data, window_duration, int(raw.info['sfreq']), overlap)
    
    # Prepare batch
    batch = prepare_batch_for_eegpt(windows, device=device)
    
    # Run inference
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for i in range(0, len(batch), 32):  # Process in mini-batches
            mini_batch = batch[i:i+32]
            
            if probe:
                # Use trained probe
                logits = probe(mini_batch, return_all_temporal=True)
                probs = torch.softmax(logits, dim=-1)
                abnormal_prob = probs[:, 1]  # Abnormal class probability
            else:
                # Use features directly (no trained probe)
                features = model.extract_features(mini_batch, return_all_temporal=True)
                # Simple heuristic: use mean activation as abnormality score
                abnormal_prob = features.mean(dim=(1, 2, 3))
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
        "triage": _get_triage_level(overall_confidence, overall_prediction)
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


def analyze(
    model_or_path: Any,
    raw: MNERaw,
    analysis_type: Literal["abnormality", "sleep", "events", "quality"] = "abnormality",
    **kwargs
) -> Dict[str, Any]:
    """Orchestrate different EEG analysis types.
    
    Args:
        model_or_path: EEGPT model instance or path
        raw: MNE Raw object
        analysis_type: Type of analysis to perform
        **kwargs: Additional arguments for specific analysis
        
    Returns:
        Analysis results dictionary
    """
    if analysis_type == "abnormality":
        return predict_abnormality(model_or_path, raw, **kwargs)
    
    elif analysis_type == "sleep":
        # Delegate to YASA adapter
        from brain_go_brrr.services.yasa_adapter import YASASleepStager
        stager = YASASleepStager()
        
        # Get EEGPT features for enhanced analysis
        model = _ensure_model(model_or_path)
        data = preprocess_for_eegpt(raw)
        windows = extract_windows(data, window_duration=30.0, overlap=0)  # 30s epochs for sleep
        batch = prepare_batch_for_eegpt(windows)
        
        with torch.no_grad():
            features = model.extract_features(batch, return_all_temporal=True)
            features = features.cpu().numpy()
        
        # Run YASA with EEGPT features as additional input
        results = stager.stage_sleep(raw)
        results['eegpt_features'] = features
        return results
    
    elif analysis_type == "events":
        # Event detection (epileptiform, etc.)
        return _detect_events(model_or_path, raw, **kwargs)
    
    elif analysis_type == "quality":
        # Quality control
        return _assess_quality(model_or_path, raw, **kwargs)
    
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def _ensure_model(model_or_path: Any) -> Any:
    """Ensure we have a model instance."""
    if isinstance(model_or_path, (str, Path)):
        return create_normalized_eegpt(str(model_or_path))
    return model_or_path


def _detect_events(model: Any, raw: MNERaw, **kwargs) -> Dict[str, Any]:
    """Detect EEG events (placeholder for future implementation)."""
    logger.warning("Event detection not yet implemented")
    return {
        "events": [],
        "message": "Event detection coming soon"
    }


def _assess_quality(model: Any, raw: MNERaw, **kwargs) -> Dict[str, Any]:
    """Assess EEG data quality."""
    from brain_go_brrr.services.qc_flagger import QualityController
    
    qc = QualityController()
    results = qc.assess_quality(raw)
    
    # Enhance with EEGPT features
    model = _ensure_model(model)
    data = preprocess_for_eegpt(raw)
    batch = prepare_batch_for_eegpt([data])
    
    with torch.no_grad():
        features = model.extract_features(batch)
        # Use feature statistics as quality indicators
        feature_std = features.std().item()
        feature_mean = features.mean().item()
    
    results['eegpt_quality_score'] = 1.0 / (1.0 + abs(feature_mean) + feature_std)
    
    return results


def batch_analyze(
    model_or_path: Any,
    edf_paths: list[Path],
    analysis_type: str = "abnormality",
    output_dir: Optional[Path] = None,
    **kwargs
) -> list[Dict[str, Any]]:
    """Analyze multiple EDF files in batch.
    
    Args:
        model_or_path: EEGPT model or path
        edf_paths: List of EDF file paths
        analysis_type: Type of analysis
        output_dir: Directory to save results (optional)
        **kwargs: Additional arguments
        
    Returns:
        List of analysis results
    """
    model = _ensure_model(model_or_path)
    results = []
    
    for edf_path in edf_paths:
        try:
            logger.info(f"Processing {edf_path.name}")
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            result = analyze(model, raw, analysis_type, **kwargs)
            result['file'] = str(edf_path)
            results.append(result)
            
            # Save if requested
            if output_dir:
                output_dir.mkdir(exist_ok=True, parents=True)
                output_file = output_dir / f"{edf_path.stem}_results.json"
                import json
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to process {edf_path}: {e}")
            results.append({
                'file': str(edf_path),
                'error': str(e)
            })
    
    return results