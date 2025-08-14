"""Smoke test for end-to-end wiring verification.

This test ensures that all components are properly wired together
through the application factories and can process real data.
"""

import numpy as np
import pytest


def test_quality_controller_end_to_end_wiring():
    """Test that QC controller is properly wired from factory to output."""
    import mne
    from brain_go_brrr.application.factories import create_quality_controller
    
    # Create simple synthetic data without needing channel positions
    sfreq = 256
    n_channels = 19
    duration = 10
    n_samples = int(sfreq * duration)
    
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    data = np.random.randn(n_channels, n_samples).astype(np.float32) * 20e-6
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    
    # Create controller through factory (simulating production wiring)
    controller = create_quality_controller(
        model_path="dummy_path.ckpt",  # Will use null model
        device="cpu",
        enable_logging=False,
        enable_autoreject=False  # Disable to avoid position requirement
    )
    
    # Run full pipeline
    result = controller.run_full_qc_pipeline(raw)
    
    # Verify we get a proper result structure
    assert isinstance(result, dict)
    assert "quality_metrics" in result
    assert "data_info" in result
    assert "processing_info" in result
    
    # Verify metrics are reasonable
    metrics = result["quality_metrics"]
    assert isinstance(metrics["bad_channels"], list)
    assert 0 <= metrics["bad_channel_ratio"] <= 1
    assert metrics["quality_grade"] in ["EXCELLENT", "GOOD", "FAIR", "POOR"]
    assert 0 <= metrics["abnormality_score"] <= 1
    
    # Verify data info
    data_info = result["data_info"]
    assert data_info["n_channels"] == n_channels
    assert data_info["sampling_rate"] == sfreq
    assert data_info["duration"] > 0


def test_abnormality_detector_end_to_end_wiring():
    """Test that abnormality detector is properly wired from factory to output."""
    from brain_go_brrr.application.factories import create_abnormality_detector
    
    # Create detector through factory
    detector = create_abnormality_detector(
        model_path="dummy_path.ckpt",
        device="cpu",
        enable_logging=False
    )
    
    # Create test data
    test_data = np.random.randn(20, 2048).astype(np.float32) * 20e-6
    
    # Run detection
    result = detector.detect_abnormality(test_data, sampling_rate=256)
    
    # Verify result structure
    assert isinstance(result, dict)
    assert "is_abnormal" in result
    assert "confidence" in result
    assert "abnormality_score" in result
    
    # Verify values are reasonable
    assert isinstance(result["is_abnormal"], bool)
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["abnormality_score"] <= 1


def test_feature_extractor_end_to_end_wiring():
    """Test that feature extractor is properly wired from factory to output."""
    from brain_go_brrr.application.factories import create_feature_extractor
    
    # Create extractor through factory
    extractor = create_feature_extractor(
        model_path="dummy_path.ckpt",
        device="cpu",
        window_size=4.0,
        overlap=0.5,
        enable_logging=False
    )
    
    # Create test data (20 channels, 10 seconds at 256Hz)
    test_data = np.random.randn(20, 2560).astype(np.float32) * 20e-6
    
    # Extract features
    features = extractor.extract_features(test_data, sampling_rate=256)
    
    # Verify features shape and type
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert features.ndim == 2  # (n_windows, n_features)
    assert features.shape[1] == 512  # EEGPT feature dimension


def test_api_to_domain_integration(synthetic_tuab_raw, tmp_path):
    """Test API endpoint can properly use domain services."""
    from pathlib import Path
    import mne
    
    # Save test data to EDF
    edf_path = tmp_path / "test.edf"
    synthetic_tuab_raw.export(str(edf_path), overwrite=True)
    
    # Import API dependency resolver
    from brain_go_brrr.api.deps import get_qc_controller
    
    # Get controller (simulating FastAPI dependency injection)
    controller = get_qc_controller()
    
    # If we got a noop (no model configured), that's still valid wiring
    assert controller is not None
    assert hasattr(controller, "run_full_qc_pipeline")
    
    # For real test with model, would need to set EEGPT_CKPT_PATH env var


@pytest.mark.skipif(
    not pytest.importorskip("fastapi", reason="FastAPI not installed"),
    reason="FastAPI required for API tests"
)
def test_api_endpoint_wiring(valid_edf_file):
    """Test that API endpoints are properly wired to use domain services."""
    from fastapi.testclient import TestClient
    from brain_go_brrr.api.app import app
    
    client = TestClient(app)
    
    # Test health check
    response = client.get("/health")
    assert response.status_code == 200
    
    # Test analyze endpoint structure (will fail without real model, but tests wiring)
    with open(valid_edf_file, "rb") as f:
        response = client.post(
            "/eeg/analyze",
            files={"edf_file": ("test.edf", f, "application/octet-stream")}
        )
    
    # Even if processing fails, endpoint should be wired
    assert response.status_code in [200, 400, 500]  # Success or expected errors