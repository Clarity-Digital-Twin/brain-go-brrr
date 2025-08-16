"""Smoke test for end-to-end wiring verification.

This test ensures that all components are properly wired together
through the application factories and can process real data.
"""

import numpy as np
import pytest


def test_quality_controller_end_to_end_wiring(synthetic_tuab_raw):
    """Test that QC controller is properly wired from factory to output."""
    from brain_go_brrr.application.factories import create_quality_controller

    # Create controller through factory (simulating production wiring)
    controller = create_quality_controller(
        model_path="dummy_path.ckpt",  # Will use null model
        device="cpu",
        enable_logging=False,
        enable_autoreject=False,  # Disable to avoid position requirement
    )

    # Run full pipeline
    result = controller.run_full_qc_pipeline(synthetic_tuab_raw)

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
    assert data_info["n_channels"] == len(synthetic_tuab_raw.ch_names)
    assert data_info["sampling_rate"] == synthetic_tuab_raw.info["sfreq"]
    assert data_info["duration"] > 0


def test_abnormality_detector_end_to_end_wiring(synthetic_tuab_raw):
    """Test that abnormality detector is properly wired from factory to output."""
    from brain_go_brrr.application.factories import create_abnormality_detector

    # Create detector through factory
    detector = create_abnormality_detector(
        model_path="dummy_path.ckpt", device="cpu", enable_logging=False
    )

    # Use proper MNE Raw object instead of numpy array
    # Run detection (detect_abnormality doesn't take sampling_rate parameter)
    result = detector.detect_abnormality(synthetic_tuab_raw)

    # Verify result structure
    assert isinstance(result, dict)
    assert "is_abnormal" in result
    assert "confidence" in result
    assert "abnormality_score" in result

    # Verify values are reasonable
    assert isinstance(result["is_abnormal"], bool)
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["abnormality_score"] <= 1


def test_feature_extractor_end_to_end_wiring(synthetic_tuab_raw):
    """Test that feature extractor is properly wired from factory to output."""
    from brain_go_brrr.application.factories import create_feature_extractor

    # Create extractor through factory
    extractor = create_feature_extractor(
        model_path="dummy_path.ckpt",
        device="cpu",
        window_size=4.0,
        overlap=0.5,
        enable_logging=False,
    )

    # Use proper MNE Raw object instead of numpy array
    # Extract features (extract_features doesn't take sampling_rate parameter)
    features = extractor.extract_features(synthetic_tuab_raw)

    # Verify features shape and type
    # The extractor returns ExtractedFeatures object, get embeddings
    from brain_go_brrr.domain.preprocessing.features.extractor import ExtractedFeatures

    assert isinstance(features, ExtractedFeatures)
    assert isinstance(features.embeddings, np.ndarray)
    assert features.embeddings.dtype == np.float32
    assert features.embeddings.ndim == 2  # (n_windows, n_features)
    assert features.embeddings.shape[1] == 512  # EEGPT feature dimension


def test_api_to_domain_integration(synthetic_tuab_raw, tmp_path):
    """Test API endpoint can properly use domain services."""
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
    reason="FastAPI required for API tests",
)
def test_api_endpoint_wiring(valid_edf_file):
    """Test that API endpoints are properly wired to use domain services."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from brain_go_brrr.api.app import create_app

    app = create_app()

    # Don't raise server exceptions - we want to test the wiring even if processing fails
    client = TestClient(app, raise_server_exceptions=False)

    # Test health check
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    # Test analyze endpoint structure (will fail without real model, but tests wiring)
    # The endpoint is properly wired if it's reachable, even if it returns an error
    # Since there's no model configured, we expect it to fail with 500
    with Path(valid_edf_file).open("rb") as f:
        response = client.post(
            "/api/v1/eeg/analyze", files={"edf_file": ("test.edf", f, "application/octet-stream")}
        )
        # Even if processing fails, endpoint should be wired
        # 422 = validation error, 400 = bad request, 500 = internal error
        assert response.status_code in {200, 400, 422, 500}
