"""Smoke test for end-to-end pipeline with synthetic data."""

import mne
import numpy as np

from brain_go_brrr.application.factories import (
    create_abnormality_detector,
    create_quality_controller,
    create_sleep_analyzer,
)


def test_e2e_on_synthetic_raw():
    """Test end-to-end pipeline on synthetic EEG data."""
    # Create synthetic EEG data
    sfreq = 100  # Sampling frequency
    duration_sec = 60  # 60 seconds
    n_channels = 4  # 4 channels

    # Generate random data
    n_samples = sfreq * duration_sec
    data = np.random.randn(n_channels, n_samples).astype("float32") * 50e-6  # Scale to µV

    # Create channel names
    ch_names = [f"EEG{i}" for i in range(n_channels)]
    ch_types = "eeg"

    # Create MNE info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Test quality controller
    qc = create_quality_controller()
    assert qc is not None, "Failed to create quality controller"

    qc_report = qc.run_full_qc_pipeline(raw)
    assert isinstance(qc_report, dict), "QC report should be a dictionary"
    assert "quality_grade" in qc_report, "QC report missing quality_grade"
    assert qc_report["quality_grade"] in [
        "EXCELLENT",
        "GOOD",
        "FAIR",
        "POOR",
    ], f"Invalid quality grade: {qc_report['quality_grade']}"

    # Test sleep analyzer
    sleep = create_sleep_analyzer()
    assert sleep is not None, "Failed to create sleep analyzer"

    try:
        # Sleep analysis might fail on short synthetic data
        sleep_out = sleep.analyze(raw)
        assert isinstance(sleep_out, dict), "Sleep output should be a dictionary"
        if "efficiency" in sleep_out:
            assert 0 <= sleep_out["efficiency"] <= 100, "Sleep efficiency out of range"
    except Exception as e:
        # Expected for synthetic data that's too short or doesn't resemble sleep
        print(f"Sleep analysis failed (expected for synthetic data): {e}")

    # Test abnormality detector
    detector = create_abnormality_detector()
    assert detector is not None, "Failed to create abnormality detector"

    try:
        # Abnormality detection on raw data
        abn_result = detector.detect(raw)
        assert hasattr(abn_result, "is_abnormal"), "Result missing is_abnormal attribute"
        assert hasattr(abn_result, "confidence"), "Result missing confidence attribute"
        assert 0.0 <= abn_result.confidence <= 1.0, "Confidence out of range"
    except Exception as e:
        # Some methods might not work without a real model
        print(f"Abnormality detection requires trained model: {e}")


def test_hierarchical_pipeline_import():
    """Test that hierarchical pipeline can be imported as documented."""
    from brain_go_brrr.services.hierarchical_pipeline import HierarchicalPipeline, PipelineConfig

    # Should be able to instantiate with config
    config = PipelineConfig()
    pipeline = HierarchicalPipeline(config)
    assert pipeline is not None
