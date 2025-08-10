"""CLEAN tests for EEG Quality Controller - dependency injection, no bullshit mocks."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from brain_go_brrr.core.quality.controller import EEGQualityController
from brain_go_brrr.core.exceptions import QualityCheckError


class TestEEGQualityControllerClean:
    """Test EEG Quality Controller with real logic and DI."""

    @pytest.fixture
    def synthetic_raw(self):
        """Create synthetic EEG data for testing."""
        import mne
        
        np.random.seed(1337)
        sfreq = 256
        duration = 60  # 1 minute
        n_channels = 19  # Standard 10-20 subset
        
        ch_names = [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
            "T3", "C3", "CZ", "C4", "T4",
            "T5", "P3", "PZ", "P4", "T6",
            "O1", "O2"
        ]
        
        n_samples = int(sfreq * duration)
        times = np.arange(n_samples) / sfreq
        
        # Generate clean EEG-like data
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        for i in range(n_channels):
            # Alpha rhythm (8-13 Hz)
            data[i] += 20e-6 * np.sin(2 * np.pi * 10 * times + i * 0.1)
            # Beta rhythm (13-30 Hz)
            data[i] += 10e-6 * np.sin(2 * np.pi * 20 * times + i * 0.2)
            # Some pink noise
            data[i] += 5e-6 * np.random.randn(n_samples)
            
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        
        return raw

    @pytest.fixture
    def raw_with_artifacts(self, synthetic_raw):
        """Create EEG data with artifacts."""
        raw = synthetic_raw.copy()
        data = raw.get_data()
        
        # Add bad channel (constant high amplitude noise)
        data[0, :] = 200e-6 * np.random.randn(data.shape[1])
        
        # Add eye blink artifacts
        for i in range(10):
            start = i * 600 + 100
            data[1:3, start:start+50] += 100e-6 * np.sin(np.linspace(0, np.pi, 50))
            
        # Add muscle artifacts
        data[5, 5000:5500] += 150e-6 * np.random.randn(500)
        
        # Update raw data
        raw._data = data
        
        return raw

    @pytest.fixture
    def mock_eegpt_model(self):
        """Create mock EEGPT model with DI."""
        model = MagicMock()
        model.is_loaded = True
        model.extract_features = MagicMock(return_value=np.random.randn(1, 512).astype(np.float32))
        return model

    def test_init_controller(self):
        """Test initialization of quality controller."""
        controller = EEGQualityController(
            rejection_threshold=0.1,
            interpolation_threshold=0.8,
            random_state=42
        )
        
        assert controller is not None
        assert controller.rejection_threshold == 0.1
        assert controller.interpolation_threshold == 0.8
        assert controller.random_state == 42
        assert controller.eegpt_model is None  # Not loaded yet

    def test_init_with_eegpt_path(self, tmp_path):
        """Test initialization with EEGPT model path."""
        # Create fake checkpoint
        checkpoint_path = tmp_path / "eegpt.ckpt"
        checkpoint_path.touch()
        
        controller = EEGQualityController(
            eegpt_model_path=checkpoint_path
        )
        
        assert controller.eegpt_model_path == checkpoint_path

    def test_detect_bad_channels(self, raw_with_artifacts):
        """Test bad channel detection."""
        controller = EEGQualityController()
        
        bad_channels = controller.detect_bad_channels(raw_with_artifacts)
        
        assert isinstance(bad_channels, list)
        assert len(bad_channels) > 0
        # Should detect the first channel we made bad
        assert "FP1" in bad_channels

    def test_compute_channel_statistics(self, synthetic_raw):
        """Test channel statistics computation."""
        controller = EEGQualityController()
        
        stats = controller.compute_channel_statistics(synthetic_raw)
        
        assert "mean" in stats
        assert "std" in stats
        assert "variance" in stats
        assert "peak_to_peak" in stats
        assert "kurtosis" in stats
        assert "skewness" in stats
        
        # Check shape
        n_channels = len(synthetic_raw.ch_names)
        assert stats["mean"].shape == (n_channels,)
        assert stats["std"].shape == (n_channels,)

    def test_flag_artifacts_in_epochs(self, synthetic_raw):
        """Test artifact flagging in epochs."""
        controller = EEGQualityController()
        
        # Create epochs from raw
        events = mne.make_fixed_length_events(synthetic_raw, duration=1.0)
        epochs = mne.Epochs(
            synthetic_raw, events, tmin=0, tmax=1.0,
            baseline=None, preload=True
        )
        
        # Flag artifacts
        artifact_flags = controller.flag_artifacts_in_epochs(epochs)
        
        assert len(artifact_flags) == len(epochs)
        assert all(isinstance(flag, bool) for flag in artifact_flags)

    def test_interpolate_bad_channels(self, raw_with_artifacts):
        """Test bad channel interpolation."""
        controller = EEGQualityController()
        
        # Mark bad channels
        raw_with_artifacts.info["bads"] = ["FP1"]
        
        # Interpolate
        interpolated = controller.interpolate_bad_channels(raw_with_artifacts)
        
        assert interpolated is not None
        # Bad channel should be interpolated
        assert "FP1" not in interpolated.info["bads"]

    def test_apply_autoreject_pipeline(self, synthetic_raw):
        """Test autoreject pipeline application."""
        controller = EEGQualityController()
        
        # Create epochs
        events = mne.make_fixed_length_events(synthetic_raw, duration=2.0)
        epochs = mne.Epochs(
            synthetic_raw, events, tmin=0, tmax=2.0,
            baseline=None, preload=True
        )
        
        # Mock autoreject if not available
        if not controller.has_autoreject:
            controller.autoreject = MagicMock()
            controller.autoreject.fit_transform = MagicMock(return_value=epochs)
            controller.has_autoreject = True
        
        # Apply pipeline
        cleaned_epochs = controller.apply_autoreject_pipeline(epochs)
        
        assert cleaned_epochs is not None
        assert len(cleaned_epochs) <= len(epochs)

    def test_compute_quality_metrics(self, synthetic_raw):
        """Test quality metrics computation."""
        controller = EEGQualityController()
        
        metrics = controller.compute_quality_metrics(synthetic_raw)
        
        assert "snr" in metrics
        assert "artifact_ratio" in metrics
        assert "channel_correlation" in metrics
        assert "spectral_flatness" in metrics
        assert "line_noise_ratio" in metrics
        
        # Check value ranges
        assert metrics["snr"] > 0
        assert 0 <= metrics["artifact_ratio"] <= 1
        assert -1 <= metrics["channel_correlation"] <= 1

    def test_assess_signal_quality(self, synthetic_raw, raw_with_artifacts):
        """Test overall signal quality assessment."""
        controller = EEGQualityController()
        
        # Clean data should have good quality
        clean_quality = controller.assess_signal_quality(synthetic_raw)
        assert clean_quality["overall_quality"] == "good"
        assert clean_quality["quality_score"] > 0.7
        
        # Artifacted data should have lower quality
        artifact_quality = controller.assess_signal_quality(raw_with_artifacts)
        assert artifact_quality["quality_score"] < clean_quality["quality_score"]

    def test_generate_qc_report(self, synthetic_raw):
        """Test QC report generation."""
        controller = EEGQualityController()
        
        report = controller.generate_qc_report(synthetic_raw)
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "bad_channels" in report
        assert "quality_metrics" in report
        assert "recommendations" in report
        assert "timestamp" in report

    def test_eegpt_abnormality_scoring(self, synthetic_raw, mock_eegpt_model):
        """Test EEGPT-based abnormality scoring."""
        controller = EEGQualityController()
        controller.eegpt_model = mock_eegpt_model
        
        # Score abnormality
        score = controller.compute_eegpt_abnormality_score(synthetic_raw)
        
        assert 0 <= score <= 1
        mock_eegpt_model.extract_features.assert_called()

    def test_detect_specific_artifacts(self, raw_with_artifacts):
        """Test detection of specific artifact types."""
        controller = EEGQualityController()
        
        artifacts = controller.detect_specific_artifacts(raw_with_artifacts)
        
        assert "eye_blinks" in artifacts
        assert "muscle" in artifacts
        assert "bad_channels" in artifacts
        assert "motion" in artifacts
        
        # Should detect artifacts we added
        assert len(artifacts["bad_channels"]) > 0

    def test_adaptive_thresholding(self, synthetic_raw):
        """Test adaptive threshold computation."""
        controller = EEGQualityController()
        
        thresholds = controller.compute_adaptive_thresholds(synthetic_raw)
        
        assert "amplitude" in thresholds
        assert "gradient" in thresholds
        assert "variance" in thresholds
        
        # Thresholds should be positive
        assert all(v > 0 for v in thresholds.values())

    def test_channel_wise_snr(self, synthetic_raw):
        """Test channel-wise SNR computation."""
        controller = EEGQualityController()
        
        snr = controller.compute_channel_snr(synthetic_raw)
        
        assert len(snr) == len(synthetic_raw.ch_names)
        assert all(s > 0 for s in snr.values())

    def test_impedance_estimation(self, synthetic_raw):
        """Test impedance estimation from signal characteristics."""
        controller = EEGQualityController()
        
        impedances = controller.estimate_impedances(synthetic_raw)
        
        assert len(impedances) == len(synthetic_raw.ch_names)
        # Impedances should be in reasonable range (kOhms)
        assert all(0 < z < 100 for z in impedances.values())

    def test_filter_optimization(self, synthetic_raw):
        """Test filter parameter optimization."""
        controller = EEGQualityController()
        
        # Get optimal filter params
        filter_params = controller.optimize_filter_params(synthetic_raw)
        
        assert "highpass" in filter_params
        assert "lowpass" in filter_params
        assert "notch" in filter_params
        
        # Check reasonable values
        assert 0.1 <= filter_params["highpass"] <= 2.0
        assert 30 <= filter_params["lowpass"] <= 100
        assert filter_params["notch"] in [50, 60, None]

    def test_quality_trend_analysis(self, synthetic_raw):
        """Test quality trend analysis over time."""
        controller = EEGQualityController()
        
        # Analyze quality trends
        trends = controller.analyze_quality_trends(
            synthetic_raw,
            window_size=5.0,  # 5-second windows
            overlap=0.5
        )
        
        assert "quality_over_time" in trends
        assert "degradation_points" in trends
        assert "improvement_points" in trends
        
        # Should have multiple time points
        assert len(trends["quality_over_time"]) > 1

    def test_reference_optimization(self, synthetic_raw):
        """Test reference electrode optimization."""
        controller = EEGQualityController()
        
        # Find optimal reference
        optimal_ref = controller.find_optimal_reference(synthetic_raw)
        
        assert optimal_ref in ["average", "linked_ears", "Cz", "nose"]
        
        # Apply optimal reference
        rereferenced = controller.apply_optimal_reference(
            synthetic_raw,
            reference_type=optimal_ref
        )
        
        assert rereferenced is not None

    def test_epoch_quality_scoring(self, synthetic_raw):
        """Test quality scoring for individual epochs."""
        controller = EEGQualityController()
        
        # Create epochs
        events = mne.make_fixed_length_events(synthetic_raw, duration=1.0)
        epochs = mne.Epochs(
            synthetic_raw, events, tmin=0, tmax=1.0,
            baseline=None, preload=True
        )
        
        # Score each epoch
        scores = controller.score_epoch_quality(epochs)
        
        assert len(scores) == len(epochs)
        assert all(0 <= s <= 1 for s in scores)

    def test_save_qc_results(self, synthetic_raw, tmp_path):
        """Test saving QC results to file."""
        controller = EEGQualityController()
        
        # Generate QC report
        report = controller.generate_qc_report(synthetic_raw)
        
        # Save results
        save_path = tmp_path / "qc_results.json"
        controller.save_qc_results(report, save_path)
        
        assert save_path.exists()
        
        # Load and verify
        loaded = controller.load_qc_results(save_path)
        assert loaded["summary"] == report["summary"]

    def test_batch_qc_processing(self, synthetic_raw):
        """Test batch processing of multiple EEG files."""
        controller = EEGQualityController()
        
        # Create multiple raw objects
        raw_list = [synthetic_raw.copy() for _ in range(3)]
        
        # Batch process
        results = controller.batch_process_qc(raw_list)
        
        assert len(results) == 3
        assert all("quality_score" in r for r in results)

    def test_qc_with_missing_channels(self, synthetic_raw):
        """Test QC with missing channels."""
        controller = EEGQualityController()
        
        # Drop some channels
        raw_subset = synthetic_raw.copy().pick_channels(
            synthetic_raw.ch_names[:10]
        )
        
        # Should still work with subset
        report = controller.generate_qc_report(raw_subset)
        
        assert report is not None
        assert len(report["bad_channels"]) <= 10

    def test_handle_low_sampling_rate(self):
        """Test handling of low sampling rate data."""
        import mne
        
        controller = EEGQualityController()
        
        # Create low sampling rate data (64 Hz)
        sfreq = 64
        data = np.random.randn(5, 640) * 20e-6  # 10 seconds
        info = mne.create_info(
            ch_names=["C3", "C4", "O1", "O2", "EOG"],
            sfreq=sfreq,
            ch_types=["eeg"] * 4 + ["eog"]
        )
        raw_low_sr = mne.io.RawArray(data, info)
        
        # Should handle low sampling rate appropriately
        filter_params = controller.optimize_filter_params(raw_low_sr)
        
        # Highpass should be lower for low sampling rate
        assert filter_params["highpass"] <= 0.5
        assert filter_params["lowpass"] <= 30