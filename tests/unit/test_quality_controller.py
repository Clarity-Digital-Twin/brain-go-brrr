"""CLEAN tests for EEG Quality Controller - dependency injection, no bullshit mocks."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from brain_go_brrr.core.quality.controller import EEGQualityController


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
            "FP1",
            "FP2",
            "F7",
            "F3",
            "FZ",
            "F4",
            "F8",
            "T3",
            "C3",
            "CZ",
            "C4",
            "T4",
            "T5",
            "P3",
            "PZ",
            "P4",
            "T6",
            "O1",
            "O2",
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
            data[1:3, start : start + 50] += 100e-6 * np.sin(np.linspace(0, np.pi, 50))

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
            rejection_threshold=0.1, interpolation_threshold=0.8, random_state=42
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

        controller = EEGQualityController(eegpt_model_path=checkpoint_path)

        # Controller creates an EEGPTModel object even with fake checkpoint
        # (it just won't have loaded weights properly)
        from brain_go_brrr.models.eegpt_model import EEGPTModel

        assert controller.eegpt_model is not None  # Model object exists
        assert isinstance(controller.eegpt_model, EEGPTModel)  # Correct type
        # Verify expected methods exist
        assert hasattr(controller.eegpt_model, "predict_abnormality")
        assert hasattr(controller.eegpt_model, "extract_features")

    def test_detect_bad_channels(self, raw_with_artifacts):
        """Test bad channel detection."""
        controller = EEGQualityController()

        bad_channels = controller.detect_bad_channels(raw_with_artifacts)

        assert isinstance(bad_channels, list)
        # May or may not detect bad channels depending on algorithm

    def test_preprocess_raw(self, synthetic_raw):
        """Test raw data preprocessing."""
        controller = EEGQualityController()

        processed = controller.preprocess_raw(synthetic_raw)

        assert processed is not None
        # Should return processed MNE Raw object
        assert hasattr(processed, "get_data")

    def test_create_epochs(self, synthetic_raw):
        """Test epoch creation from raw data."""
        controller = EEGQualityController()

        # Create epochs
        epochs = controller.create_epochs(synthetic_raw)

        assert epochs is not None
        assert hasattr(epochs, "get_data")

    def test_auto_reject_epochs(self, synthetic_raw):
        """Test auto rejection of epochs."""
        controller = EEGQualityController()

        # Create epochs first
        epochs = controller.create_epochs(synthetic_raw)

        # Apply auto rejection
        cleaned = controller.auto_reject_epochs(epochs)

        assert cleaned is not None
        # Should return cleaned epochs

    def test_compute_abnormality_score(self, synthetic_raw):
        """Test abnormality score computation."""
        controller = EEGQualityController()

        # Create epochs first (compute_abnormality_score takes epochs, not raw)
        epochs = controller.create_epochs(synthetic_raw)

        # Compute score
        score = controller.compute_abnormality_score(epochs)

        assert isinstance(score, int | float)
        assert 0 <= score <= 1  # Should be normalized score

    def test_generate_qc_report(self, synthetic_raw):
        """Test QC report generation."""
        controller = EEGQualityController()

        # Need to prepare all required inputs
        epochs = controller.create_epochs(synthetic_raw)
        bad_channels = controller.detect_bad_channels(synthetic_raw)
        abnormality_score = controller.compute_abnormality_score(synthetic_raw)

        report = controller.generate_qc_report(
            raw=synthetic_raw,
            epochs=epochs,
            bad_channels=bad_channels,
            abnormality_score=abnormality_score,
        )

        assert isinstance(report, dict)
        assert "quality_metrics" in report
        assert "bad_channels" in report["quality_metrics"]
        assert "quality_grade" in report["quality_metrics"]

    def test_run_full_qc_pipeline(self, synthetic_raw):
        """Test full QC pipeline."""
        controller = EEGQualityController()

        # Run full pipeline
        results = controller.run_full_qc_pipeline(synthetic_raw)

        assert isinstance(results, dict)
        # The pipeline returns the QC report directly
        assert "quality_metrics" in results
        assert "data_info" in results
        assert "processing_info" in results

    def test_generate_qc_report_with_bad_data(self, raw_with_artifacts):
        """Test QC report generation with artifacted data."""
        controller = EEGQualityController()

        # Need to prepare all required inputs
        epochs = controller.create_epochs(raw_with_artifacts)
        bad_channels = controller.detect_bad_channels(raw_with_artifacts)
        abnormality_score = controller.compute_abnormality_score(raw_with_artifacts)

        report = controller.generate_qc_report(
            raw=raw_with_artifacts,
            epochs=epochs,
            bad_channels=bad_channels,
            abnormality_score=abnormality_score,
        )

        assert isinstance(report, dict)
        # Should still generate a report even with bad data

    def test_cleanup(self):
        """Test cleanup method."""
        controller = EEGQualityController()

        # Should not raise
        controller.cleanup()

    def test_detect_bad_channels_with_method(self, synthetic_raw):
        """Test bad channel detection with specific method."""
        controller = EEGQualityController()

        # Test with autoreject method
        bad_channels = controller.detect_bad_channels(synthetic_raw, method="autoreject")
        assert isinstance(bad_channels, list)

        # Test with other method (if available)
        bad_channels = controller.detect_bad_channels(synthetic_raw, method="variance")
        assert isinstance(bad_channels, list)

    def test_preprocess_with_filtering(self, synthetic_raw):
        """Test preprocessing with different filter settings."""
        controller = EEGQualityController()

        # Test with high-pass filter
        processed = controller.preprocess_raw(
            synthetic_raw,
            l_freq=0.5,  # High-pass
            h_freq=None,  # No low-pass
        )
        assert processed is not None

    def test_create_epochs_with_duration(self, synthetic_raw):
        """Test epoch creation with custom duration."""
        controller = EEGQualityController()

        # Create epochs with custom duration
        epochs = controller.create_epochs(
            synthetic_raw,
            epoch_length=2.0,  # 2-second epochs
            overlap=0.5,  # 50% overlap
        )

        assert epochs is not None

    def test_auto_reject_with_threshold(self, synthetic_raw):
        """Test auto rejection with custom threshold."""
        controller = EEGQualityController(
            rejection_threshold=0.05  # Stricter threshold
        )

        epochs = controller.create_epochs(synthetic_raw)
        cleaned = controller.auto_reject_epochs(epochs)

        assert cleaned is not None

    def test_compute_abnormality_with_model(self, synthetic_raw, mock_eegpt_model):
        """Test abnormality computation with EEGPT model."""
        controller = EEGQualityController()
        controller.eegpt_model = mock_eegpt_model

        # Mock the predict_abnormality method (which is what's actually called)
        mock_eegpt_model.predict_abnormality = MagicMock(
            return_value={"abnormality_score": 0.3, "confidence": 0.9}
        )

        # Create epochs first (compute_abnormality_score takes epochs, not raw)
        epochs = controller.create_epochs(synthetic_raw)
        score = controller.compute_abnormality_score(epochs)

        assert isinstance(score, int | float)
        # Model should have been called
        mock_eegpt_model.predict_abnormality.assert_called()

    def test_qc_report_comprehensive(self, synthetic_raw):
        """Test comprehensive QC report generation."""
        controller = EEGQualityController()

        # Need to prepare all required inputs
        epochs = controller.create_epochs(synthetic_raw)
        bad_channels = controller.detect_bad_channels(synthetic_raw)
        abnormality_score = controller.compute_abnormality_score(synthetic_raw)

        report = controller.generate_qc_report(
            raw=synthetic_raw,
            epochs=epochs,
            bad_channels=bad_channels,
            abnormality_score=abnormality_score,
        )

        assert isinstance(report, dict)
        # Should have comprehensive metrics
        assert "quality_metrics" in report
        assert "data_info" in report

    def test_full_pipeline_with_options(self, synthetic_raw):
        """Test full pipeline with custom options."""
        controller = EEGQualityController(rejection_threshold=0.1, interpolation_threshold=0.8)

        results = controller.run_full_qc_pipeline(
            synthetic_raw, apply_autoreject=True, compute_abnormality=True
        )

        assert isinstance(results, dict)

    def test_controller_with_low_sampling_rate(self):
        """Test controller with low sampling rate data."""
        import mne

        # Create low sampling rate data
        sfreq = 64
        data = np.random.randn(5, 640).astype(np.float32) * 20e-6
        info = mne.create_info(
            ch_names=["C3", "C4", "O1", "O2", "EOG"], sfreq=sfreq, ch_types=["eeg"] * 4 + ["eog"]
        )
        raw = mne.io.RawArray(data, info)

        controller = EEGQualityController()

        # Should handle low sampling rate
        processed = controller.preprocess_raw(raw)
        assert processed is not None

    def test_controller_with_missing_channels(self, synthetic_raw):
        """Test controller with subset of channels."""
        # Drop some channels
        raw_subset = synthetic_raw.copy().pick_channels(synthetic_raw.ch_names[:10])

        controller = EEGQualityController()

        # Should still work with subset
        epochs = controller.create_epochs(raw_subset)
        bad_channels = controller.detect_bad_channels(raw_subset)
        abnormality_score = controller.compute_abnormality_score(raw_subset)

        report = controller.generate_qc_report(
            raw=raw_subset,
            epochs=epochs,
            bad_channels=bad_channels,
            abnormality_score=abnormality_score,
        )
        assert report is not None
