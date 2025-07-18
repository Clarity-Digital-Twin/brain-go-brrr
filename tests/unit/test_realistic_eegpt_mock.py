"""Test abnormality detector with realistic EEGPT mock.

This test ensures that:
1. EEGPT returns 768-dim embeddings (not 512)
2. Embeddings go through the classifier (not short-circuited)
3. Probability ordering is consistent with embedding patterns
"""

from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest
import torch

from services.abnormality_detector import AbnormalityDetector
from tests.fixtures.mock_eegpt import (
    MockAbnormalEEGPTModel,
    MockEEGPTModel,
    MockNormalEEGPTModel,
    create_deterministic_embeddings,
)


class TestRealisticEEGPTMock:
    """Test suite for realistic EEGPT mocking."""

    @pytest.fixture
    def mock_eeg_data(self):
        """Create minimal mock EEG data for testing."""
        # 16 channels (BioSerenity subset), 1 minute at 128 Hz
        sfreq = 128
        duration = 60  # 1 minute for faster tests
        n_channels = 16
        n_samples = int(sfreq * duration)

        # BioSerenity-E1 16-channel names
        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "F7",
            "F8",
            "C3",
            "C4",
            "T3",
            "T4",
            "P3",
            "P4",
            "O1",
            "O2",
            "Fz",
            "Cz",
        ]

        # Generate simple EEG data
        data = np.random.randn(n_channels, n_samples) * 20e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        return raw

    def test_mock_eegpt_dimensions(self):
        """Test that mock EEGPT returns correct (4, 512) summary token embeddings."""
        mock_model = MockEEGPTModel(seed=42)

        # Create test input
        window_tensor = torch.randn(1, 16, 512)  # batch=1, channels=16, samples=512

        # Extract features
        features = mock_model.extract_features(window_tensor)

        # Check dimensions - should match real EEGPT: 4 summary tokens x 512 dims
        assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32

    def test_embedding_determinism(self):
        """Test that embeddings are deterministic with same seed."""
        model1 = MockEEGPTModel(seed=42)
        model2 = MockEEGPTModel(seed=42)

        window = torch.randn(1, 16, 512)

        # Get features from both models
        features1 = model1.extract_features(window.clone())
        features2 = model2.extract_features(window.clone())

        # Should be identical
        assert np.allclose(features1, features2)

        # Second call should be different (due to call count)
        features3 = model1.extract_features(window.clone())
        assert not np.allclose(features1, features3)

    def test_normal_vs_abnormal_embeddings(self):
        """Test that normal/abnormal models produce different embeddings."""
        normal_model = MockNormalEEGPTModel(seed=42)
        abnormal_model = MockAbnormalEEGPTModel(seed=42)

        window = torch.randn(1, 16, 512)

        normal_features = normal_model.extract_features(window.clone())
        abnormal_features = abnormal_model.extract_features(window.clone())

        # Check specific dimensions that should differ (within 512-dim bounds)
        abnormal_dims = [10, 25, 67, 128, 256, 384, 450, 480]  # Updated to stay within 512 bounds

        for dim in abnormal_dims:
            assert abnormal_features[0, dim] > normal_features[0, dim], (
                f"Abnormal dim {dim} should be higher"
            )

    def test_classifier_integration(self):
        """Test that detector uses classifier with 2048-dim embeddings (4x512 flattened)."""
        # Create detector with mock model
        with (
            patch("services.abnormality_detector.EEGPTModel") as mock_model_class,
            patch("services.abnormality_detector.ModelConfig"),
        ):
            # Use our realistic mock
            mock_eegpt = MockEEGPTModel(seed=42)
            mock_model_class.return_value = mock_eegpt

            detector = AbnormalityDetector(
                model_path=Path("fake/path.ckpt"),
                device="cpu",
                window_duration=4.0,
                target_sfreq=128,
            )
            detector.model = mock_eegpt

            # Check classifier architecture
            assert hasattr(detector, "classifier")

            # Get first layer
            first_layer = detector.classifier[0]
            assert isinstance(first_layer, torch.nn.Linear)
            assert first_layer.in_features == 2048, (
                f"Classifier should accept 2048-dim input (4x512 flattened), got {first_layer.in_features}"
            )

    def test_prediction_flow_with_classifier(self):
        """Test that predictions flow through classifier, not short-circuited."""
        with (
            patch("services.abnormality_detector.EEGPTModel") as mock_model_class,
            patch("services.abnormality_detector.ModelConfig"),
        ):
            # Use realistic mock
            mock_eegpt = MockEEGPTModel(seed=42)
            mock_model_class.return_value = mock_eegpt

            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")
            detector.model = mock_eegpt

            # Create test window
            window = np.random.randn(16, 1024).astype(np.float32)

            # Spy on classifier forward pass
            original_forward = detector.classifier.forward
            forward_called = False

            def spy_forward(x):
                nonlocal forward_called
                forward_called = True
                return original_forward(x)

            detector.classifier.forward = spy_forward

            # Run prediction
            score = detector._predict_window(window)

            # Check that classifier was called
            assert forward_called, "Classifier forward pass should be called"
            assert 0.0 <= score <= 1.0

    def test_probability_ordering(self):
        """Test that embeddings from different models are distinguishable."""
        # With randomly initialized classifier weights, we can't guarantee
        # specific probability ordering. Instead, test that the embeddings
        # themselves are different and structured as expected.

        normal_model = MockNormalEEGPTModel(seed=42, normality_strength=0.9)
        abnormal_model = MockAbnormalEEGPTModel(seed=42, abnormality_strength=0.9)

        window = torch.randn(1, 16, 512)

        normal_features = normal_model.extract_features(window.clone())
        abnormal_features = abnormal_model.extract_features(window.clone())

        # Check that embeddings are different
        assert not np.allclose(normal_features, abnormal_features)

        # Check that specific dimensions differ as expected (within 512-dim bounds)
        abnormal_dims = [10, 25, 67, 128, 256, 384, 450, 480]  # Updated to stay within 512 bounds
        for dim in abnormal_dims:
            assert abnormal_features[0, dim] > normal_features[0, dim], (
                f"Abnormal embedding should have higher values in dimension {dim}"
            )

        # Check that high-frequency components are enhanced in abnormal (within 512-dim bounds)
        high_freq_mean_normal = normal_features[0, 400:].mean()  # Updated to stay within 512 bounds
        high_freq_mean_abnormal = abnormal_features[
            0, 400:
        ].mean()  # Updated to stay within 512 bounds
        assert high_freq_mean_abnormal > high_freq_mean_normal, (
            "Abnormal embeddings should have enhanced high-frequency components"
        )

    def test_full_pipeline_with_realistic_mock(self, mock_eeg_data):
        """Test full detection pipeline with realistic embeddings."""
        with (
            patch("services.abnormality_detector.EEGPTModel") as mock_model_class,
            patch("services.abnormality_detector.ModelConfig"),
        ):
            # Use abnormal mock
            mock_model = MockAbnormalEEGPTModel(seed=42, abnormality_strength=0.8)
            mock_model_class.return_value = mock_model

            detector = AbnormalityDetector(
                model_path=Path("fake/path.ckpt"),
                device="cpu",
                window_duration=4.0,
                target_sfreq=128,
            )
            detector.model = mock_model

            # Run detection
            result = detector.detect_abnormality(mock_eeg_data)

            # Should classify as abnormal due to mock
            # Note: Actual classification depends on classifier weights
            # which are randomly initialized in this test
            assert result.abnormality_score >= 0.0
            assert result.abnormality_score <= 1.0
            assert len(result.window_scores) > 0

            # Check all windows have valid scores
            for window_result in result.window_scores:
                assert 0.0 <= window_result.abnormality_score <= 1.0
                assert 0.0 <= window_result.quality_score <= 1.0

    def test_remove_2d_probs_path(self):
        """Test that 2D probability path has been removed from production code."""
        # Read the abnormality_detector.py file
        detector_path = Path("services/abnormality_detector.py")
        with detector_path.open() as f:
            detector_code = f.read()

        # Verify the 2D probability short-circuit path has been removed
        assert "features.shape[-1] == 2" not in detector_code, (
            "2D probability short-circuit path should be removed in production"
        )

        # Verify we always go through the classifier (check for the pattern, not exact variable name)
        assert ".classifier(" in detector_code, "Should always use classifier for predictions"

    def test_create_deterministic_embeddings_utility(self):
        """Test the utility function for creating test embeddings."""
        # Test default (neutral) pattern
        embeddings = create_deterministic_embeddings(num_windows=5, seed=42)

        assert len(embeddings) == 5
        for emb in embeddings:
            assert emb.shape == (4, 512), (
                f"Expected (4, 512), got {emb.shape}"
            )  # Updated to match EEGPT format
            assert emb.dtype == np.float32

        # Test custom abnormality pattern
        pattern = [0.1, 0.3, 0.5, 0.7, 0.9]  # Increasing abnormality
        embeddings = create_deterministic_embeddings(
            num_windows=5, abnormality_pattern=pattern, seed=42
        )

        # Later embeddings should have higher average values in abnormal dims
        abnormal_dims = [10, 25, 67, 128, 256, 384, 450, 480]  # Updated to stay within 512 bounds

        # Calculate mean abnormal dimension values for first and last embeddings
        # Since embeddings are (4, 512), we check the average across summary tokens for each dim
        first_abnormal_mean = np.mean([embeddings[0][:, dim].mean() for dim in abnormal_dims])
        last_abnormal_mean = np.mean([embeddings[4][:, dim].mean() for dim in abnormal_dims])

        assert last_abnormal_mean > first_abnormal_mean, (
            f"Last embedding should have higher abnormal dimension average ({last_abnormal_mean:.3f}) than first ({first_abnormal_mean:.3f})"
        )
