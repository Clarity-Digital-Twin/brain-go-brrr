"""Test classifier-embedding dimension compatibility.

This ensures the classifier head matches the EEGPT embedding dimensions
to prevent runtime failures from config/model mismatches.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from brain_go_brrr.core.abnormal import AbnormalityDetector
from brain_go_brrr.core.abnormality_config import AbnormalityConfig


class TestClassifierCompatibility:
    """Test suite for classifier dimension compatibility."""

    def test_classifier_matches_config_feature_dim(self) -> None:
        """Test that classifier input dimension matches config feature_dim."""
        with (
            patch("services.abnormality_detector.EEGPTModel"),
            patch("services.abnormality_detector.ModelConfig"),
        ):
            # Create detector with default config (768-dim)
            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")

            # Check classifier first layer matches config
            first_layer = detector.classifier[0]
            assert isinstance(first_layer, torch.nn.Linear)
            assert first_layer.in_features == detector.config.model.feature_dim, (
                f"Classifier expects {first_layer.in_features} features, "
                f"but config specifies {detector.config.model.feature_dim}"
            )

    def test_incompatible_classifier_raises_error(self) -> None:
        """Test that loading incompatible classifier weights raises clear error."""
        with (
            patch("services.abnormality_detector.EEGPTModel"),
            patch("services.abnormality_detector.ModelConfig"),
        ):
            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")

            # Create incompatible state dict (512-dim input instead of 768)
            incompatible_state = {
                "0.weight": torch.randn(256, 512),  # Wrong input dimension
                "0.bias": torch.randn(256),
                # ... other layers would be here
            }

            # Mock the torch.load to return incompatible weights
            with (
                patch("torch.load", return_value=incompatible_state),
                pytest.raises(RuntimeError, match="Classifier.*dimension.*mismatch"),
            ):
                detector._load_classifier_weights(Path("fake/classifier.pth"))

    def test_custom_feature_dim_propagates_to_classifier(self) -> None:
        """Test that custom feature dimensions are properly used."""
        custom_config = AbnormalityConfig()
        custom_config.model.feature_dim = 512  # Different from default 768

        with (
            patch("services.abnormality_detector.EEGPTModel"),
            patch("services.abnormality_detector.ModelConfig"),
        ):
            detector = AbnormalityDetector(
                model_path=Path("fake/path.ckpt"), device="cpu", config=custom_config
            )

            # Classifier should use custom dimension
            first_layer = detector.classifier[0]
            assert first_layer.in_features == 512, (
                f"Classifier should use custom feature_dim=512, got {first_layer.in_features}"
            )

    def test_validate_model_compatibility_method(self) -> None:
        """Test the validate_model_compatibility method."""
        with (
            patch("services.abnormality_detector.EEGPTModel") as mock_model_class,
            patch("services.abnormality_detector.ModelConfig"),
        ):
            # Mock EEGPT model with correct dimensions
            mock_model = MagicMock()
            mock_model.embedding_dim = 512  # Correct embedding dim from checkpoint
            mock_model.n_summary_tokens = 4  # 4 summary tokens
            mock_model_class.return_value = mock_model

            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")
            detector.model = mock_model

            # Should not raise when dimensions match (4 x 512 = 2048)
            detector.validate_model_compatibility()

            # Should raise when dimensions mismatch
            mock_model.embedding_dim = 256  # Wrong embedding dim: 4 x 256 = 1024 != 2048
            with pytest.raises(RuntimeError, match="dimension mismatch"):
                detector.validate_model_compatibility()

    def test_runtime_validation_on_detect(self) -> None:
        """Test that dimension validation happens during detection."""
        with (
            patch("services.abnormality_detector.EEGPTModel") as mock_model_class,
            patch("services.abnormality_detector.ModelConfig"),
        ):
            # Create mock with wrong dimensions
            mock_model = MagicMock()
            mock_model.embedding_dim = 512  # Wrong!
            mock_model.extract_features.return_value = torch.randn(1, 512)
            mock_model_class.return_value = mock_model

            detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"), device="cpu")
            detector.model = mock_model

            # Create mock EEG data
            import mne
            import numpy as np

            sfreq = 256
            ch_names = [
                "Fp1",
                "Fp2",
                "F3",
                "F4",
                "C3",
                "C4",
                "P3",
                "P4",
                "O1",
                "O2",
                "F7",
                "F8",
                "T3",
                "T4",
                "T5",
                "T6",
            ]
            data = np.random.randn(16, sfreq * 60) * 20e-6
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data, info)

            # Should catch dimension mismatch during detection
            with pytest.raises(RuntimeError, match="Model dimension mismatch"):
                detector.detect_abnormality(raw)
