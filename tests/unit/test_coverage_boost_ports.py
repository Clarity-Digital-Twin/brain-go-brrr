"""Tests to boost coverage for domain ports and settings."""

from pathlib import Path
from typing import Any

import numpy as np

from brain_go_brrr.domain.abnormal.ports import (
    AbnormalityDetectorPort,
    EEGPTFeatureExtractorPort,
    LinearProbePort,
    PreprocessorPort,
)
from brain_go_brrr.domain.abnormal.settings import (
    AbnormalitySettings,
    ModelSettings,
    PreprocessingSettings,
    ThresholdSettings,
)


class TestAbnormalityPorts:
    """Test abnormality detector ports - currently at 0% coverage."""

    def test_abnormality_detector_port_interface(self):
        """Test the abnormality detector port interface."""

        class ConcreteDetector(AbnormalityDetectorPort):
            def detect(self, eeg_data: np.ndarray) -> dict[str, Any]:
                return {"is_abnormal": True, "confidence": 0.85, "features": np.zeros(2048)}

            def get_confidence(self, eeg_data: np.ndarray) -> float:
                return 0.85

            def get_features(self, eeg_data: np.ndarray) -> np.ndarray:
                return np.zeros(2048)

        detector = ConcreteDetector()

        # Test detection
        data = np.random.randn(19, 1024)
        result = detector.detect(data)
        assert result["is_abnormal"] is True
        assert result["confidence"] == 0.85
        assert result["features"].shape == (2048,)

        # Test individual methods
        assert detector.get_confidence(data) == 0.85
        assert detector.get_features(data).shape == (2048,)

    def test_eegpt_feature_extractor_port(self):
        """Test EEGPT feature extractor port interface."""

        class ConcreteExtractor(EEGPTFeatureExtractorPort):
            def extract_features(self, eeg_data: np.ndarray, summary: bool = True) -> np.ndarray:
                if summary:
                    return np.zeros(2048)  # Flattened 4x512
                else:
                    return np.zeros((4, 512))  # 4 summary tokens

            def get_embedding_dim(self) -> int:
                return 2048

            def requires_normalization(self) -> bool:
                return True

        extractor = ConcreteExtractor()
        data = np.random.randn(19, 1024)

        # Test summary features
        features = extractor.extract_features(data, summary=True)
        assert features.shape == (2048,)

        # Test non-summary features
        features = extractor.extract_features(data, summary=False)
        assert features.shape == (4, 512)

        # Test metadata methods
        assert extractor.get_embedding_dim() == 2048
        assert extractor.requires_normalization() is True

    def test_linear_probe_port(self):
        """Test linear probe port interface."""

        class ConcreteProbe(LinearProbePort):
            def __init__(self):
                self.input_dim = 2048
                self.output_dim = 1

            def predict(self, features: np.ndarray) -> np.ndarray:
                # Simulate binary classification
                return np.array([0.7])  # Probability of abnormal

            def predict_proba(self, features: np.ndarray) -> np.ndarray:
                return np.array([[0.3, 0.7]])  # [normal, abnormal]

            def get_input_dim(self) -> int:
                return self.input_dim

            def get_output_dim(self) -> int:
                return self.output_dim

        probe = ConcreteProbe()
        features = np.zeros(2048)

        # Test prediction
        pred = probe.predict(features)
        assert pred.shape == (1,)
        assert 0 <= pred[0] <= 1

        # Test probability prediction
        proba = probe.predict_proba(features)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(), 1.0)

        # Test dimensions
        assert probe.get_input_dim() == 2048
        assert probe.get_output_dim() == 1

    def test_preprocessor_port(self):
        """Test preprocessor port interface."""

        class ConcretePreprocessor(PreprocessorPort):
            def preprocess(self, eeg_data: np.ndarray, sampling_rate: int = 256) -> np.ndarray:
                # Simulate preprocessing
                return eeg_data * 1e-6  # Scale to microvolts

            def apply_filters(
                self, eeg_data: np.ndarray, low_freq: float = 0.5, high_freq: float = 50.0
            ) -> np.ndarray:
                # Simulate filtering
                return eeg_data

            def remove_artifacts(self, eeg_data: np.ndarray) -> np.ndarray:
                # Simulate artifact removal
                return eeg_data

            def normalize(self, eeg_data: np.ndarray) -> np.ndarray:
                # Z-score normalization
                mean = np.mean(eeg_data, axis=1, keepdims=True)
                std = np.std(eeg_data, axis=1, keepdims=True)
                return (eeg_data - mean) / (std + 1e-8)

        preprocessor = ConcretePreprocessor()
        data = np.random.randn(19, 1024) * 100  # Simulate raw EEG in nV

        # Test preprocessing
        processed = preprocessor.preprocess(data)
        assert processed.shape == data.shape
        assert np.abs(processed).max() < np.abs(data).max()  # Scaled down

        # Test filtering
        filtered = preprocessor.apply_filters(processed)
        assert filtered.shape == data.shape

        # Test artifact removal
        clean = preprocessor.remove_artifacts(filtered)
        assert clean.shape == data.shape

        # Test normalization
        normalized = preprocessor.normalize(clean)
        assert normalized.shape == data.shape
        # Check normalization worked (approximately zero mean, unit variance)
        assert np.abs(normalized.mean()) < 0.1
        assert 0.8 < normalized.std() < 1.2


class TestAbnormalitySettings:
    """Test abnormality detection settings - currently at 0% coverage."""

    def test_abnormality_settings_defaults(self):
        """Test default abnormality settings."""
        settings = AbnormalitySettings()

        assert settings.confidence_threshold == 0.5
        assert settings.use_autoreject is True
        assert settings.max_bad_channels == 5
        assert settings.min_channels == 19
        assert settings.window_size == 4.0
        assert settings.window_stride == 2.0

    def test_abnormality_settings_custom(self):
        """Test custom abnormality settings."""
        settings = AbnormalitySettings(
            confidence_threshold=0.7,
            use_autoreject=False,
            max_bad_channels=3,
            min_channels=16,
            window_size=2.0,
            window_stride=1.0,
        )

        assert settings.confidence_threshold == 0.7
        assert settings.use_autoreject is False
        assert settings.max_bad_channels == 3
        assert settings.min_channels == 16
        assert settings.window_size == 2.0
        assert settings.window_stride == 1.0

    def test_preprocessing_settings(self):
        """Test preprocessing settings."""
        settings = PreprocessingSettings()

        # Check defaults
        assert settings.sampling_rate == 256
        assert settings.low_freq == 0.5
        assert settings.high_freq == 50.0
        assert settings.notch_freq == 60.0
        assert settings.apply_autoreject is True
        assert settings.normalize is True

        # Test custom settings
        custom = PreprocessingSettings(
            sampling_rate=512,
            low_freq=1.0,
            high_freq=40.0,
            notch_freq=50.0,  # European power line
            apply_autoreject=False,
            normalize=False,
        )

        assert custom.sampling_rate == 512
        assert custom.low_freq == 1.0
        assert custom.high_freq == 40.0
        assert custom.notch_freq == 50.0
        assert custom.apply_autoreject is False
        assert custom.normalize is False

    def test_model_settings(self):
        """Test model settings."""
        settings = ModelSettings()

        # Check defaults
        assert settings.model_type == "eegpt"
        assert settings.checkpoint_path is None
        assert settings.device == "cpu"
        assert settings.batch_size == 32
        assert settings.use_amp is False

        # Test custom settings
        custom = ModelSettings(
            model_type="custom",
            checkpoint_path=Path("/models/custom.ckpt"),
            device="cuda",
            batch_size=64,
            use_amp=True,
        )

        assert custom.model_type == "custom"
        assert custom.checkpoint_path == Path("/models/custom.ckpt")
        assert custom.device == "cuda"
        assert custom.batch_size == 64
        assert custom.use_amp is True

    def test_threshold_settings(self):
        """Test threshold settings for abnormality detection."""
        settings = ThresholdSettings()

        # Check defaults
        assert settings.normal_threshold == 0.5
        assert settings.abnormal_threshold == 0.5
        assert settings.confidence_min == 0.0
        assert settings.confidence_max == 1.0

        # Test custom thresholds
        custom = ThresholdSettings(
            normal_threshold=0.3, abnormal_threshold=0.7, confidence_min=0.2, confidence_max=0.9
        )

        assert custom.normal_threshold == 0.3
        assert custom.abnormal_threshold == 0.7
        assert custom.confidence_min == 0.2
        assert custom.confidence_max == 0.9

    def test_settings_validation(self):
        """Test settings validation."""
        # Valid settings should work
        valid = AbnormalitySettings(confidence_threshold=0.6, max_bad_channels=4, min_channels=18)
        assert valid.confidence_threshold == 0.6

        # Test boundary conditions
        boundary = PreprocessingSettings(sampling_rate=256, low_freq=0.1, high_freq=100.0)
        assert boundary.low_freq == 0.1
        assert boundary.high_freq == 100.0

        # Test that settings can be converted to dict
        settings = AbnormalitySettings()
        settings_dict = settings.to_dict()
        assert isinstance(settings_dict, dict)
        assert "confidence_threshold" in settings_dict
        assert "use_autoreject" in settings_dict

    def test_settings_from_dict(self):
        """Test creating settings from dictionary."""
        config = {
            "confidence_threshold": 0.8,
            "use_autoreject": False,
            "max_bad_channels": 2,
            "min_channels": 20,
            "window_size": 3.0,
            "window_stride": 1.5,
        }

        settings = AbnormalitySettings.from_dict(config)
        assert settings.confidence_threshold == 0.8
        assert settings.use_autoreject is False
        assert settings.max_bad_channels == 2
        assert settings.min_channels == 20
        assert settings.window_size == 3.0
        assert settings.window_stride == 1.5

    def test_settings_update(self):
        """Test updating settings."""
        settings = AbnormalitySettings()

        # Update specific fields
        settings.update(confidence_threshold=0.9, max_bad_channels=1)

        assert settings.confidence_threshold == 0.9
        assert settings.max_bad_channels == 1
        # Other fields should remain unchanged
        assert settings.use_autoreject is True
        assert settings.min_channels == 19

    def test_settings_copy(self):
        """Test copying settings."""
        original = AbnormalitySettings(confidence_threshold=0.75, use_autoreject=False)

        # Create a copy
        copy = original.copy()

        # Verify copy has same values
        assert copy.confidence_threshold == original.confidence_threshold
        assert copy.use_autoreject == original.use_autoreject

        # Modify copy shouldn't affect original
        copy.confidence_threshold = 0.9
        assert original.confidence_threshold == 0.75
        assert copy.confidence_threshold == 0.9
