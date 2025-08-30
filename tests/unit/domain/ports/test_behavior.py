"""Behavioral tests for domain ports - Testing PROTOCOL CONTRACTS."""

from typing import Any

import numpy as np
import numpy.typing as npt

from brain_go_brrr.domain.abnormal.ports import (
    AbnormalityHeadPort,
    EEGPreprocessorPort,
    FeatureExtractorPort,
    LoggerPort,
    MneRaw,
)


class ConcretePreprocessor:
    """Concrete preprocessor that implements the protocol."""

    def transform(self, raw: MneRaw) -> npt.NDArray[np.float32]:
        """Transform raw EEG to preprocessed array."""
        data = raw.get_data()
        # Simple normalization
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
        return normalized.astype(np.float32)


class ConcreteAbnormalityHead:
    """Concrete abnormality head that implements the protocol."""

    def __init__(self, threshold: float = 0.5):
        """Initialize with abnormality threshold."""
        self.threshold = threshold

    def predict_proba(self, X: npt.NDArray[np.float32]) -> float:  # noqa: N803
        """Predict abnormality probability based on simple heuristic."""
        # Simple heuristic: high variance = abnormal
        variance = np.var(X)
        # Sigmoid-like transformation
        prob = 1.0 / (1.0 + np.exp(-variance + self.threshold))
        return float(prob)


class ConcreteFeatureExtractor:
    """Concrete feature extractor that implements the protocol."""

    def __init__(self, n_features: int = 10):
        """Initialize with number of features to extract."""
        self.n_features = n_features

    def extract(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Extract statistical features from EEG."""
        features = []

        # Extract simple statistical features
        features.append(np.mean(X))
        features.append(np.std(X))
        features.append(np.var(X))
        features.append(np.min(X))
        features.append(np.max(X))
        features.append(np.median(X))

        # Pad with zeros if needed
        while len(features) < self.n_features:
            features.append(0.0)

        return np.array(features[: self.n_features], dtype=np.float32)


class ConcreteLogger:
    """Concrete logger that implements the protocol."""

    def __init__(self):
        """Initialize the logger with empty message list."""
        self.messages = []

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.messages.append(("DEBUG", msg, args, kwargs))

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.messages.append(("INFO", msg, args, kwargs))

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.messages.append(("WARNING", msg, args, kwargs))

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.messages.append(("ERROR", msg, args, kwargs))

    def clear(self):
        """Clear logged messages for testing."""
        self.messages = []


class MockMneRaw:
    """Mock MNE Raw object that implements the protocol."""

    def __init__(self, n_channels: int = 19, n_samples: int = 1024):
        """Initialize mock MNE Raw with random EEG data."""
        self._data = np.random.randn(n_channels, n_samples) * 50e-6  # Realistic EEG scale
        self._ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
        self._sfreq = 256.0

    @property
    def info(self) -> dict[str, Any]:
        return {"sfreq": self._sfreq, "ch_names": self._ch_names}

    @property
    def ch_names(self) -> list[str]:
        return self._ch_names

    @property
    def n_times(self) -> int:
        return self._data.shape[1]

    def get_data(self) -> npt.NDArray[np.float64]:
        return self._data

    def copy(self) -> "MockMneRaw":
        new = MockMneRaw(0, 0)
        new._data = self._data.copy()
        new._ch_names = self._ch_names.copy()
        new._sfreq = self._sfreq
        return new


class TestEEGPreprocessorPort:
    """Test EEGPreprocessor protocol behavior."""

    def test_preprocessor_implements_protocol(self):
        """Test that concrete class implements the protocol."""
        preprocessor = ConcretePreprocessor()
        assert isinstance(preprocessor, EEGPreprocessorPort)

    def test_transform_normalizes_data(self):
        """Test transform properly normalizes EEG data."""
        preprocessor = ConcretePreprocessor()
        raw = MockMneRaw(n_channels=19, n_samples=512)

        # Transform data
        result = preprocessor.transform(raw)

        # Check shape preserved
        assert result.shape == (19, 512)
        assert result.dtype == np.float32

        # Check normalization (should be roughly zero mean, unit variance)
        assert abs(np.mean(result)) < 0.1  # Close to zero
        assert 0.8 < np.std(result) < 1.2  # Close to 1

    def test_transform_handles_different_sizes(self):
        """Test transform handles various input sizes."""
        preprocessor = ConcretePreprocessor()

        # Small data
        raw_small = MockMneRaw(n_channels=5, n_samples=100)
        result_small = preprocessor.transform(raw_small)
        assert result_small.shape == (5, 100)

        # Large data
        raw_large = MockMneRaw(n_channels=64, n_samples=10000)
        result_large = preprocessor.transform(raw_large)
        assert result_large.shape == (64, 10000)


class TestAbnormalityHeadPort:
    """Test AbnormalityHead protocol behavior."""

    def test_head_implements_protocol(self):
        """Test that concrete class implements the protocol."""
        head = ConcreteAbnormalityHead()
        assert isinstance(head, AbnormalityHeadPort)

    def test_predict_proba_returns_probability(self):
        """Test predict_proba returns valid probability."""
        head = ConcreteAbnormalityHead(threshold=0.5)

        # Normal-like data (low variance)
        normal_data = np.ones((19, 1024), dtype=np.float32) * 0.1
        normal_prob = head.predict_proba(normal_data)
        assert 0 <= normal_prob <= 1
        assert normal_prob < 0.5  # Should be low probability

        # Abnormal-like data (high variance)
        abnormal_data = np.random.randn(19, 1024).astype(np.float32) * 10
        abnormal_prob = head.predict_proba(abnormal_data)
        assert 0 <= abnormal_prob <= 1
        assert abnormal_prob > 0.5  # Should be high probability

    def test_threshold_affects_predictions(self):
        """Test that threshold parameter affects predictions."""
        data = np.random.randn(19, 1024).astype(np.float32)

        head_low = ConcreteAbnormalityHead(threshold=0.1)
        head_high = ConcreteAbnormalityHead(threshold=1.0)

        prob_low = head_low.predict_proba(data)
        prob_high = head_high.predict_proba(data)

        # Higher threshold should give lower probability
        assert prob_high < prob_low


class TestFeatureExtractorPort:
    """Test FeatureExtractor protocol behavior."""

    def test_extractor_implements_protocol(self):
        """Test that concrete class implements the protocol."""
        extractor = ConcreteFeatureExtractor()
        assert isinstance(extractor, FeatureExtractorPort)

    def test_extract_returns_features(self):
        """Test extract returns feature vector."""
        extractor = ConcreteFeatureExtractor(n_features=10)
        data = np.random.randn(19, 1024).astype(np.float32)

        features = extractor.extract(data)

        assert features.shape == (10,)
        assert features.dtype == np.float32

        # Check some features are non-zero
        assert not np.allclose(features, 0)

    def test_extract_consistent_features(self):
        """Test extract gives consistent features for same input."""
        extractor = ConcreteFeatureExtractor(n_features=5)
        data = np.ones((19, 1024), dtype=np.float32) * 2.0

        features1 = extractor.extract(data)
        features2 = extractor.extract(data)

        # Should be identical for same input
        assert np.allclose(features1, features2)

        # Check expected values for constant input
        assert np.isclose(features1[0], 2.0)  # mean
        assert np.isclose(features1[1], 0.0)  # std
        assert np.isclose(features1[2], 0.0)  # var


class TestLoggerPort:
    """Test Logger protocol behavior."""

    def test_logger_implements_protocol(self):
        """Test that concrete class implements the protocol."""
        logger = ConcreteLogger()
        assert isinstance(logger, LoggerPort)

    def test_logger_methods_capture_messages(self):
        """Test all logger methods capture messages correctly."""
        logger = ConcreteLogger()

        logger.debug("Debug message", "arg1", key="value")
        logger.info("Info message")
        logger.warning("Warning message", extra={"user": "test"})
        logger.error("Error message", exc_info=True)

        assert len(logger.messages) == 4

        # Check debug
        assert logger.messages[0][0] == "DEBUG"
        assert logger.messages[0][1] == "Debug message"
        assert logger.messages[0][2] == ("arg1",)
        assert logger.messages[0][3] == {"key": "value"}

        # Check info
        assert logger.messages[1][0] == "INFO"
        assert logger.messages[1][1] == "Info message"

        # Check warning
        assert logger.messages[2][0] == "WARNING"
        assert logger.messages[2][3] == {"extra": {"user": "test"}}

        # Check error
        assert logger.messages[3][0] == "ERROR"
        assert logger.messages[3][3] == {"exc_info": True}

    def test_logger_clear_functionality(self):
        """Test logger can be cleared for testing."""
        logger = ConcreteLogger()

        logger.info("Message 1")
        logger.info("Message 2")
        assert len(logger.messages) == 2

        logger.clear()
        assert len(logger.messages) == 0

        logger.info("Message 3")
        assert len(logger.messages) == 1


class TestMneRawProtocol:
    """Test MneRaw protocol behavior."""

    def test_mock_implements_protocol(self):
        """Test that mock implements the protocol."""
        raw = MockMneRaw()
        # MneRaw is not @runtime_checkable, so we check methods instead
        assert hasattr(raw, 'info')
        assert hasattr(raw, 'ch_names')
        assert hasattr(raw, 'n_times')
        assert hasattr(raw, 'get_data')
        assert hasattr(raw, 'copy')

    def test_raw_properties(self):
        """Test MneRaw protocol properties."""
        raw = MockMneRaw(n_channels=21, n_samples=2048)

        # Test info property
        info = raw.info
        assert info["sfreq"] == 256.0
        assert len(info["ch_names"]) == 21

        # Test ch_names property
        assert len(raw.ch_names) == 21
        assert all(name.startswith("EEG") for name in raw.ch_names)

        # Test n_times property
        assert raw.n_times == 2048

    def test_raw_get_data(self):
        """Test get_data returns correct array."""
        raw = MockMneRaw(n_channels=19, n_samples=1024)
        data = raw.get_data()

        assert data.shape == (19, 1024)
        assert data.dtype == np.float64

        # Check realistic EEG scale (microvolts)
        assert np.abs(data).max() < 1e-3  # Less than 1 mV

    def test_raw_copy(self):
        """Test copy creates independent copy."""
        raw1 = MockMneRaw(n_channels=5, n_samples=100)
        raw2 = raw1.copy()

        # Modify original
        raw1._data[0, 0] = 999.0

        # Copy should be unchanged
        assert raw2.get_data()[0, 0] != 999.0

        # But should have same shape
        assert raw2.get_data().shape == raw1.get_data().shape


class TestProtocolComposition:
    """Test composing multiple protocols together."""

    def test_pipeline_composition(self):
        """Test complete pipeline using protocols."""
        # Create components
        preprocessor = ConcretePreprocessor()
        extractor = ConcreteFeatureExtractor(n_features=15)
        head = ConcreteAbnormalityHead(threshold=0.3)
        logger = ConcreteLogger()

        # Create data
        raw = MockMneRaw(n_channels=19, n_samples=1024)

        # Run pipeline
        logger.info("Starting pipeline")

        # Step 1: Preprocess
        preprocessed = preprocessor.transform(raw)
        logger.debug(f"Preprocessed shape: {preprocessed.shape}")

        # Step 2: Extract features
        features = extractor.extract(preprocessed)
        logger.debug(f"Extracted {len(features)} features")

        # Step 3: Predict abnormality
        probability = head.predict_proba(features.reshape(1, -1))
        logger.info(f"Abnormality probability: {probability:.3f}")

        # Verify results
        assert 0 <= probability <= 1
        assert len(logger.messages) == 4  # 2 info, 2 debug

        # Check pipeline preserves data flow
        assert preprocessed.shape == (19, 1024)
        assert features.shape == (15,)
        assert isinstance(probability, float)
