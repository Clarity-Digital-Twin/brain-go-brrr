"""Mock EEGPT model for testing with realistic feature dimensions.

This module provides deterministic mock implementations of EEGPT
that return proper 512-dimensional embeddings for testing.
"""

from typing import Any

import numpy as np
import torch

# Abnormal dimensions to modify (key regions for classification)
ABNORMAL_DIMS = [10, 25, 67, 128, 256, 384, 450, 480]


class MockEEGPTModel:
    """Mock EEGPT model that provides realistic behavior for testing."""

    def __init__(self, seed: int = 42, device: str = "cpu"):
        """Initialize the mock EEGPT model.

        Args:
            seed: Random seed for reproducible results
            device: Device to use for computations
        """
        self.seed = seed
        self.device = device
        self.embedding_dim = 512
        self._call_count = 0

        # Add missing attributes expected by AbnormalityDetector
        self.is_loaded = True

        # Mock config for compatibility
        from types import SimpleNamespace

        self.config = SimpleNamespace(model_path="fake/path.ckpt")

    def extract_features(self, window_data, channel_names=None):
        """Extract features with simple normal vs abnormal distinction."""
        import numpy as np

        self._call_count += 1

        # Accept torch tensor or ndarray, handle batching
        if torch.is_tensor(window_data):
            data_np = window_data.detach().cpu().numpy()
        else:
            data_np = window_data

        if data_np.ndim == 3:  # (B, C, T) -> (C, T)
            data_np = data_np[0]

        # Create base embeddings
        rng = np.random.RandomState(self.seed + self._call_count)
        features = rng.randn(4, self.embedding_dim).astype(np.float32) * 0.1  # Small base variance

        # Simple amplitude-based pattern detection
        max_amplitude = np.max(np.abs(data_np))

        if max_amplitude > 8e-5:  # Abnormal pattern
            # Strong abnormal signature
            features[:, ABNORMAL_DIMS] += 4.0
            features[:, 400:] *= 3.0  # High frequency boost
        else:  # Normal pattern
            # Strong normal signature
            features[:, 100:150] *= 2.5  # Alpha rhythm boost
            features[:, 400:] *= 0.2  # Reduce high frequency

        return features

    def to(self, device):
        """Mock device transfer."""
        self.device = device
        return self


class MockNormalEEGPTModel(MockEEGPTModel):
    """Mock model biased toward normal classifications."""

    def extract_features(self, window_data, channel_names=None):
        """Always return normal-biased features."""
        features = super().extract_features(window_data, channel_names)
        # Force normal signature regardless of input
        features[:, ABNORMAL_DIMS] *= 0.1  # Reduce abnormal dims
        features[:, 100:150] *= 3.0  # Strong alpha
        features[:, 400:] *= 0.1  # Low high freq
        return features


class MockAbnormalEEGPTModel(MockEEGPTModel):
    """Mock model biased toward abnormal classifications."""

    def __init__(self, abnormality_strength: float = 0.8, seed: int = 42, device: str = "cpu"):
        """Initialize the mock abnormal EEGPT model.

        Args:
            abnormality_strength: Strength of abnormal signal injection
            seed: Random seed for reproducible results
            device: Device to use for computations
        """
        super().__init__(seed, device)
        self.abnormality_strength = abnormality_strength

    def extract_features(self, window_data, channel_names=None):
        """Always return abnormal-biased features."""
        features = super().extract_features(window_data, channel_names)
        # Force abnormal signature regardless of input
        features[:, ABNORMAL_DIMS] += 5.0 * self.abnormality_strength
        features[:, 400:] *= 4.0  # Very high freq
        features[:, 100:150] *= 0.2  # Reduce alpha
        return features


def create_deterministic_embeddings(
    num_windows: int, abnormality_pattern: list[float] | None = None, seed: int = 42
) -> list[np.ndarray]:
    """Create a sequence of deterministic embeddings for testing.

    Args:
        num_windows: Number of embedding vectors to create
        abnormality_pattern: Optional list of abnormality levels (0-1) per window
        seed: Random seed for reproducibility

    Returns:
        List of (4, 512) embedding arrays matching EEGPT format
    """
    rng = np.random.RandomState(seed)
    embeddings = []

    if abnormality_pattern is None:
        abnormality_pattern = [0.5] * num_windows

    for _i, abnorm_level in enumerate(abnormality_pattern):
        # Base embedding: (4, 512) to match EEGPT
        embedding = rng.randn(4, 512).astype(np.float32)

        # Apply structure similar to MockEEGPTModel
        embedding[:, :256] *= 0.5  # Lower variance for stable features
        embedding[:, 256:] *= 1.2  # Higher variance for task-specific

        # Shift based on abnormality level
        if abnorm_level > 0.5:
            # Make more abnormal
            strength = (abnorm_level - 0.5) * 2
            abnormal_dims = [10, 25, 67, 128, 256, 384, 450, 480]  # Within 512 bounds
            for dim in abnormal_dims:
                embedding[:, dim] += 2.0 * strength
        else:
            # Make more normal
            strength = (0.5 - abnorm_level) * 2
            embedding[:, 100:150] *= 1 + strength  # Enhance alpha
            embedding[:, 400:] *= 1 - strength * 0.5  # Reduce high freq

        embeddings.append(embedding)

    return embeddings


def create_mock_detector_with_realistic_model(seed: int = 42, normal_bias: bool = True) -> Any:
    """Create an AbnormalityDetector with realistic mock EEGPT model.

    Args:
        seed: Random seed for reproducible results
        normal_bias: If True, use MockNormalEEGPTModel that biases toward normal classifications

    Returns:
        AbnormalityDetector instance with mocked EEGPT model
    """
    from pathlib import Path
    from unittest.mock import patch

    from services.abnormality_detector import AbnormalityDetector

    # Create the appropriate mock model based on bias
    if normal_bias:
        mock_model = MockNormalEEGPTModel(seed=seed, device="cpu")
    else:
        mock_model = MockAbnormalEEGPTModel(seed=seed, device="cpu")

    mock_model.embedding_dim = 512

    # Mock the classifier with simple threshold-based logic
    def mock_classifier_forward(features):
        """Simple classifier based on key embedding regions."""
        batch_size = features.shape[0]
        logits = torch.zeros(batch_size, 2, device=features.device)

        for i in range(batch_size):
            feature_vec = features[i]

            # Check abnormal dimensions - debug what we're getting
            abnormal_strength = sum(feature_vec[dim].item() for dim in ABNORMAL_DIMS)

            # Much lower threshold since we're adding 4.0 to 8 dimensions = 32.0 expected
            if abnormal_strength > 10.0:  # Reduced from 20.0
                logits[i, 1] = 4.0  # High abnormal logit
                logits[i, 0] = -4.0
            else:  # Normal signature
                logits[i, 0] = 4.0  # High normal logit
                logits[i, 1] = -4.0

        return logits

    with (
        patch("services.abnormality_detector.EEGPTModel") as mock_model_class,
        patch("services.abnormality_detector.ModelConfig"),
    ):
        mock_model_class.return_value = mock_model

        # Create detector with required model_path
        detector = AbnormalityDetector(model_path=Path("fake/path.ckpt"))

        # Replace the model with our mock
        detector.model = mock_model

        # Mock the classifier forward method
        detector.classifier.forward = mock_classifier_forward

        return detector
