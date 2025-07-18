"""Mock EEGPT model for testing with realistic feature dimensions.

This module provides deterministic mock implementations of EEGPT
that return proper 768-dimensional embeddings for testing.
"""

from typing import Any

import numpy as np
import torch


class MockEEGPTModel:
    """Mock EEGPT model that returns deterministic 768-dim embeddings."""

    def __init__(self, seed: int = 42, device: str = "cpu"):
        """Initialize mock model with deterministic seed.

        Args:
            seed: Random seed for reproducible embeddings
            device: Device for tensors (cpu/cuda)
        """
        self.seed = seed
        self.device = device
        self.is_loaded = True
        self._call_count = 0

        # Mock config for compatibility
        from types import SimpleNamespace

        self.config = SimpleNamespace(model_path="fake/path.ckpt")

    def extract_features(self, window_tensor: torch.Tensor) -> torch.Tensor:
        """Extract 768-dimensional features from EEG window.

        Args:
            window_tensor: Input tensor of shape (batch, channels, samples)

        Returns:
            Feature tensor of shape (batch, 768)
        """
        batch_size = window_tensor.shape[0]

        # Use seed + call count for deterministic but varying outputs
        rng = np.random.RandomState(self.seed + self._call_count)
        self._call_count += 1

        # Generate realistic embeddings with structure
        # EEGPT embeddings typically have some structure, not pure random
        base_embedding = rng.randn(batch_size, 768).astype(np.float32)

        # Add some realistic structure:
        # - First 256 dims: lower variance (more stable features)
        # - Middle 256 dims: medium variance
        # - Last 256 dims: higher variance (more task-specific)
        base_embedding[:, :256] *= 0.5
        base_embedding[:, 256:512] *= 0.8
        base_embedding[:, 512:] *= 1.2

        # Add some correlation structure within feature groups
        for i in range(0, 768, 64):
            group_factor = rng.randn(batch_size, 1)
            base_embedding[:, i : i + 64] += group_factor * 0.3

        # Convert to tensor
        features = torch.from_numpy(base_embedding).float().to(self.device)

        return features


class MockAbnormalEEGPTModel(MockEEGPTModel):
    """Mock EEGPT model that produces embeddings for abnormal EEG."""

    def __init__(self, abnormality_strength: float = 0.8, **kwargs: Any) -> None:
        """Initialize mock for abnormal patterns.

        Args:
            abnormality_strength: How abnormal the embeddings should be (0-1)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        self.abnormality_strength = abnormality_strength

    def extract_features(self, window_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features that will produce high abnormality scores.

        The embeddings are shifted in a way that would make a trained
        classifier detect abnormality.
        """
        # Get base embeddings
        features = super().extract_features(window_tensor)

        # Shift certain dimensions that correlate with abnormality
        # In real EEGPT, certain embedding dimensions correlate with
        # epileptiform activity, slowing, etc.
        abnormal_dims = [10, 25, 67, 128, 256, 384, 512, 640]

        for dim in abnormal_dims:
            features[:, dim] += 2.0 * self.abnormality_strength

        # Add some high-frequency components (often abnormal)
        high_freq_dims = range(600, 768)
        features[:, high_freq_dims] *= 1 + self.abnormality_strength

        return features


class MockNormalEEGPTModel(MockEEGPTModel):
    """Mock EEGPT model that produces embeddings for normal EEG."""

    def __init__(self, normality_strength: float = 0.9, **kwargs: Any) -> None:
        """Initialize mock for normal patterns.

        Args:
            normality_strength: How normal the embeddings should be (0-1)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        self.normality_strength = normality_strength

    def extract_features(self, window_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features that will produce low abnormality scores.

        The embeddings represent typical healthy EEG patterns.
        """
        # Get base embeddings
        features = super().extract_features(window_tensor)

        # Reduce variance in dimensions associated with abnormality
        abnormal_dims = [10, 25, 67, 128, 256, 384, 512, 640]

        for dim in abnormal_dims:
            features[:, dim] *= 1 - self.normality_strength

        # Reduce high-frequency components
        high_freq_dims = range(600, 768)
        features[:, high_freq_dims] *= 0.5

        # Enhance alpha rhythm components (normal EEG)
        alpha_dims = range(100, 150)
        features[:, alpha_dims] *= 1.5

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
        List of 768-dimensional embedding arrays
    """
    rng = np.random.RandomState(seed)
    embeddings = []

    if abnormality_pattern is None:
        abnormality_pattern = [0.5] * num_windows

    for _i, abnorm_level in enumerate(abnormality_pattern):
        # Base embedding
        embedding = rng.randn(768).astype(np.float32)

        # Apply structure similar to MockEEGPTModel
        embedding[:256] *= 0.5
        embedding[256:512] *= 0.8
        embedding[512:] *= 1.2

        # Shift based on abnormality level
        if abnorm_level > 0.5:
            # Make more abnormal
            strength = (abnorm_level - 0.5) * 2
            abnormal_dims = [10, 25, 67, 128, 256, 384, 512, 640]
            for dim in abnormal_dims:
                embedding[dim] += 2.0 * strength
        else:
            # Make more normal
            strength = (0.5 - abnorm_level) * 2
            embedding[100:150] *= 1 + strength  # Enhance alpha
            embedding[600:] *= 1 - strength * 0.5  # Reduce high freq

        embeddings.append(embedding)

    return embeddings
