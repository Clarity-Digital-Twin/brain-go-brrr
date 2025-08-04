"""EEGPT model wrapper with proper preprocessing.

The pretrained EEGPT model expects normalized input data. Raw EEG signals
(typically 50 microvolts) are too small compared to the model's bias terms,
causing all outputs to be identical. This wrapper handles the necessary
preprocessing.
"""

import logging
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from .eegpt_architecture import create_eegpt_model

logger = logging.getLogger(__name__)


class EEGPTWrapper(nn.Module):
    """EEGPT model with proper input preprocessing."""

    def __init__(self, checkpoint_path: str | None = None, normalization_path: str | None = None):
        """Initialize EEGPT with preprocessing.

        Args:
            checkpoint_path: Path to pretrained checkpoint
            normalization_path: Path to normalization stats JSON
        """
        super().__init__()
        self.model = create_eegpt_model(checkpoint_path)

        # Load normalization parameters from file if available
        if normalization_path and Path(normalization_path).exists():
            import json

            with Path(normalization_path).open() as f:
                stats = json.load(f)
            self.register_buffer("input_mean", torch.tensor(stats["mean"]))
            self.register_buffer("input_std", torch.tensor(stats["std"]))
            self.normalize = True
            self._stats_source = "file"
        else:
            # Default normalization parameters - TUAB is already normalized!
            self.register_buffer("input_mean", torch.zeros(1))
            self.register_buffer("input_std", torch.ones(1))
            self.normalize = True
            self._stats_source = "default"
            logger.warning("No normalization file found - using identity normalization (mean=0, std=1)")

    def set_normalization_params(self, mean: float, std: float) -> None:
        """Set normalization parameters.

        Args:
            mean: Mean value for normalization
            std: Standard deviation for normalization
        """
        self.input_mean = torch.tensor(mean)
        self.input_std = torch.tensor(std)

    def estimate_normalization_params(self, data: torch.Tensor) -> None:
        """Estimate normalization parameters from data.

        Args:
            data: Input tensor of shape (B, C, T) or (C, T)
        """
        if data.dim() == 2:
            data = data.unsqueeze(0)

        # Estimate per-channel statistics
        channel_means = data.mean(dim=(0, 2))  # Mean across batch and time
        channel_stds = data.std(dim=(0, 2))  # Std across batch and time

        # Use global statistics
        self.input_mean = channel_means.mean()
        self.input_std = channel_stds.mean()

        logger.info(
            f"Estimated normalization: mean={self.input_mean.item():.6f}, "
            f"std={self.input_std.item():.6f}"
        )

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with preprocessing.

        Args:
            x: Input tensor of shape (B, C, T)
            chan_ids: Channel IDs for positional embedding

        Returns:
            Summary tokens of shape (B, embed_num, embed_dim)
        """
        # Normalize input if enabled
        if self.normalize:
            x = (x - self.input_mean) / (self.input_std + 1e-8)

        # Forward through model
        return cast("torch.Tensor", self.model(x, chan_ids))

    def extract_features(
        self, x: torch.Tensor, chan_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract features (alias for forward).

        Args:
            x: Input tensor of shape (B, C, T)
            chan_ids: Channel IDs

        Returns:
            Features of shape (B, embed_num, embed_dim)
        """
        return self.forward(x, chan_ids)


def create_normalized_eegpt(
    checkpoint_path: str | None = None,
    normalize: bool = True,
    mean: float | None = None,
    std: float | None = None,
    normalization_path: str | None = None,
) -> EEGPTWrapper:
    """Create EEGPT model with normalization.

    Args:
        checkpoint_path: Path to checkpoint
        normalize: Whether to normalize inputs
        mean: Mean for normalization (overrides file)
        std: Standard deviation for normalization (overrides file)
        normalization_path: Path to normalization stats JSON

    Returns:
        EEGPT model with preprocessing
    """
    # Try to find normalization file if not specified
    if normalization_path is None and checkpoint_path:
        checkpoint_dir = Path(checkpoint_path).parent
        default_norm_path = checkpoint_dir / "normalization.json"
        if default_norm_path.exists():
            normalization_path = str(default_norm_path)

    model = EEGPTWrapper(checkpoint_path, normalization_path)
    model.normalize = normalize

    # Override with explicit values if provided
    if normalize and mean is not None and std is not None:
        model.set_normalization_params(mean, std)

    return model
