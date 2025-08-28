"""EEGPT model wrapper with proper preprocessing.

The pretrained EEGPT model expects normalized input data. Raw EEG signals
(typically 50 microvolts) are too small compared to the model's bias terms,
causing all outputs to be identical. This wrapper handles the necessary
preprocessing.
"""

import inspect
import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from brain_go_brrr.domain.preprocessing.nan_policy import validate_no_nan

from .eegpt_architecture import create_eegpt_model

logger = logging.getLogger(__name__)


class EEGPTWrapper(nn.Module):
    """EEGPT model with proper input preprocessing."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        normalization_path: str | None = None,
        model: nn.Module | None = None,
    ):
        """Initialize EEGPT with preprocessing.

        Args:
            checkpoint_path: Path to pretrained checkpoint
            normalization_path: Path to normalization stats JSON
            model: Optional pre-initialized model (for testing/DI)
        """
        super().__init__()
        # Allow dependency injection for better testability
        self.model = model if model is not None else create_eegpt_model(checkpoint_path)

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
            # CRITICAL: MNE outputs Volts (1e-5 scale), NOT normalized!
            # We MUST normalize or the model sees everything as zero
            # Use reasonable defaults for EEG in microvolts
            self.register_buffer("input_mean", torch.tensor(0.0))
            self.register_buffer("input_std", torch.tensor(50e-6))  # 50 microvolts typical
            self.normalize = True
            self._stats_source = "default_microvolt"
            logger.warning(
                "No normalization file found - using typical EEG scale (std=50μV). "
                "For production, compute stats from your training data!"
            )

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

    def _accepts_param(self, method: Any, param_name: str) -> bool:
        """Check if a method accepts a specific parameter.

        Args:
            method: The method to check
            param_name: Name of the parameter to look for

        Returns:
            True if the method accepts the parameter
        """
        try:
            return param_name in inspect.signature(method).parameters
        except Exception:
            return False

    def forward(
        self,
        x: torch.Tensor,
        chan_ids: torch.Tensor | None = None,
        return_all_temporal: bool = False,
    ) -> torch.Tensor:
        """Forward pass with preprocessing.

        Args:
            x: Input tensor of shape (B, C, T)
            chan_ids: Channel IDs for positional embedding
            return_all_temporal: If True, return all temporal features (B, N_temporal, 4, embed_dim)

        Returns:
            If return_all_temporal=False: Summary tokens of shape (B, embed_num, embed_dim)
            If return_all_temporal=True: All temporal features (B, N_temporal, embed_num, embed_dim)
        """
        if self.normalize:
            # Always normalize - datasets now provide raw mV data
            x = (x - self.input_mean) / (self.input_std + 1e-8)

        # Check if model accepts return_all_temporal parameter
        if self._accepts_param(self.model.forward, 'return_all_temporal'):
            return cast("torch.Tensor", self.model(x, chan_ids, return_all_temporal))
        else:
            # Model doesn't support the parameter, log once and fall back
            if not getattr(self, '_warned_ret_temporal', False):
                logger.debug("Model doesn't accept return_all_temporal; ignoring parameter")
                self._warned_ret_temporal = True
            return cast("torch.Tensor", self.model(x, chan_ids))

    def extract_features(
        self, x: torch.Tensor, chan_ids: torch.Tensor | None = None, summary: bool = True
    ) -> torch.Tensor:
        """Extract features with explicit shape contract.

        Args:
            x: Input tensor of shape (B, C, T)
            chan_ids: Channel IDs for positional embedding
            summary: If True, return flattened summary (B, 512). If False, return (B, 4, 512)

        Returns:
            If summary=True: Features of shape (B, 512)
            If summary=False: Features of shape (B, 4, 512)

        Raises:
            ValueError: If input contains NaN or Inf values
        """
        # Validate input - reject NaN/Inf
        validate_no_nan(x, "EEG input")

        # Get features from model
        features = self.forward(x, chan_ids, return_all_temporal=False)

        # Ensure consistent output shape based on summary flag
        if features.ndim == 3 and features.shape[1] == 4:  # (B, 4, 512)
            if summary:
                # Flatten the 4 summary tokens to (B, 512) by averaging
                return features.mean(dim=1)  # (B, 512)
            else:
                return features  # (B, 4, 512)
        elif features.ndim == 2:  # Already flattened or averaged
            if summary:
                return features  # (B, 512)
            else:
                # Can't unflatten, return as is with warning
                logger.warning("Cannot unflatten features to (B, 4, 512)")
                return features
        else:
            # Unexpected shape, log and return
            logger.warning(f"Unexpected feature shape: {features.shape}")
            return features


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
