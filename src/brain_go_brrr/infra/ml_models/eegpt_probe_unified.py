"""Unified EEGPT Probe implementation.

This replaces the multiple probe variants with a single configurable implementation.
Supports both linear and two-layer architectures, with optional robust mode.
"""

import inspect
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from brain_go_brrr.domain.constraints import LinearWithConstraint
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt

logger = logging.getLogger(__name__)


class EEGPTProbe(nn.Module):
    """Unified EEGPT probe with configurable architecture.

    .. deprecated:: 1.1.0
        Use :class:`ProbeFactory` instead. This class will be removed in v2.0.0.

        Migration example::

            # Old way
            probe = EEGPTProbe(checkpoint_path, architecture="two_layer")

            # New way
            from brain_go_brrr.infra.ml_models import ProbeFactory
            probe = ProbeFactory.create(2048, 256, 2, architecture="two_layer")

    This replaces:
    - EEGPTLinearProbe
    - RobustEEGPTLinearProbe
    - EEGPTTwoLayerProbe

    All with a single configurable implementation.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        n_classes: int = 2,
        n_input_channels: int = 20,
        architecture: str = "linear",  # "linear" or "two_layer"
        robust_mode: bool = False,  # Enable NaN handling
        channel_adapter: bool = False,  # Use channel adaptation layer
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        max_norm: float = 0.25,  # For LinearWithConstraint
        input_clip_value: float = 50.0,  # For robust mode
        backbone: nn.Module | None = None,  # Allow dependency injection
    ):
        """Initialize unified EEGPT probe.

        Args:
            checkpoint_path: Path to EEGPT checkpoint (optional if backbone provided)
            n_classes: Number of output classes
            n_input_channels: Number of input EEG channels
            architecture: "linear" for single layer, "two_layer" for two layers
            robust_mode: Enable input validation and NaN prevention
            channel_adapter: Use 1x1 conv for channel adaptation
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            freeze_backbone: Freeze EEGPT backbone weights
            max_norm: Maximum norm for weight constraint
            input_clip_value: Clipping value for robust mode
            backbone: Pre-initialized backbone (for testing)
        """
        super().__init__()

        # Issue deprecation warning
        warnings.warn(
            "EEGPTProbe is deprecated and will be removed in v2.0.0. "
            "Use ProbeFactory.create() instead for new code.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Load or use provided backbone
        if backbone is not None:
            self.backbone = backbone
        elif checkpoint_path is not None:
            self.backbone = create_normalized_eegpt(str(checkpoint_path))
        else:
            raise ValueError("Either checkpoint_path or backbone must be provided")

        # Freeze backbone if requested
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optional channel adapter (1x1 conv)
        self.use_channel_adapter = channel_adapter
        if channel_adapter:
            self.channel_adapter = nn.Conv1d(
                in_channels=n_input_channels,
                out_channels=20,  # EEGPT expects 20 channels
                kernel_size=1,
                stride=1,
                padding=0,
            )

        # Build probe architecture
        self.architecture = architecture
        if architecture == "linear":
            # Single layer probe with constraint
            self.probe = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                LinearWithConstraint(hidden_dim, n_classes, max_norm=max_norm),
            )
        elif architecture == "two_layer":
            # Two layer probe
            self.probe = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                LinearWithConstraint(hidden_dim // 2, n_classes, max_norm=max_norm),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Robust mode settings
        self.robust_mode = robust_mode
        self.input_clip_value = input_clip_value

        # Store config for reference
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def _accepts_param(self, param_name: str) -> bool:
        """Check if backbone's extract_features accepts a parameter.

        Args:
            param_name: Name of the parameter to check

        Returns:
            True if the method accepts the parameter
        """
        try:
            if hasattr(self.backbone, 'extract_features'):
                method = self.backbone.extract_features
                if callable(method):
                    return param_name in inspect.signature(method).parameters
            return False
        except Exception:
            return False

    def forward(self, x: torch.Tensor, return_all_temporal: bool = False) -> torch.Tensor:
        """Forward pass through probe.

        Args:
            x: Input tensor of shape (B, C, T)
            return_all_temporal: If True, extract all temporal features

        Returns:
            Logits of shape (B, n_classes)
        """
        # Robust mode: validate and clip input
        if self.robust_mode:
            # Check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("NaN or Inf detected in input, replacing with zeros")
                x = torch.nan_to_num(
                    x, nan=0.0, posinf=self.input_clip_value, neginf=-self.input_clip_value
                )

            # Clip extreme values
            x = torch.clamp(x, min=-self.input_clip_value, max=self.input_clip_value)

        # Channel adaptation if enabled
        if self.use_channel_adapter:
            x = self.channel_adapter(x)

        # Extract features from backbone
        with torch.no_grad():  # Always use no_grad for now
            # Check if backbone accepts return_all_temporal parameter
            if self._accepts_param('return_all_temporal'):
                if hasattr(self.backbone, 'extract_features') and callable(
                    self.backbone.extract_features
                ):
                    features = self.backbone.extract_features(
                        x, return_all_temporal=return_all_temporal
                    )
                else:
                    features = self.backbone(x)
            else:
                # Fallback for older backbones
                if hasattr(self.backbone, 'extract_features') and callable(
                    self.backbone.extract_features
                ):
                    features = self.backbone.extract_features(x)
                else:
                    features = self.backbone(x)

        # Handle different feature shapes
        if return_all_temporal:
            # Deprecated for production: do NOT use all temporal patches for probes.
            # If explicitly requested, flatten everything for backward compatibility.
            # Expected features: (B, N_temporal, 4, 512)
            batch_size = features.shape[0]
            features = features.reshape(batch_size, -1)
        else:
            # Preferred production path: use summary tokens only.
            # Expected features: (B, 4, 512) → flatten to (B, 2048)
            if features.dim() == 3 and features.shape[1] == 4 and features.shape[2] == 512:
                features = features.reshape(features.size(0), -1)
            elif features.dim() == 2:
                # Already flattened (e.g., 2048) or averaged (512). Pass through.
                pass
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

        # Robust mode: check features
        if self.robust_mode and (torch.isnan(features).any() or torch.isinf(features).any()):
            logger.warning("NaN or Inf in features, replacing with zeros")
            features = torch.nan_to_num(features, nan=0.0)

        # Pass through probe
        logits = self.probe(features)

        # Final robust check
        if self.robust_mode and (torch.isnan(logits).any() or torch.isinf(logits).any()):
            logger.error("NaN or Inf in output logits!")
            logits = torch.nan_to_num(logits, nan=0.0)

        return logits  # type: ignore[no-any-return]

    def get_feature_dim(self) -> int:
        """Get the expected feature dimension after backbone."""
        # This is mainly for compatibility
        return self.hidden_dim

    # Compatibility methods for existing tests
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Probabilities of shape (B, n_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_num_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        from contextlib import suppress

        # Handle uninitialized LazyLinear parameters
        count = 0
        for p in self.parameters():
            if p.requires_grad:
                with suppress(RuntimeError, ValueError):
                    count += p.numel()
        return count

    def save_probe(self, path: Path | str) -> None:
        """Save probe state.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        state = {
            'probe_state_dict': self.probe.state_dict(),
            'config': {
                'n_classes': self.n_classes,
                'architecture': self.architecture,
                'robust_mode': self.robust_mode,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
            },
        }
        torch.save(state, path)

    def load_probe(self, path: Path | str) -> None:
        """Load probe state.

        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        # Use weights_only=False to handle uninitialized LazyLinear parameters
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # nosec:weights_only - checkpoint contains model weights and optimizer state
        self.probe.load_state_dict(checkpoint['probe_state_dict'])

    @property
    def classifier(self) -> nn.Module:
        """Alias for probe head (backwards compatibility)."""
        return self.probe


# Compatibility aliases for migration
def create_eegpt_probe(
    checkpoint_path: Path | str,
    n_classes: int = 2,
    probe_type: str = "linear",
    robust: bool = False,
    **kwargs: Any,
) -> EEGPTProbe:
    """Factory function for creating EEGPT probes.

    Args:
        checkpoint_path: Path to EEGPT checkpoint
        n_classes: Number of output classes
        probe_type: "linear", "two_layer", or "robust" (sets robust_mode)
        robust: Enable robust mode (alternative to probe_type="robust")
        **kwargs: Additional arguments passed to EEGPTProbe

    Returns:
        Configured EEGPT probe
    """
    if probe_type == "robust":
        robust = True
        probe_type = "linear"

    architecture = "two_layer" if probe_type == "two_layer" else "linear"

    return EEGPTProbe(
        checkpoint_path=checkpoint_path,
        n_classes=n_classes,
        architecture=architecture,
        robust_mode=robust,
        **kwargs,
    )
