"""Robust EEGPT Linear Probe with comprehensive NaN prevention."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.modules.constraints import LinearWithConstraint

logger = logging.getLogger(__name__)


class RobustEEGPTLinearProbe(nn.Module):
    """EEGPT Linear Probe with robust NaN handling.

    Key improvements:
    1. Input validation and clipping
    2. Stable normalization with epsilon
    3. Feature validation after EEGPT
    4. Gradient-friendly operations
    """

    def __init__(
        self,
        checkpoint_path: Path,
        n_input_channels: int,
        n_classes: int,
        embed_dim: int = 512,
        n_summary_tokens: int = 4,
        max_norm: float = 0.25,
        freeze_backbone: bool = True,
        input_clip_value: float = 50.0,
        normalization_eps: float = 1e-5,
    ) -> None:
        """Initialize Robust EEGPT Linear Probe.

        Args:
            checkpoint_path: Path to pretrained EEGPT checkpoint
            n_input_channels: Number of input EEG channels
            n_classes: Number of output classes
            embed_dim: EEGPT embedding dimension (default: 512)
            n_summary_tokens: Number of summary tokens from EEGPT (default: 4)
            max_norm: Maximum norm for weight constraint (default: 0.25)
            freeze_backbone: Whether to freeze EEGPT backbone (default: True)
            input_clip_value: Max absolute value for input clipping (default: 50.0)
            normalization_eps: Epsilon for stable normalization (default: 1e-5)
        """
        super().__init__()

        # Load pretrained EEGPT
        logger.info(f"Loading EEGPT from {checkpoint_path}")
        self.backbone = create_normalized_eegpt(
            checkpoint_path=str(checkpoint_path), normalize=True
        )

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing EEGPT backbone parameters")
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Channel adaptation with initialization
        self.n_eegpt_channels = 20
        self.channel_adapter = nn.Conv1d(
            in_channels=n_input_channels,
            out_channels=self.n_eegpt_channels,
            kernel_size=1,
            bias=True,  # Add bias for stability
        )

        # Initialize channel adapter with small weights
        nn.init.xavier_uniform_(self.channel_adapter.weight, gain=0.1)
        if self.channel_adapter.bias is not None:
            nn.init.zeros_(self.channel_adapter.bias)

        # Classification head with dropout
        feature_dim = embed_dim * n_summary_tokens  # 512 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  # Add dropout
            LinearWithConstraint(feature_dim, feature_dim, max_norm=max_norm),
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout
            LinearWithConstraint(feature_dim, n_classes, max_norm=max_norm),
        )

        # Initialize classifier weights
        for module in self.classifier.modules():
            if isinstance(module, LinearWithConstraint):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Store config
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.n_summary_tokens = n_summary_tokens
        self.freeze_backbone = freeze_backbone
        self.input_clip_value = input_clip_value
        self.normalization_eps = normalization_eps

        # Statistics tracking
        self.register_buffer("nan_count", torch.tensor(0))
        self.register_buffer("clip_count", torch.tensor(0))

        logger.info(
            f"Initialized RobustEEGPTLinearProbe: {n_input_channels} channels -> {n_classes} classes"
        )

    def _validate_and_clean_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and clean input tensor."""
        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.nan_count = self.nan_count + 1  # type: ignore[operator,assignment]
            logger.warning(f"Found NaN/Inf in input (count: {self.nan_count.item()})")  # type: ignore[operator]
            x = torch.nan_to_num(
                x, nan=0.0, posinf=self.input_clip_value, neginf=-self.input_clip_value
            )

        # Clip extreme values
        before_clip = x
        x = torch.clamp(x, min=-self.input_clip_value, max=self.input_clip_value)
        if not torch.equal(before_clip, x):
            self.clip_count = self.clip_count + 1  # type: ignore[operator,assignment]
            logger.debug(f"Clipped input values (count: {self.clip_count.item()})")  # type: ignore[operator]

        return x

    def _robust_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Robust z-score normalization per channel."""
        # Calculate stats with numerical stability
        mean = x.mean(dim=-1, keepdim=True)

        # Robust std calculation
        x_centered = x - mean
        variance = (x_centered**2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.normalization_eps)

        # Normalize
        x_normalized = x_centered / std

        # Final safety check
        x_normalized = torch.clamp(x_normalized, min=-10, max=10)

        return x_normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with robust error handling.

        Args:
            x: Input EEG data [batch, channels, time]

        Returns:
            Logits [batch, n_classes]
        """
        # Ensure input is 3D
        if x.dim() == 4:
            x = x.squeeze(1)

        # Validate and clean input
        x = self._validate_and_clean_input(x)

        # Robust normalization
        x = self._robust_normalize(x)

        # Adapt channels
        x = self.channel_adapter(x)  # [batch, 20, time]

        # Extract features with backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.extract_features(x)
        else:
            features = self.backbone.extract_features(x)

        # Validate features
        if torch.isnan(features).any() or torch.isinf(features).any():
            nan_mask = torch.isnan(features) | torch.isinf(features)
            nan_percentage = nan_mask.float().mean() * 100
            logger.warning(f"NaN/Inf in EEGPT features: {nan_percentage:.1f}% of values")

            # Replace with zeros as last resort
            features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

            # Add small noise to prevent all-zero features
            if (features == 0).all():
                features = features + torch.randn_like(features) * 0.01

        # Flatten features
        batch_size = features.shape[0]
        features = features.reshape(batch_size, -1)  # [batch, 2048]

        # Final feature validation
        features = torch.clamp(features, min=-100, max=100)

        # Classify
        logits = self.classifier(features)

        # Final logit validation
        logits = torch.clamp(logits, min=-20, max=20)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities with numerical stability."""
        logits = self.forward(x)
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        return probs

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_probe(self, path: Path) -> None:
        """Save only the probe weights (not backbone)."""
        probe_state = {
            "channel_adapter": self.channel_adapter.state_dict(),
            "classifier": self.classifier.state_dict(),
            "config": {
                "n_input_channels": self.n_input_channels,
                "n_classes": self.n_classes,
                "embed_dim": self.embed_dim,
                "n_summary_tokens": self.n_summary_tokens,
                "input_clip_value": self.input_clip_value,
                "normalization_eps": self.normalization_eps,
            },
            "statistics": {
                "nan_count": self.nan_count.item(),  # type: ignore[operator]
                "clip_count": self.clip_count.item(),  # type: ignore[operator]
            },
        }
        torch.save(probe_state, path)
        logger.info(f"Saved probe weights to {path}")

    def load_probe(self, path: Path) -> None:
        """Load probe weights."""
        checkpoint = torch.load(path, map_location="cpu")
        self.channel_adapter.load_state_dict(checkpoint["channel_adapter"])
        self.classifier.load_state_dict(checkpoint["classifier"])

        # Load statistics if available
        if "statistics" in checkpoint:
            self.nan_count.fill_(checkpoint["statistics"]["nan_count"])  # type: ignore[operator]
            self.clip_count.fill_(checkpoint["statistics"]["clip_count"])  # type: ignore[operator]

        logger.info(f"Loaded probe weights from {path}")
