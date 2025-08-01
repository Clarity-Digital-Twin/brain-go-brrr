"""EEGPT Linear Probe implementation.

Based on the EEGPT paper: "EEGPT: Pretrained Transformer for Universal
and Reliable Representation of EEG Signals"
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.modules.constraints import LinearWithConstraint

logger = logging.getLogger(__name__)


class EEGPTLinearProbe(nn.Module):
    """EEGPT Linear Probe following the paper implementation.

    Architecture:
    1. Channel adaptation layer (1x1 conv)
    2. Frozen EEGPT encoder
    3. Two-layer linear classifier with GELU activation

    This probe is designed to work with frozen EEGPT features,
    requiring minimal training while achieving strong performance.
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
    ) -> None:
        """Initialize EEGPT Linear Probe.

        Args:
            checkpoint_path: Path to pretrained EEGPT checkpoint
            n_input_channels: Number of input EEG channels
            n_classes: Number of output classes
            embed_dim: EEGPT embedding dimension (default: 512)
            n_summary_tokens: Number of summary tokens from EEGPT (default: 4)
            max_norm: Maximum norm for weight constraint (default: 0.25)
            freeze_backbone: Whether to freeze EEGPT backbone (default: True)
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

        # Channel adaptation layer
        # Reference implementation uses 20 channels for downstream tasks
        self.n_eegpt_channels = 20
        self.channel_adapter = nn.Conv1d(
            in_channels=n_input_channels,
            out_channels=self.n_eegpt_channels,
            kernel_size=1,
            bias=False,  # Reference uses no bias
        )

        # Classification head
        feature_dim = embed_dim * n_summary_tokens  # 512 * 4 = 2048
        self.classifier = nn.Sequential(
            LinearWithConstraint(feature_dim, feature_dim, max_norm=max_norm),
            nn.GELU(),
            LinearWithConstraint(feature_dim, n_classes, max_norm=max_norm),
        )

        # Store config
        self.n_input_channels = n_input_channels
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.n_summary_tokens = n_summary_tokens
        self.freeze_backbone = freeze_backbone

        logger.info(
            f"Initialized EEGPTLinearProbe: {n_input_channels} channels -> {n_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input EEG data [batch, channels, time]

        Returns:
            Logits [batch, n_classes]
        """
        # Ensure input is 3D
        if x.dim() == 4:
            # Remove extra dimension if present
            x = x.squeeze(1)

        # Input normalization (z-score per channel)
        mean = x.mean(dim=-1, keepdim=True)  # Average over time
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mean) / std

        # Adapt channels from input to EEGPT's expected channels
        x = self.channel_adapter(x)  # [batch, 20, time]

        # Extract features with backbone
        # EEGPT returns [batch, embed_num, embed_dim]
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.extract_features(x)
        else:
            features = self.backbone.extract_features(x)
            
        # Check for NaN in features and replace with zeros if found
        if torch.isnan(features).any():
            print(f"WARNING: NaN detected in EEGPT features, replacing with zeros")
            features = torch.nan_to_num(features, nan=0.0)

        # Flatten the features: [batch, embed_num, embed_dim] -> [batch, embed_num * embed_dim]
        batch_size = features.shape[0]
        features = features.reshape(batch_size, -1)  # [batch, 2048]

        # Classify
        logits = self.classifier(features)

        return logits  # type: ignore[no-any-return]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input EEG data [batch, channels, time]

        Returns:
            Probabilities [batch, n_classes]
        """
        logits = self.forward(x)
        return functional.softmax(logits, dim=-1)

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_probe(self, path: Path) -> None:
        """Save only the probe weights (not backbone).

        Args:
            path: Path to save checkpoint
        """
        # Only save non-backbone parameters
        probe_state = {
            "channel_adapter": self.channel_adapter.state_dict(),
            "classifier": self.classifier.state_dict(),
            "config": {
                "n_input_channels": self.n_input_channels,
                "n_classes": self.n_classes,
                "embed_dim": self.embed_dim,
                "n_summary_tokens": self.n_summary_tokens,
            },
        }
        torch.save(probe_state, path)
        logger.info(f"Saved probe weights to {path}")

    def load_probe(self, path: Path) -> None:
        """Load probe weights.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.channel_adapter.load_state_dict(checkpoint["channel_adapter"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        logger.info(f"Loaded probe weights from {path}")
