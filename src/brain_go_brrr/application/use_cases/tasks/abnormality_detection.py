"""TUAB Abnormality Detection Probe (composition).

Binary classification task for detecting abnormal EEG patterns.
Target: AUROC ≥ 0.93 (from EEGPT paper)

P2: Migrated from deprecated EEGPTProbe inheritance to composition with
ProbeFactory and the EEGPT wrapper for feature extraction.
The orchestration pipeline already uses this pattern.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, cast

import torch

from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.infra.ml_models.probe_factory import (
    ProbeFactory,
    migrate_eegpt_probe_to_factory,
)
from brain_go_brrr.utils.probe_utils import prepare_probe_features

logger = logging.getLogger(__name__)


class AbnormalityDetectionProbe:
    """TUAB abnormality detection probe using composition.

    Binary classification: normal (0) vs abnormal (1)
    Input: 20 channels, 8s windows at 256Hz

    Expected performance (from EEGPT paper):
    - AUROC: ≥ 0.93
    - Training time: < 1 hour on single GPU
    """

    # TUAB channel names with modern naming (20 channels)
    TUAB_CHANNELS = [
        "FP1",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "T7",  # Modern name for T3
        "C3",
        "CZ",
        "C4",
        "T8",  # Modern name for T4
        "P7",  # Modern name for T5
        "P3",
        "PZ",
        "P4",
        "P8",  # Modern name for T6
        "O1",
        "O2",
        "OZ",
    ]

    def __init__(self, checkpoint_path: Path, n_input_channels: int = 20) -> None:
        """Initialize abnormality detection probe.

        Args:
            checkpoint_path: Path to pretrained EEGPT checkpoint
            n_input_channels: Number of input channels (default: 20 for TUAB)
        """
        # EEGPT backbone (normalized)
        self.backbone = create_normalized_eegpt(checkpoint_path=str(checkpoint_path))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Probe head that expects (B, 2048)
        self.head = ProbeFactory.create_for_task("abnormality", n_classes=2)

        logger.info(f"Initialized AbnormalityDetectionProbe for {n_input_channels} channels")
        warnings.warn(
            "AbnormalityDetectionProbe API may change in a future minor release. "
            "Prefer orchestration helpers for long-lived code.",
            PendingDeprecationWarning,
            stacklevel=2,
        )

    def forward(self, x: torch.Tensor, channel_names: list[str] | None = None) -> torch.Tensor:
        """Forward through backbone then probe head.

        Args:
            x: Input EEG tensor [B, C, T]
            channel_names: Optional list of channel names (for wrapper compatibility)

        Returns:
            Logits tensor [B, 2]
        """
        with torch.no_grad():
            # Extract features with summary=False to get (B, 4, 512)
            # Note: channel_names is for API compatibility but not used by extract_features
            features = self.backbone.extract_features(x, chan_ids=None, summary=False)
        probe_in = prepare_probe_features(features)  # (B, 2048)
        logits = self.head(probe_in)
        return cast("torch.Tensor", logits)

    def load_head_checkpoint(self, checkpoint: dict[str, Any] | str | Path) -> None:
        """Load probe head weights from either legacy or current formats.

        Supports both legacy EEGPTProbe format (with "probe_state_dict") and the
        current ProbeFactory format (with "model_state_dict").
        """
        if isinstance(checkpoint, (str, Path)):
            from brain_go_brrr.infra.safe_load import safe_load

            payload = safe_load(str(checkpoint))
        else:
            payload = checkpoint

        if "probe_state_dict" in payload:
            migrated = migrate_eegpt_probe_to_factory(payload["probe_state_dict"])
            self.head.load_state_dict(migrated)
        elif "model_state_dict" in payload:
            self.head.load_state_dict(payload["model_state_dict"])
        else:
            raise KeyError("Checkpoint must contain 'probe_state_dict' or 'model_state_dict'.")

    def predict_proba(
        self, x: torch.Tensor, channel_names: list[str] | None = None
    ) -> torch.Tensor:
        """Softmax probabilities for abnormality classes."""
        logits = self.forward(x, channel_names)
        return torch.softmax(logits, dim=-1)

    def get_abnormality_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Get abnormality probability for each sample.

        Args:
            x: Input EEG data [batch, channels, time]

        Returns:
            Abnormality probabilities [batch]
        """
        probs = self.predict_proba(x)
        return probs[:, 1]  # Abnormal class probability

    def predict_with_confidence(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Make prediction with confidence scores.

        Args:
            x: Input EEG data [batch, channels, time]

        Returns:
            Dictionary with:
                - predictions: Binary predictions [batch]
                - probabilities: Abnormality probabilities [batch]
                - confidence: Confidence scores [batch]
        """
        with torch.no_grad():
            probs = self.predict_proba(x)
            abnormal_probs = probs[:, 1]

            # Binary predictions (threshold at 0.5)
            predictions = (abnormal_probs > 0.5).long()

            # Confidence is how far from decision boundary
            confidence = torch.abs(abnormal_probs - 0.5) * 2

        return {
            "predictions": predictions,
            "probabilities": abnormal_probs,
            "confidence": confidence,
        }

    @staticmethod
    def get_data_requirements() -> dict[str, Any]:
        """Get data requirements for TUAB dataset."""
        return {
            "sampling_rate": 256,  # Hz
            "window_duration": 8.0,  # seconds
            "window_samples": 2048,  # Must be divisible by EEGPT patch_size (64)
            "n_channels": 20,
            "channel_names": AbnormalityDetectionProbe.TUAB_CHANNELS,
            "preprocessing": {
                "bandpass": (0.5, 50.0),  # Hz
                "notch": 60.0,  # Hz (US power line)
                "reference": "average",
            },
        }
