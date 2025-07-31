"""TUAB Abnormality Detection Linear Probe.

Binary classification task for detecting abnormal EEG patterns.
Target: AUROC ≥ 0.93 (from EEGPT paper)
"""

import logging
from pathlib import Path
from typing import Any

import torch

from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe

logger = logging.getLogger(__name__)


class AbnormalityDetectionProbe(EEGPTLinearProbe):
    """TUAB abnormality detection probe.

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
        super().__init__(
            checkpoint_path=checkpoint_path,
            n_input_channels=n_input_channels,
            n_classes=2,  # normal/abnormal
            freeze_backbone=True,  # Always freeze for linear probe
        )

        logger.info(f"Initialized AbnormalityDetectionProbe for {n_input_channels} channels")

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
