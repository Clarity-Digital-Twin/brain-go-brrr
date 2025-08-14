"""Infrastructure adapter for EEGPT classifier head.

This adapter implements the domain port for abnormality classification,
wrapping the actual EEGPT model implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002
import torch

if TYPE_CHECKING:
    import numpy.typing as npt

from brain_go_brrr.domain.abnormal.ports import AbnormalityHeadPort


class EEGPTClassifierAdapter(AbnormalityHeadPort):
    """Adapter wrapping EEGPT classifier to implement domain port."""

    def __init__(self, classifier: torch.nn.Module, device: str = "cpu") -> None:
        """Initialize the adapter with a PyTorch classifier.

        Args:
            classifier: PyTorch classifier module (already loaded)
            device: Device to run inference on
        """
        self._classifier = classifier.eval()
        self._device = torch.device(device)
        self._classifier = self._classifier.to(self._device)

    def predict_proba(self, X: npt.NDArray[np.float32]) -> float:  # noqa: N803
        """Predict abnormality probability.

        Implements the domain port interface.

        Args:
            X: Feature vector from EEGPT

        Returns:
            Probability of abnormality (0-1)
        """
        with torch.inference_mode():
            # Convert numpy array to tensor
            features = torch.from_numpy(X).float()

            # Add batch dimension if needed
            if features.dim() == 1:
                features = features.unsqueeze(0)

            # Move to device
            features = features.to(self._device, non_blocking=True)

            # Run inference
            logits = self._classifier(features)

            # Apply sigmoid or softmax depending on output shape
            if logits.shape[-1] == 1:
                # Binary classification with single output
                probs = torch.sigmoid(logits)
                prob = probs.squeeze().cpu().item()
            else:
                # Multi-class, take abnormal class (index 1)
                probs = torch.softmax(logits, dim=-1)
                prob = probs[0, 1].cpu().item()

        return float(prob)
