"""Linear probe classifier for EEGPT downstream tasks."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional


class LinearProbeHead(nn.Module):
    """Linear probe classifier for EEGPT features.

    According to the EEGPT paper, linear probing freezes the pretrained encoder
    and only trains a linear classification layer on top of the summary tokens.
    """

    def __init__(
        self,
        input_dim: int = 2048,  # 4 summary tokens x 512 dims
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize linear probe.

        Args:
            input_dim: Input dimension (n_summary_tokens x embed_dim)
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Single linear layer as per paper
        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear probe.

        Args:
            features: EEGPT features of shape (batch_size, n_summary_tokens, embed_dim)
                     or (batch_size, n_summary_tokens * embed_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Flatten summary tokens if needed
        if features.dim() == 3:
            features = features.view(features.size(0), -1)

        # Apply dropout
        features = self.dropout(features)

        # Linear classification
        logits = self.classifier(features)

        return logits  # type: ignore[no-any-return]

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            features: EEGPT features

        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(features)
        return functional.softmax(logits, dim=-1)


class SleepStageProbe(LinearProbeHead):
    """Specialized probe for 5-stage sleep classification."""

    def __init__(self, input_dim: int = 2048, dropout: float = 0.1):
        """Initialize sleep stage probe with 5 classes."""
        super().__init__(
            input_dim=input_dim,
            num_classes=5,  # W, N1, N2, N3, REM
            dropout=dropout,
        )

        # Sleep stage mapping
        self.stage_names = ["W", "N1", "N2", "N3", "REM"]
        self.stage_to_idx = {name: i for i, name in enumerate(self.stage_names)}

    def predict_stage(self, features: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Predict sleep stage names and confidence scores.

        Args:
            features: EEGPT features

        Returns:
            Tuple of (stage_names, confidence_scores)
        """
        probs = self.predict_proba(features)
        confidences, indices = torch.max(probs, dim=1)

        stage_names = [self.stage_names[idx.item()] for idx in indices]

        return stage_names, confidences


class AbnormalityProbe(LinearProbeHead):
    """Specialized probe for binary abnormality detection."""

    def __init__(self, input_dim: int = 2048, dropout: float = 0.1):
        """Initialize abnormality probe with 2 classes."""
        super().__init__(
            input_dim=input_dim,
            num_classes=2,  # Normal, Abnormal
            dropout=dropout,
        )

    def predict_abnormal_probability(self, features: torch.Tensor) -> torch.Tensor:
        """Get probability of abnormality.

        Args:
            features: EEGPT features

        Returns:
            Abnormality probabilities of shape (batch_size,)
        """
        probs = self.predict_proba(features)
        return probs[:, 1]  # Return abnormal class probability


class MotorImageryProbe(LinearProbeHead):
    """Specialized probe for motor imagery classification."""

    def __init__(
        self,
        input_dim: int = 2048,
        num_classes: int = 4,  # Left hand, right hand, feet, tongue
        dropout: float = 0.1,
    ):
        """Initialize motor imagery probe."""
        super().__init__(input_dim=input_dim, num_classes=num_classes, dropout=dropout)

        # Motor imagery class mapping
        self.class_names = ["left_hand", "right_hand", "feet", "tongue"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}


def create_probe_for_task(
    task: str, input_dim: int = 2048, dropout: float = 0.1, **kwargs: Any
) -> LinearProbeHead:
    """Factory function to create appropriate probe for task.

    Args:
        task: Task name ('sleep', 'abnormality', 'motor_imagery', etc.)
        input_dim: Input dimension
        dropout: Dropout rate
        **kwargs: Additional task-specific arguments

    Returns:
        Appropriate probe instance
    """
    task = task.lower()

    if task == "sleep":
        return SleepStageProbe(input_dim=input_dim, dropout=dropout)
    elif task == "abnormality":
        return AbnormalityProbe(input_dim=input_dim, dropout=dropout)
    elif task == "motor_imagery":
        num_classes = kwargs.get("num_classes", 4)
        return MotorImageryProbe(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    else:
        # Generic probe for other tasks
        num_classes = kwargs.get("num_classes", 2)
        return LinearProbeHead(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
