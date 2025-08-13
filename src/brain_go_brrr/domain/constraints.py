"""Constrained neural network layers.

Based on EEGPT implementation for weight-constrained layers.
Reference: Lawhern et al. "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces"
"""

from typing import Any

import torch
import torch.nn as nn


class LinearWithConstraint(nn.Linear):
    """Linear layer with weight norm constraint.

    Applies max norm constraint to weights during forward pass.
    This helps with regularization and prevents weight explosion.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        do_weight_norm: bool = True,
        max_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize constrained linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to use bias
            do_weight_norm: Whether to apply weight normalization
            max_norm: Maximum L2 norm for weight vectors
            **kwargs: Additional arguments for nn.Linear
        """
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(in_features, out_features, bias, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight constraint."""
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class Conv1dWithConstraint(nn.Conv1d):
    """1D Convolution with weight norm constraint."""

    def __init__(
        self, *args: Any, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs: Any
    ) -> None:
        """Initialize constrained conv1d layer."""
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight constraint."""
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)
