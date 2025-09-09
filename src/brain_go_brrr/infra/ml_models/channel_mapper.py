"""TUEV Channel Mapper for paper parity.

Maps 23 TUEV channels to 20 standard channels using learnable convolutions.
Based on EEGPT reference implementation.
"""

import torch
import torch.nn as nn

from brain_go_brrr.domain.constraints import Conv2dWithConstraint


class TUEVChannelMapper(nn.Module):
    """Learnable 23→20 channel mapper for TUEV paper parity.

    Matches EEGPT reference implementation exactly.

    Architecture from reference:
    - Conv2dWithConstraint(23→20, kernel=1x1)
    - BatchNorm2d(20)
    - GELU activation
    - Conv2d depthwise (kernel=1x55, groups=20)
    - BatchNorm2d(20)
    - Dropout(0.8)
    """

    def __init__(self, in_channels: int = 23, out_channels: int = 20, dropout: float = 0.8):
        """Initialize the channel mapper.

        Args:
            in_channels: Number of input channels (23 for TUEV)
            out_channels: Number of output channels (20 for EEGPT)
            dropout: Dropout rate (0.8 from EEGPT reference)
        """
        super().__init__()

        # Spatial convolution (23→20 learned mapping)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Temporal convolution (depthwise, kernel=1x55)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 55),
                groups=out_channels,
                padding=(0, 27),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mapper.

        Args:
            x: Input tensor of shape (B, 23, T) or (B, 23, 1, T)
               where B=batch, T=time samples (1024 for 4s @ 256Hz)

        Returns:
            Output tensor of shape (B, 20, T) or (B, 20, 1, T)
        """
        # Handle both 3D and 4D inputs
        if x.ndim == 3:
            # (B, C, T) -> (B, C, 1, T) for conv2d
            x = x.unsqueeze(2)
            squeeze_output = True
        else:
            squeeze_output = False

        # Apply spatial mapping: (B, 23, 1, T) -> (B, 20, 1, T)
        x = self.spatial_conv(x)

        # Apply temporal convolution: (B, 20, 1, T) -> (B, 20, 1, T)
        x = self.temporal_conv(x)

        # Remove height dimension if we added it
        if squeeze_output:
            x = x.squeeze(2)  # (B, 20, 1, T) -> (B, 20, T)

        return x
