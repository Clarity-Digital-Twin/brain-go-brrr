"""Wu 2025 Seizure Transformer architecture.

CNN+Transformer architecture for per-timestep seizure detection.
Copied from reference_repos/SeizureTransformer/wu_2025/ to match pretrained weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class SeizureTransformer(nn.Module):
    """Wu 2025 Seizure Transformer with CNN encoder and transformer blocks."""

    def __init__(
        self,
        in_channels: int = 19,
        in_samples: int = 15360,
        drop_rate: float = 0.1,
    ) -> None:
        """Initialize SeizureTransformer.

        Args:
            in_channels: Number of EEG channels (default: 19).
            in_samples: Number of samples per window (default: 15360 for 60s@256Hz).
            drop_rate: Dropout rate (default: 0.1).
        """
        super().__init__()

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.drop_rate = drop_rate

        # Parameters from EQTransformer repository
        self.filters = [32, 64, 128, 256, 512]  # Number of filters for the convolutions
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]  # Kernel sizes for the convolutions
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        # Encoder stack
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        self.position_encoding = PositionalEncoding(d_model=512)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=8
        )

        # Detection decoder and final Conv
        self.decoder_d = Decoder(
            input_channels=512,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples,
            original_compatible=False,
        )
        self.conv_d = nn.Conv1d(
            in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning per-timestep probabilities.

        Args:
            x: Input tensor of shape (batch, channels, samples).

        Returns:
            Per-timestep predictions of shape (batch, samples).
        """
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        x, skips = self.encoder(x)
        res_x = self.res_cnn_stack(x)

        x = res_x.permute(2, 0, 1)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = x + res_x

        detection = self.decoder_d(x, skips)
        detection = torch.sigmoid(self.conv_d(detection))
        detection = torch.squeeze(detection, dim=1)  # Remove channel dimension

        return detection


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim].

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):
    """Encoder stack."""

    def __init__(
        self,
        input_channels: int,
        filters: list[int],
        kernel_sizes: list[int],
        in_samples: int,
    ) -> None:
        """Initialize Encoder.

        Args:
            input_channels: Number of input channels.
            filters: List of filter sizes for each layer.
            kernel_sizes: List of kernel sizes for each layer.
            in_samples: Number of input samples.
        """
        super().__init__()

        convs = []
        pools = []
        elus = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels, *filters[:-1]], filters, kernel_sizes, strict=False
        ):
            convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            elus.append(nn.ELU(inplace=True))
            in_samples = (in_samples + padding) // 2

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.elus = nn.ModuleList(elus)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through encoder.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (encoded tensor, skip connections).
        """
        skips = []
        for conv, pool, padding, elu in zip(
            self.convs, self.pools, self.paddings, self.elus, strict=False
        ):
            x = elu(conv(x))
            skips.append(x)
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)

        return x, skips


class Decoder(nn.Module):
    """Decoder stack."""

    def __init__(
        self,
        input_channels: int,
        filters: list[int],
        kernel_sizes: list[int],
        out_samples: int,
        original_compatible: bool = False,
    ) -> None:
        """Initialize Decoder.

        Args:
            input_channels: Number of input channels.
            filters: List of filter sizes for each layer.
            kernel_sizes: List of kernel sizes for each layer.
            out_samples: Number of output samples.
            original_compatible: Whether to use original-compatible mode.
        """
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        elus = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels, *filters[:-1]], filters, kernel_sizes, strict=False
        ):
            convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            elus.append(nn.ELU(inplace=True))

        self.convs = nn.ModuleList(convs)
        self.elus = nn.ModuleList(elus)

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            x: Input tensor.
            skip_connections: Skip connections from encoder.

        Returns:
            Decoded tensor.
        """
        for i, (conv, elu) in enumerate(zip(self.convs, self.elus, strict=False)):
            x = self.upsample(x)
            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]
            x = elu(conv(x))
            if skip_connections is not None and i < len(skip_connections):
                # Use reverse order: first decoder block gets skip from the last encoder block.
                skip = skip_connections[-(i + 1)]
                x = x + skip
        return x


class ResCNNStack(nn.Module):
    """Residual CNN stack."""

    def __init__(self, kernel_sizes: list[int], filters: int, drop_rate: float) -> None:
        """Initialize ResCNNStack.

        Args:
            kernel_sizes: List of kernel sizes for residual blocks.
            filters: Number of filters.
            drop_rate: Dropout rate.
        """
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual stack.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        for member in self.members:
            x = member(x)
        return x


class ResCNNBlock(nn.Module):
    """Residual CNN block."""

    def __init__(self, filters: int, ker: int, drop_rate: float) -> None:
        """Initialize ResCNNBlock.

        Args:
            filters: Number of filters.
            ker: Kernel size.
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection.
        """
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class SpatialDropout1d(nn.Module):
    """1D spatial dropout layer."""

    def __init__(self, drop_rate: float) -> None:
        """Initialize SpatialDropout1d.

        Args:
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial dropout.

        Args:
            x: Input tensor.

        Returns:
            Tensor with spatial dropout applied.
        """
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x
