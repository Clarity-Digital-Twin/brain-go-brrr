"""SeizureTransformer architecture from Wu et al. 2025.

This is a self-contained implementation based on the reference paper.
The model is a Vision Transformer-based architecture optimized for
seizure detection on EEG signals.

Architecture:
- Input: (batch, 19 channels, 15360 samples) at 256Hz
- Patch embedding: 1D convolution with kernel_size=256 (1 second patches)
- Transformer encoder: 6 layers, 8 heads, 512 dim
- Output: Per-timestep seizure predictions

This implementation follows the SOLID principles and DRY pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert EEG signal into patch embeddings."""

    def __init__(
        self,
        in_channels: int = 19,
        patch_size: int = 256,  # 1 second at 256Hz
        embed_dim: int = 512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through patch embedding.

        Args:
            x: (batch, channels, samples)

        Returns:
            (batch, n_patches, embed_dim)
        """
        # x: (B, C, T) -> (B, E, N_patches)
        x = self.proj(x)
        # Transpose to (B, N_patches, E)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""

    def __init__(self, n_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        embed_dim: int = 512,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        # MLP block
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections."""
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class SeizureTransformer(nn.Module):
    """Transformer-based model for seizure detection.

    This implementation follows the architecture from Wu et al. 2025.
    """

    def __init__(
        self,
        in_channels: int = 19,
        in_samples: int = 15360,  # 60s at 256Hz
        patch_size: int = 256,  # 1s patches
        embed_dim: int = 512,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        """Initialize SeizureTransformer.

        Args:
            in_channels: Number of EEG channels (19 for standard 10-20)
            in_samples: Number of samples in input window (15360 for 60s@256Hz)
            patch_size: Size of each patch in samples (256 for 1s patches)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            drop_rate: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.patch_size = patch_size

        # Calculate number of patches
        self.n_patches = in_samples // patch_size  # 60 patches for 60s

        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)

        # Positional encoding
        self.pos_embed = PositionalEncoding(self.n_patches, embed_dim)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, drop_rate) for _ in range(depth)]
        )

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head - per-patch binary classification
        self.head = nn.Linear(embed_dim, patch_size)  # Predict for each sample in patch

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, samples)

        Returns:
            Per-sample predictions of shape (batch, samples)
        """
        b, c, t = x.shape

        # Validate input shape
        assert self.in_channels == c, f"Expected {self.in_channels} channels, got {c}"
        assert self.in_samples == t, f"Expected {self.in_samples} samples, got {t}"

        # Patch embedding: (B, C, T) -> (B, N_patches, E)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Predict for each patch: (B, N_patches, E) -> (B, N_patches, patch_size)
        x = self.head(x)

        # Reshape to per-sample predictions: (B, N_patches, patch_size) -> (B, T)
        x = x.reshape(b, -1)

        return x


class SeizureTransformerWithPretraining(SeizureTransformer):
    """Extended version with support for loading pretrained weights.

    This class adds utilities for loading weights from the reference
    implementation checkpoint format.
    """

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str, device: torch.device | None = None, **kwargs: any
    ) -> SeizureTransformerWithPretraining:
        """Load model with pretrained weights.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            **kwargs: Additional arguments for model initialization

        Returns:
            Model with loaded weights
        """
        # Initialize model with default or provided params
        model = cls(**kwargs)

        # Load checkpoint
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,  # nosec:weights_only - model weights contain custom objects
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        return model

    def freeze_backbone(self) -> None:
        """Freeze transformer backbone for fine-tuning."""
        # Freeze all except the prediction head
        for name, param in self.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# Factory function for backward compatibility
def build_seizure_transformer(
    n_channels: int = 19,
    pretrained_path: str | None = None,
) -> SeizureTransformer:
    """Build SeizureTransformer model.

    Args:
        n_channels: Number of input channels
        pretrained_path: Optional path to pretrained weights

    Returns:
        SeizureTransformer model instance
    """
    if pretrained_path:
        return SeizureTransformerWithPretraining.from_pretrained(
            pretrained_path, in_channels=n_channels
        )
    else:
        return SeizureTransformer(in_channels=n_channels)


__all__ = [
    "SeizureTransformer",
    "SeizureTransformerWithPretraining",
    "build_seizure_transformer",
]
