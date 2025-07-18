"""EEGPT Architecture Implementation.

Based on the official EEGPT paper and reference implementation.
Vision Transformer architecture adapted for EEG signals.
"""


import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# Standard 10-20 EEG channel mapping
CHANNEL_DICT = {
    'FP1': 0, 'FPZ': 1, 'FP2': 2,
    'AF7': 3, 'AF3': 4, 'AF4': 5, 'AF8': 6,
    'F7': 7, 'F5': 8, 'F3': 9, 'F1': 10, 'FZ': 11, 'F2': 12, 'F4': 13, 'F6': 14, 'F8': 15,
    'FT7': 16, 'FC5': 17, 'FC3': 18, 'FC1': 19, 'FCZ': 20, 'FC2': 21, 'FC4': 22, 'FC6': 23, 'FT8': 24,
    'T7': 25, 'T3': 25, 'C5': 26, 'C3': 27, 'C1': 28, 'CZ': 29, 'C2': 30, 'C4': 31, 'C6': 32, 'T8': 33, 'T4': 33,
    'TP7': 34, 'CP5': 35, 'CP3': 36, 'CP1': 37, 'CPZ': 38, 'CP2': 39, 'CP4': 40, 'CP6': 41, 'TP8': 42,
    'P7': 43, 'P5': 44, 'P3': 45, 'P1': 46, 'PZ': 47, 'P2': 48, 'P4': 49, 'P6': 50, 'P8': 51,
    'PO7': 52, 'PO3': 53, 'POZ': 54, 'PO4': 55, 'PO8': 56,
    'O1': 57, 'OZ': 58, 'O2': 59,
    'IZ': 60, 'FCZ_REF': 61
}


class RoPE(nn.Module):
    """Rotary Position Embedding implementation."""

    position: Tensor
    div_term: Tensor

    def __init__(self, embed_dim: int, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Create frequency matrix
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           -(math.log(10000.0) / embed_dim))

        # Register as buffer (non-parameter)
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning cosine and sine embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            
        Returns:
            Tuple of (cos_emb, sin_emb) each of shape (seq_len, embed_dim//2)
        """
        seq_len = x.size(1)
        position = self.position[:seq_len]  # (seq_len, 1)
        
        # Apply frequency terms: (seq_len, embed_dim//2)
        angles = position * self.div_term.unsqueeze(0)  # (seq_len, embed_dim//2)
        
        cos_emb = torch.cos(angles)  # (seq_len, embed_dim//2)
        sin_emb = torch.sin(angles)  # (seq_len, embed_dim//2)
        
        return cos_emb, sin_emb


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper function for rotary embeddings."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    # Ensure cos and sin have the right shape for broadcasting
    # q, k shape: (batch, num_heads, seq_len, head_dim)
    # cos, sin shape: (1, seq_len, 1, head_dim) -> need (1, 1, seq_len, head_dim) for broadcasting
    cos = cos.transpose(1, 2)  # (1, 1, seq_len, head_dim)
    sin = sin.transpose(1, 2)  # (1, 1, seq_len, head_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        """Initialize multi-head attention.

        Args:
            dim: Input dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in QKV projections.
            attn_drop: Attention dropout rate.
            proj_drop: Output projection dropout rate.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Rotary embeddings
        self.rotary_emb = RoPE(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        cos_emb, sin_emb = self.rotary_emb(x)
        q, k = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,  # Fixed type annotation
        drop: float = 0.0
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,  # Fixed type annotation
        norm_layer: type[nn.Module] = nn.LayerNorm  # Fixed type annotation
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding for EEG signals."""

    def __init__(self, img_size: list[int] | None = None, patch_size: int = 64, in_chans: int = 1, embed_dim: int = 512):
        """Initialize patch embedding.

        Args:
            img_size: Input image size [channels, time_steps].
            patch_size: Size of each patch in samples.
            in_chans: Number of input channels.
            embed_dim: Embedding dimension.
        """
        super().__init__()
        if img_size is None:
            img_size = [58, 1024]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0], img_size[1] // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps = x.shape
        x = x.unsqueeze(1)  # Add channel dimension for conv2d
        x = self.proj(x)  # batch_size, embed_dim, channels, time_steps//patch_size
        x = x.permute(0, 2, 3, 1)  # batch_size, channels, time_steps//patch_size, embed_dim
        x = x.reshape(batch_size, -1, x.shape[-1])  # batch_size, channels*num_patches, embed_dim
        return x


class EEGTransformer(nn.Module):
    """EEG Transformer model."""

    def __init__(
        self,
        n_channels: list | None = None,
        patch_size: int = 64,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm  # Fixed type annotation
    ) -> None:
        super().__init__()
        self.n_channels = n_channels or list(range(58))  # Default to 58 channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

    def prepare_chan_ids(self, channel_names: list) -> Tensor:
        """Prepare channel IDs for the model."""
        # Map channel names to indices
        chan_ids = []
        for name in channel_names:
            if name in self.n_channels:
                chan_ids.append(self.n_channels.index(name))
            else:
                chan_ids.append(0)  # Default channel
        return torch.tensor(chan_ids, dtype=torch.long)

    def forward(self, x: Tensor, chan_ids: Tensor | None = None) -> Tensor:
        # Patch embedding
        x = self.patch_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x


def create_eegpt_model(checkpoint_path: str | None = None, **kwargs: Any) -> EEGTransformer:
    """Create EEGPT model and optionally load pretrained weights.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        **kwargs: Model configuration parameters

    Returns:
        EEGPT model
    """
    # Default configuration for large model
    default_config = {
        'img_size': [58, 1024],
        'patch_size': 64,
        'embed_dim': 512,
        'embed_num': 4,
        'depth': 8,
        'num_heads': 8,
        'mlp_ratio': 4.0,
    }
    default_config.update(kwargs)

    model = _init_eeg_transformer(**default_config)

    if checkpoint_path is not None:
        # Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract encoder weights
        encoder_state = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('encoder.'):
                encoder_state[k[8:]] = v  # Remove 'encoder.' prefix
            elif k.startswith('target_encoder.'):
                encoder_state[k[15:]] = v  # Remove 'target_encoder.' prefix

        model.load_state_dict(encoder_state, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    return model


def _init_eeg_transformer(**kwargs: Any) -> EEGTransformer:
    """Initialize EEG Transformer with proper argument handling."""
    # Extract known arguments
    known_args = {
        'n_channels': kwargs.get('n_channels'),
        'patch_size': kwargs.get('patch_size', 64),
        'embed_dim': kwargs.get('embed_dim', 768),
        'depth': kwargs.get('depth', 12),
        'num_heads': kwargs.get('num_heads', 12),
        'mlp_ratio': kwargs.get('mlp_ratio', 4.0),
        'qkv_bias': kwargs.get('qkv_bias', True),
        'drop_rate': kwargs.get('drop_rate', 0.0),
        'attn_drop_rate': kwargs.get('attn_drop_rate', 0.0),
        'norm_layer': kwargs.get('norm_layer', nn.LayerNorm)
    }

    # Filter out None values
    filtered_args = {k: v for k, v in known_args.items() if v is not None}

    return EEGTransformer(**filtered_args)
