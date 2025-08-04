"""EEGPT Architecture Implementation.

Based on the official EEGPT paper and reference implementation.
Vision Transformer architecture adapted for EEG signals.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Standard 10-20 EEG channel mapping
CHANNEL_DICT = {
    "FP1": 0,
    "FPZ": 1,
    "FP2": 2,
    "AF7": 3,
    "AF3": 4,
    "AF4": 5,
    "AF8": 6,
    "F7": 7,
    "F5": 8,
    "F3": 9,
    "F1": 10,
    "FZ": 11,
    "F2": 12,
    "F4": 13,
    "F6": 14,
    "F8": 15,
    "FT7": 16,
    "FC5": 17,
    "FC3": 18,
    "FC1": 19,
    "FCZ": 20,
    "FC2": 21,
    "FC4": 22,
    "FC6": 23,
    "FT8": 24,
    "T7": 25,
    "T3": 25,
    "C5": 26,
    "C3": 27,
    "C1": 28,
    "CZ": 29,
    "C2": 30,
    "C4": 31,
    "C6": 32,
    "T8": 33,
    "T4": 33,
    "TP7": 34,
    "CP5": 35,
    "CP3": 36,
    "CP1": 37,
    "CPZ": 38,
    "CP2": 39,
    "CP4": 40,
    "CP6": 41,
    "TP8": 42,
    "P7": 43,
    "P5": 44,
    "P3": 45,
    "P1": 46,
    "PZ": 47,
    "P2": 48,
    "P4": 49,
    "P6": 50,
    "P8": 51,
    "PO7": 52,
    "PO3": 53,
    "POZ": 54,
    "PO4": 55,
    "PO8": 56,
    "O1": 57,
    "OZ": 58,
    "O2": 59,
    "IZ": 60,
    "FCZ_REF": 61,
}


# rotary embedding helper functions
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x = x.reshape((*x.shape[:-1], x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rotary_emb(
    freqs: torch.Tensor, t: torch.Tensor, start_index: int = 0, scale: float = 1.0
) -> torch.Tensor:
    """Apply rotary positional embeddings to a tensor."""
    freqs = freqs.to(t.device)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    if rot_dim > t.shape[-1]:
        raise ValueError(
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

    t_left, t_middle, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )

    # Apply rotary embeddings to the middle segment
    t_rotated_middle = (t_middle * freqs.cos() * scale) + (
        rotate_half(t_middle) * freqs.sin() * scale
    )

    return torch.cat((t_left, t_rotated_middle, t_right), dim=-1)


class RoPE(nn.Module):
    """Rotary Position Embedding implementation based on original EEGPT."""

    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 1024) -> None:
        """Initialize rotary position embeddings."""
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Initialize frequency parameters (matching original EEGPT)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs)
        self.cache: dict[str, torch.Tensor] = {}

    def prepare_freqs(
        self,
        num_patches: tuple[int, int],
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        offset: int = 0,
    ) -> torch.Tensor:
        """Prepare frequency embeddings for given number of patches."""
        c, n = num_patches
        cache_key = f"freqs:{num_patches}"

        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate sequence positions and apply offset
        seq_pos = torch.arange(n, device=device, dtype=dtype).repeat_interleave(repeats=c)
        seq_pos = seq_pos + offset

        # Compute outer product of positions and frequencies, then expand along the last dimension
        # Cast self.freqs to Tensor (it's registered as a buffer, so it's always a tensor)
        freqs_tensor = torch.as_tensor(self.freqs)
        freqs_scaled = torch.outer(seq_pos.to(freqs_tensor.dtype), freqs_tensor).repeat_interleave(
            repeats=2, dim=-1
        )

        # Cache and return the computed frequencies
        self.cache[cache_key] = freqs_scaled
        return freqs_scaled

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - prepare frequencies for the input."""
        batch_size, seq_len, embed_dim = x.shape
        # For EEGPT, we assume patches are arranged as (channels, patches_per_channel)
        # Simplified: treat seq_len as total patches
        return self.prepare_freqs((1, seq_len), device=str(x.device), dtype=x.dtype)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys - deprecated, use apply_rotary_emb instead."""
    # This is kept for backward compatibility but not used in the main implementation
    _ = cos  # Mark as intentionally unused
    _ = sin  # Mark as intentionally unused
    return q, k


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        """Initialize attention module."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Rotary embeddings for head dimension
        if self.use_rope:
            self.rotary_emb = RoPE(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with rotary position embeddings."""
        batch_size, seq_len, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings if enabled
        if self.use_rope:
            freqs = self.rotary_emb(x)  # Get frequency embeddings
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        # Efficient attention using Flash Attention
        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0,
            is_causal=False,
        )

        x = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
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
        drop: float = 0.0,
    ) -> None:
        """Initialize MLP block."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP layer."""
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
        norm_layer: type[nn.Module] = nn.LayerNorm,  # Fixed type annotation
    ) -> None:
        """Initialize transformer block."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rope=True,  # EEGPT uses RoPE for temporal encoding
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Transformer block with self-attention and MLP."""
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding for EEG signals."""

    def __init__(
        self,
        img_size: list[int] | None = None,
        patch_size: int = 64,
        in_chans: int = 1,
        embed_dim: int = 512,
    ):
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

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert EEG signal to patch embeddings.

        Args:
            x: Input of shape (B, C, T) where B=batch, C=channels, T=time

        Returns:
            Patches of shape (B, N, C, D) where N=num_patches, D=embed_dim
        """
        # x: B, C, T
        x = x.unsqueeze(1)  # B, 1, C, T
        x = self.proj(x)  # B, embed_dim, C, T//patch_size
        x = x.transpose(1, 3)  # B, T//patch_size, C, embed_dim
        # Return shape: (B, N, C, D)
        return x


class EEGTransformer(nn.Module):
    """EEG Transformer model."""

    def __init__(
        self,
        n_channels: list | None = None,
        patch_size: int = 64,
        embed_dim: int = 768,
        embed_num: int = 4,  # Number of summary tokens
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,  # Fixed type annotation
    ) -> None:
        """Initialize EEG Transformer model."""
        super().__init__()
        self.n_channels = n_channels or list(range(58))  # Default to 58 channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.embed_num = embed_num

        # Patch embedding using PatchEmbed module to match checkpoint
        self.patch_embed = PatchEmbed(
            img_size=[len(self.n_channels), 1024],  # channels x time_steps
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
        )

        # Channel embedding - size based on checkpoint (62 channels, 0-61)
        # The checkpoint has 62 embeddings (not 63)
        self.max_channel_id = 61  # Maximum channel ID is 61 (0-indexed)
        self.chan_embed = nn.Embedding(62, embed_dim)  # 62 total embeddings

        # Summary tokens (learnable parameters)
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        nn.init.normal_(self.summary_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

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
        """Forward pass through EEG Transformer encoder.

        Args:
            x: Input tensor of shape (B, C, T) where:
               B = batch size
               C = number of channels
               T = time steps (e.g., 1024 for 4 seconds at 256 Hz)
            chan_ids: Channel IDs for positional embedding (optional)

        Returns:
            Summary tokens of shape (B, embed_num, embed_dim)
        """
        # Input shape: (B, C, T)
        batch_size, n_channels, time_steps = x.shape

        # Validate input dimensions
        if time_steps % self.patch_size != 0:
            raise ValueError(
                f"Time dimension {time_steps} must be divisible by patch_size {self.patch_size}. "
                f"Expected multiple of {self.patch_size} samples."
            )

        # Patch embedding: (B, C, T) -> (B, N, C, D)
        x = self.patch_embed(x)
        batch_size, num_patches, num_channels, embed_dim = x.shape

        # Generate channel IDs if not provided
        if chan_ids is None:
            chan_ids = torch.arange(0, num_channels, device=x.device, dtype=torch.long)
        else:
            chan_ids = chan_ids.to(x.device).long()

        # Validate channel IDs
        max_chan_id = chan_ids.max().item()
        if max_chan_id > self.max_channel_id:
            raise ValueError(
                f"Channel ID {max_chan_id} exceeds maximum supported ID {self.max_channel_id}. "
                f"Model was trained with up to {self.max_channel_id} channels."
            )

        # Add channel positional embedding
        chan_embed = (
            self.chan_embed(chan_ids).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, num_channels, embed_dim)
        x = x + chan_embed  # Broadcast to (batch_size, num_patches, num_channels, embed_dim)

        # Reshape to sequence format for transformer
        x = x.reshape(
            batch_size, num_patches * num_channels, embed_dim
        )  # (batch_size, num_patches*num_channels, embed_dim)

        # Concatenate summary tokens
        summary_tokens = self.summary_token.repeat(batch_size, 1, 1)
        x = torch.cat([x, summary_tokens], dim=1)  # Add summary tokens at the end

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract only the summary tokens from the output
        x = x[:, -self.embed_num :, :]

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
        "img_size": [58, 1024],
        "patch_size": 64,
        "embed_dim": 512,
        "embed_num": 4,
        "depth": 8,
        "num_heads": 8,
        "mlp_ratio": 4.0,
    }
    default_config.update(kwargs)

    model = _init_eeg_transformer(**default_config)

    if checkpoint_path is not None:
        # Load pretrained weights
        # NOTE: weights_only=False needed for EEGPT checkpoint format compatibility
        # This is safe as we only load from trusted model checkpoints
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )  # nosec B614 - Loading pretrained EEGPT model weights from trusted source

        # Extract encoder weights
        encoder_state = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("encoder."):
                encoder_state[k[8:]] = v  # Remove 'encoder.' prefix
            elif k.startswith("target_encoder."):
                encoder_state[k[15:]] = v  # Remove 'target_encoder.' prefix

        # Load with strict=False first to handle buffers
        missing_keys, unexpected_keys = model.load_state_dict(encoder_state, strict=False)

        # Filter out expected missing keys (RoPE buffers are initialized, not loaded)
        filtered_missing = [k for k in missing_keys if not k.endswith(".rotary_emb.freqs")]

        if filtered_missing:
            logger.warning(f"Missing keys in checkpoint: {filtered_missing}")
            # If there are real missing keys, fail
            raise RuntimeError(f"Missing required keys: {filtered_missing}")

        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        logger.info(f"Loaded pretrained weights from {checkpoint_path}")

    return model


def _init_eeg_transformer(**kwargs: Any) -> EEGTransformer:
    """Initialize EEG Transformer with proper argument handling."""
    # Extract known arguments
    known_args = {
        "n_channels": kwargs.get("n_channels"),
        "patch_size": kwargs.get("patch_size", 64),
        "embed_dim": kwargs.get("embed_dim", 768),
        "embed_num": kwargs.get("embed_num", 4),  # Summary tokens
        "depth": kwargs.get("depth", 12),
        "num_heads": kwargs.get("num_heads", 12),
        "mlp_ratio": kwargs.get("mlp_ratio", 4.0),
        "qkv_bias": kwargs.get("qkv_bias", True),
        "drop_rate": kwargs.get("drop_rate", 0.0),
        "attn_drop_rate": kwargs.get("attn_drop_rate", 0.0),
        "norm_layer": kwargs.get("norm_layer", nn.LayerNorm),
    }

    # Filter out None values
    filtered_args = {k: v for k, v in known_args.items() if v is not None}

    return EEGTransformer(**filtered_args)
