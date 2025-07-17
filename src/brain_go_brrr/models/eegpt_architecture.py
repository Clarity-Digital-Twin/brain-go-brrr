"""EEGPT Architecture Implementation.

Based on the official EEGPT paper and reference implementation.
Vision Transformer architecture adapted for EEG signals.
"""


import torch
import torch.nn as nn

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


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings for temporal encoding."""

    def __init__(self, dim: int, max_seq_len: int = 5000):
        """Initialize rotary position embeddings.
        
        Args:
            dim: Embedding dimension.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, :, None, :]
        sin_emb = emb.sin()[None, :, None, :]
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
        self.rotary_emb = RotaryEmbedding(head_dim)

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
    """Feed-forward network."""

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding for EEG signals."""

    def __init__(self, img_size: list[int] | None = None, patch_size: int = 64, in_chans: int = 1, embed_dim: int = 512):
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
    """EEGPT Encoder - Vision Transformer for EEG."""

    def __init__(self, img_size: list[int] | None = None, patch_size: int = 64, in_chans: int = 1,
                 embed_dim: int = 512, embed_num: int = 4, depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, return_all_tokens: bool = False):
        super().__init__()
        if img_size is None:
            img_size = [58, 1024]
        self.return_all_tokens = return_all_tokens
        self.embed_num = embed_num
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Channel embedding
        self.chan_embed = nn.Embedding(62, embed_dim)  # 62 possible channels

        # Summary tokens
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize summary token
        torch.nn.init.normal_(self.summary_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_chan_ids(self, channel_names: list[str]) -> torch.Tensor:
        """Convert channel names to IDs."""
        chan_ids = []
        for ch in channel_names:
            ch_upper = ch.upper()
            if ch_upper in CHANNEL_DICT:
                chan_ids.append(CHANNEL_DICT[ch_upper])
            else:
                # Default to FCZ_REF for unknown channels
                chan_ids.append(61)
        return torch.tensor(chan_ids, dtype=torch.long)

    def forward(self, x: torch.Tensor, chan_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, n_channels, n_samples)
            chan_ids: Channel IDs tensor (n_channels,)

        Returns:
            Features tensor
        """
        batch_size, channels, time_steps = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # B, C*num_patches, embed_dim

        # Add channel embeddings
        if chan_ids is not None:
            chan_embeds = self.chan_embed(chan_ids)  # C, embed_dim
            # Expand and add to each patch
            num_patches_per_chan = x.shape[1] // channels
            chan_embeds = chan_embeds.unsqueeze(1).expand(-1, num_patches_per_chan, -1)  # C, num_patches, embed_dim
            chan_embeds = chan_embeds.reshape(-1, self.embed_dim)  # C*num_patches, embed_dim
            x = x + chan_embeds.unsqueeze(0)

        # Prepend summary tokens
        summary_tokens = self.summary_token.expand(batch_size, -1, -1)
        x = torch.cat([summary_tokens, x], dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.return_all_tokens:
            return x
        else:
            # Return only summary tokens
            return x[:, :self.embed_num]


def create_eegpt_model(checkpoint_path: str | None = None, **kwargs) -> EEGTransformer:
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

    model = EEGTransformer(**default_config)

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
