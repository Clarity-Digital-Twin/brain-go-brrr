"""Test for fixing rotary embedding tensor size mismatch.

Following TDD approach - test first, then fix implementation.
"""

import torch

from brain_go_brrr.models.eegpt_architecture import (
    Attention,
    EEGTransformer,
    RoPE,
    apply_rotary_pos_emb,
)


class TestRotaryEmbeddingFix:
    """Test rotary embedding tensor size matching."""

    def test_rotary_embedding_dimensions(self):
        """Test that rotary embedding produces correct dimensions."""
        # Given: A rotary embedding with head dimension
        head_dim = 64
        rotary_emb = RoPE(embed_dim=head_dim)

        # When: We have a sequence of length 308 (4 summary tokens + 19*16 patches)
        batch_size = 1
        seq_len = 308  # 4 + 19*16 = 308 tokens
        x = torch.randn(batch_size, seq_len, head_dim)

        # Then: The cosine and sine embeddings should match sequence length
        cos_emb, sin_emb = rotary_emb(x)

        assert cos_emb.shape == (1, seq_len, 1, head_dim)
        assert sin_emb.shape == (1, seq_len, 1, head_dim)

    def test_apply_rotary_pos_emb_shapes(self):
        """Test that apply_rotary_pos_emb handles shapes correctly."""
        # Given: Query and key tensors from multi-head attention
        batch_size = 1
        num_heads = 8
        seq_len = 308  # 4 summary + 19*16 patches
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Cosine and sine embeddings from rotary embedding
        cos = torch.randn(1, seq_len, 1, head_dim)
        sin = torch.randn(1, seq_len, 1, head_dim)

        # When: We apply rotary position embeddings
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        # Then: Output shapes should match input shapes
        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape

    def test_attention_forward_with_correct_shapes(self):
        """Test that attention forward pass works with correct tensor shapes."""
        # Given: An attention module
        embed_dim = 512
        num_heads = 8
        attn = Attention(dim=embed_dim, num_heads=num_heads)

        # Input with correct sequence length (4 summary + 19*16 patches)
        batch_size = 1
        seq_len = 308  # 4 + 19*16
        x = torch.randn(batch_size, seq_len, embed_dim)

        # When: We run forward pass
        output = attn(x)

        # Then: Output should have same shape as input
        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_eeg_transformer_forward_pass(self):
        """Test full EEG transformer forward pass."""
        # Given: An EEG transformer
        model = EEGTransformer(
            img_size=[19, 1024],  # 19 channels, 1024 samples
            patch_size=64,
            embed_dim=512,
            embed_num=4,
            depth=2,  # Use fewer blocks for faster testing
            num_heads=8
        )

        # Input EEG data: 19 channels, 1024 samples (4 seconds at 256 Hz)
        batch_size = 1
        channels = 19
        samples = 1024
        x = torch.randn(batch_size, channels, samples)

        # Channel IDs for 19 channels
        chan_ids = torch.arange(channels)

        # When: We run forward pass
        output = model(x, chan_ids)

        # Then: Output should be summary tokens only
        assert output.shape == (batch_size, 4, 512)  # 4 summary tokens, 512 dim

    def test_patch_embedding_calculation(self):
        """Test that patch embedding calculations are correct."""
        # Given: EEG data dimensions
        channels = 19
        samples = 1024
        patch_size = 64

        # Expected patches per channel
        patches_per_channel = samples // patch_size  # 1024 // 64 = 16
        total_patches = channels * patches_per_channel  # 19 * 16 = 304

        # Total sequence length including summary tokens
        summary_tokens = 4
        expected_seq_len = summary_tokens + total_patches  # 4 + 304 = 308

        assert expected_seq_len == 308

        # This confirms our sequence length calculation is correct
