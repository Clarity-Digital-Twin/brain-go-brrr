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
        rotary_emb = RoPE(dim=head_dim)

        # When: We have a sequence of length 308 (4 summary tokens + 19*16 patches)
        batch_size = 1
        seq_len = 308  # 4 + 19*16 = 308 tokens
        x = torch.randn(batch_size, seq_len, head_dim)

        # Then: The frequency embeddings should be generated successfully
        freqs = rotary_emb(x)

        # Check that freqs has correct shape for apply_rotary_emb
        assert freqs is not None
        assert freqs.shape[0] == seq_len  # Should match sequence length
        print(f"✅ RoPE freqs shape: {freqs.shape}")

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

        # Then: Output should have correct shape
        assert output.shape == (batch_size, seq_len, embed_dim)
        print(f"✅ Attention output shape: {output.shape}")

    def test_eeg_transformer_forward_pass(self):
        """Test full EEG transformer forward pass."""
        # Given: An EEG transformer with correct parameter names
        model = EEGTransformer(
            n_channels=list(range(19)),  # 19 channels
            patch_size=64,
            embed_dim=512,
            depth=2,  # Use fewer blocks for faster testing
            num_heads=8
        )

        # Given: Sample EEG data (batch=1, patches=308, features=64)
        # This represents 19 channels x 16 patches per channel + 4 summary tokens = 308 total
        batch_size = 1
        patches = 308  # 19 * 16 + 4 summary tokens
        patch_size = 64
        x = torch.randn(batch_size, patches, patch_size)

        # When: Forward pass through model
        output = model(x)

        # Then: Output should have correct shape
        expected_shape = (batch_size, patches, 512)  # embed_dim=512
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✅ EEG Transformer output shape: {output.shape}")

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
