#!/usr/bin/env python3
"""Create a minimal test checkpoint for CI/CD testing.

This creates a small checkpoint with the correct structure but minimal weights
to speed up CI tests.
"""

from pathlib import Path

import torch


def create_minimal_checkpoint():
    """Create a minimal EEGPT checkpoint for testing."""
    # Model configuration (matching paper specs)
    embed_dim = 512
    # num_heads = 8  # Not used directly, but part of model spec
    depth = 8
    embed_num = 4  # Summary tokens
    # max_channels = 62  # Not used directly, but documents checkpoint format

    # Create state dict with minimal weights
    state_dict = {}

    # Summary token
    state_dict["encoder.summary_token"] = torch.randn(1, embed_num, embed_dim) * 0.02

    # Channel embedding (62 embeddings, not 63)
    state_dict["encoder.chan_embed.weight"] = torch.randn(62, embed_dim) * 0.02

    # Patch embedding
    state_dict["encoder.patch_embed.proj.weight"] = torch.randn(embed_dim, 1, 1, 64) * 0.02
    state_dict["encoder.patch_embed.proj.bias"] = torch.randn(embed_dim) * 0.02

    # Transformer blocks (all blocks for compatibility)
    for i in range(depth):  # All 8 blocks
        prefix = f"encoder.blocks.{i}"

        # Attention
        state_dict[f"{prefix}.attn.qkv.weight"] = torch.randn(3 * embed_dim, embed_dim) * 0.02
        state_dict[f"{prefix}.attn.qkv.bias"] = torch.randn(3 * embed_dim) * 0.02
        state_dict[f"{prefix}.attn.proj.weight"] = torch.randn(embed_dim, embed_dim) * 0.02
        state_dict[f"{prefix}.attn.proj.bias"] = torch.randn(embed_dim) * 0.02

        # MLP
        mlp_dim = embed_dim * 4
        state_dict[f"{prefix}.mlp.fc1.weight"] = torch.randn(mlp_dim, embed_dim) * 0.02
        state_dict[f"{prefix}.mlp.fc1.bias"] = torch.randn(mlp_dim) * 0.02
        state_dict[f"{prefix}.mlp.fc2.weight"] = torch.randn(embed_dim, mlp_dim) * 0.02
        state_dict[f"{prefix}.mlp.fc2.bias"] = torch.randn(embed_dim) * 0.02

        # Norms
        state_dict[f"{prefix}.norm1.weight"] = torch.ones(embed_dim)
        state_dict[f"{prefix}.norm1.bias"] = torch.zeros(embed_dim)
        state_dict[f"{prefix}.norm2.weight"] = torch.ones(embed_dim)
        state_dict[f"{prefix}.norm2.bias"] = torch.zeros(embed_dim)

    # Final norm
    state_dict["encoder.norm.weight"] = torch.ones(embed_dim)
    state_dict["encoder.norm.bias"] = torch.zeros(embed_dim)

    # Create checkpoint
    checkpoint = {
        "epoch": 100,
        "global_step": 10000,
        "pytorch-lightning_version": "1.5.0",
        "state_dict": state_dict,
        "loops": {},
    }

    # Save
    output_dir = Path("tests/fixtures/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "eegpt_test_checkpoint.ckpt"
    torch.save(checkpoint, output_path)

    # Check size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Created test checkpoint: {output_path}")
    print(f"Size: {size_mb:.1f} MB (vs 1020 MB for full checkpoint)")

    # Also create normalization stats
    norm_stats = {"mean": 0.0, "std": 1.0, "computed_from": "test_data", "n_samples": 1000}

    import json

    norm_path = output_dir / "normalization.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Created normalization stats: {norm_path}")

    return output_path


if __name__ == "__main__":
    create_minimal_checkpoint()
