#!/usr/bin/env python3
"""Inspect EEGPT checkpoint structure to understand the model architecture."""

import json
from pathlib import Path

import torch


def inspect_checkpoint():
    """Inspect the EEGPT checkpoint structure."""
    checkpoint_path = (
        Path(__file__).parent.parent
        / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    )

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614

    print("\n=== Checkpoint Keys ===")
    for key in checkpoint:
        if isinstance(checkpoint[key], dict):
            print(f"{key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"{key}: tensor {checkpoint[key].shape}")
        else:
            print(f"{key}: {type(checkpoint[key])}")

    # Inspect model state dict
    if "state_dict" in checkpoint:
        print("\n=== Model Architecture (first 30 keys) ===")
        model_dict = checkpoint["state_dict"]
        for i, (key, value) in enumerate(model_dict.items()):
            if i >= 30:
                print(f"... and {len(model_dict) - 30} more keys")
                break
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")

    # Look for configuration
    if "config" in checkpoint:
        print("\n=== Model Configuration ===")
        config = checkpoint["config"]
        print(json.dumps(config, indent=2))

    # Look for architecture info
    if "arch" in checkpoint:
        print(f"\n=== Architecture: {checkpoint['arch']} ===")

    # Check for specific EEGPT components
    print("\n=== EEGPT Components ===")
    if "state_dict" in checkpoint:
        model_keys = list(checkpoint["state_dict"].keys())

        # Check for encoder
        encoder_keys = [k for k in model_keys if "encoder" in k]
        print(f"Encoder keys: {len(encoder_keys)}")

        # Check for patch embedding
        patch_embed_keys = [k for k in model_keys if "patch_embed" in k]
        print(f"Patch embedding keys: {len(patch_embed_keys)}")

        # Check for transformer blocks
        block_keys = [k for k in model_keys if "blocks" in k]
        print(f"Transformer block keys: {len(block_keys)}")

        # Check for heads
        head_keys = [k for k in model_keys if "head" in k]
        print(f"Head keys: {len(head_keys)}")

        # Check for position embeddings
        pos_embed_keys = [k for k in model_keys if "pos_embed" in k or "position" in k]
        print(f"Position embedding keys: {len(pos_embed_keys)}")


if __name__ == "__main__":
    inspect_checkpoint()
