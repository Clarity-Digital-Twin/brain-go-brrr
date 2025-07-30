#!/usr/bin/env python3
"""Debug EEGPT inference to understand why features aren't discriminative."""

from pathlib import Path

import numpy as np
import torch

from brain_go_brrr.models.eegpt_architecture import create_eegpt_model


def debug_inference():
    """Debug step by step through EEGPT inference."""
    print("=== DEBUGGING EEGPT INFERENCE ===\n")

    # Load model
    checkpoint_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
    model = create_eegpt_model(str(checkpoint_path))
    model.eval()

    # Create very different inputs
    t = np.arange(1024) / 256  # 4 seconds

    # Alpha rhythm (10 Hz) - strong oscillation
    alpha = np.sin(2 * np.pi * 10 * t) * 50e-6
    alpha_multi = np.tile(alpha, (19, 1))

    # White noise - random
    noise = np.random.randn(19, 1024) * 50e-6

    # Convert to torch
    alpha_input = torch.FloatTensor(alpha_multi).unsqueeze(0)
    noise_input = torch.FloatTensor(noise).unsqueeze(0)

    print(f"Input shapes: {alpha_input.shape}")
    print(f"Alpha input range: [{alpha_input.min():.6f}, {alpha_input.max():.6f}]")
    print(f"Noise input range: [{noise_input.min():.6f}, {noise_input.max():.6f}]")

    # Forward pass with hooks to capture intermediate values
    activations = {}

    def get_activation(name):
        def hook(model, input, output):  # noqa: ARG001
            activations[name] = output.detach()

        return hook

    # Register hooks
    model.patch_embed.register_forward_hook(get_activation("patch_embed"))
    model.blocks[0].register_forward_hook(get_activation("block_0"))
    model.blocks[-1].register_forward_hook(get_activation("block_last"))

    # Run inference
    with torch.no_grad():
        alpha_features = model(alpha_input)
        alpha_acts = {k: v.clone() for k, v in activations.items()}

        noise_features = model(noise_input)
        noise_acts = {k: v.clone() for k, v in activations.items()}

    print("\n=== LAYER OUTPUTS ===")

    # Check patch embedding
    print("\nPatch embed output:")
    print(f"  Alpha shape: {alpha_acts['patch_embed'].shape}")
    print(f"  Alpha norm: {alpha_acts['patch_embed'].norm():.3f}")
    print(f"  Noise norm: {noise_acts['patch_embed'].norm():.3f}")
    patch_sim = torch.nn.functional.cosine_similarity(
        alpha_acts["patch_embed"].flatten().unsqueeze(0),
        noise_acts["patch_embed"].flatten().unsqueeze(0),
    ).item()
    print(f"  Cosine similarity: {patch_sim:.6f}")

    # Check first block
    print("\nFirst block output:")
    print(f"  Alpha norm: {alpha_acts['block_0'].norm():.3f}")
    print(f"  Noise norm: {noise_acts['block_0'].norm():.3f}")
    block0_sim = torch.nn.functional.cosine_similarity(
        alpha_acts["block_0"].flatten().unsqueeze(0), noise_acts["block_0"].flatten().unsqueeze(0)
    ).item()
    print(f"  Cosine similarity: {block0_sim:.6f}")

    # Check last block
    print("\nLast block output:")
    print(f"  Alpha norm: {alpha_acts['block_last'].norm():.3f}")
    print(f"  Noise norm: {noise_acts['block_last'].norm():.3f}")
    block_last_sim = torch.nn.functional.cosine_similarity(
        alpha_acts["block_last"].flatten().unsqueeze(0),
        noise_acts["block_last"].flatten().unsqueeze(0),
    ).item()
    print(f"  Cosine similarity: {block_last_sim:.6f}")

    # Check final features
    print("\n=== FINAL FEATURES ===")
    print(f"Alpha features shape: {alpha_features.shape}")
    print(f"Alpha features norm: {alpha_features.norm():.3f}")
    print(f"Noise features norm: {noise_features.norm():.3f}")

    # Check each summary token
    for i in range(4):
        token_sim = torch.nn.functional.cosine_similarity(
            alpha_features[:, i, :], noise_features[:, i, :]
        ).item()
        print(f"Summary token {i} similarity: {token_sim:.6f}")

    # Overall similarity
    overall_sim = torch.nn.functional.cosine_similarity(
        alpha_features.flatten().unsqueeze(0), noise_features.flatten().unsqueeze(0)
    ).item()
    print(f"\nOverall feature similarity: {overall_sim:.6f}")

    if overall_sim > 0.95:
        print("\n❌ PROBLEM: Features are NOT discriminative!")
        print("   The model outputs nearly identical features for different inputs.")
        print("   This suggests weights may not be loading correctly or")
        print("   the architecture doesn't match the checkpoint.")
    else:
        print("\n✅ SUCCESS: Features are discriminative!")


if __name__ == "__main__":
    debug_inference()
