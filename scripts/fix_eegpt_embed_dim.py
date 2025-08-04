#!/usr/bin/env python3
"""Fix EEGPT embed_dim mismatch based on paper specifications.

According to EEGPT paper Table 6:
- large model: embed_dim=512, layers=8/8/8, S=4, params=101M
- Our checkpoint: eegpt_mcae_58chs_4s_large4E.ckpt confirms this

The fix: Ensure model initialization matches checkpoint.
"""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_go_brrr  # noqa: E402.models.eegpt_architecture import EEGTransformer, create_eegpt_model


def verify_checkpoint_architecture():
    """Verify checkpoint matches paper specifications."""
    print("=== VERIFYING EEGPT CHECKPOINT ARCHITECTURE ===\n")

    checkpoint_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Analyze architecture from state dict
    print("\nAnalyzing checkpoint architecture...")

    # Get embedding dimension from summary token shape
    summary_token_shape = checkpoint["state_dict"]["encoder.summary_token"].shape
    print(f"Summary token shape: {summary_token_shape}")
    embed_dim = summary_token_shape[2]  # Shape is [1, 4, embed_dim]
    n_summary_tokens = summary_token_shape[1]

    # Count transformer blocks
    encoder_blocks = [k for k in checkpoint["state_dict"] if "encoder.blocks." in k]
    block_indices = {int(k.split(".")[2]) for k in encoder_blocks if k.split(".")[2].isdigit()}
    n_blocks = len(block_indices)

    print("\nDetected architecture:")
    print(f"  embed_dim: {embed_dim}")
    print(f"  n_summary_tokens: {n_summary_tokens}")
    print(f"  n_transformer_blocks: {n_blocks}")

    # Verify against paper
    print("\nPaper specifications (Table 6, large model):")
    print("  embed_dim: 512")
    print("  S (summary tokens): 4")
    print("  layers: 8/8/8")

    if embed_dim == 512 and n_summary_tokens == 4 and n_blocks == 8:
        print("\n✅ Checkpoint matches paper specifications!")
    else:
        print("\n❌ Checkpoint does NOT match paper!")

    return embed_dim, n_summary_tokens, n_blocks


def test_current_loading():
    """Test how model currently loads."""
    print("\n\n=== TESTING CURRENT MODEL LOADING ===\n")

    checkpoint_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    # Create model with current code
    print("Creating model with current code...")
    model = create_eegpt_model(str(checkpoint_path))

    # Check what was created
    print("\nModel architecture created:")
    print(f"  embed_dim: {model.embed_dim}")
    print(f"  embed_num: {model.embed_num}")
    print(f"  n_blocks: {len(model.blocks)}")

    # Check if summary tokens loaded correctly
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_tokens = checkpoint["state_dict"]["encoder.summary_token"]
    model_tokens = model.summary_token

    print("\nSummary token shapes:")
    print(f"  Checkpoint: {checkpoint_tokens.shape}")
    print(f"  Model: {model_tokens.shape}")

    if checkpoint_tokens.shape == model_tokens.shape:
        diff = (checkpoint_tokens - model_tokens.cpu()).norm()
        print(f"  Difference norm: {diff:.6f}")
        if diff < 0.01:
            print("  ✅ Summary tokens loaded correctly!")
        else:
            print("  ❌ Summary tokens shape matches but values differ!")
    else:
        print("  ❌ Shape mismatch - tokens cannot load!")


def create_fixed_model():
    """Create model with correct architecture."""
    print("\n\n=== CREATING FIXED MODEL ===\n")

    checkpoint_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    # The fix is already in create_eegpt_model default_config!
    # It correctly sets embed_dim=512
    # The issue must be elsewhere

    # Let's trace the exact loading process
    print("Tracing model creation and loading...")

    # Step 1: Create model with paper specs
    model = EEGTransformer(
        embed_dim=512,  # From paper Table 6
        embed_num=4,  # S=4 summary tokens
        depth=8,  # 8 transformer blocks
        num_heads=8,  # Inferred from embed_dim
        mlp_ratio=4.0,
        patch_size=64,  # 250ms at 256Hz
    )

    print("Created model with:")
    print(f"  embed_dim: {model.embed_dim}")
    print(f"  summary_token shape: {model.summary_token.shape}")

    # Step 2: Load checkpoint with strict=True
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract encoder weights
    encoder_state = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("encoder."):
            encoder_state[k[8:]] = v  # Remove 'encoder.' prefix
        elif k.startswith("target_encoder."):
            # Use target_encoder weights if encoder weights missing
            new_key = k[15:]  # Remove 'target_encoder.' prefix
            if new_key not in encoder_state:
                encoder_state[new_key] = v

    print(f"\nLoading {len(encoder_state)} state dict entries...")

    # Try loading with strict=True to catch issues
    try:
        missing, unexpected = model.load_state_dict(encoder_state, strict=True)
        print("✅ Weights loaded successfully with strict=True!")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    except RuntimeError as e:
        print(f"❌ Error with strict=True: {e}")
        print("\nTrying with strict=False to identify issues...")
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  First few missing: {missing[:5]}")
        if unexpected:
            print(f"  First few unexpected: {unexpected[:5]}")

    return model


def test_fixed_model_features():
    """Test if fixed model produces discriminative features."""
    print("\n\n=== TESTING FIXED MODEL FEATURES ===\n")

    # Create and load model properly
    model = create_fixed_model()
    model.eval()

    # Generate test data
    import numpy as np

    # Different patterns
    alpha = np.sin(2 * np.pi * 10 * np.arange(1024) / 256) * 50e-6
    beta = np.sin(2 * np.pi * 25 * np.arange(1024) / 256) * 50e-6

    # Multi-channel
    alpha_multi = np.tile(alpha, (19, 1))
    beta_multi = np.tile(beta, (19, 1))

    # Prepare as patches
    n_channels = 19
    patch_size = 64
    n_patches = 1024 // patch_size

    def prepare_input(data):
        # Reshape to patches
        data_patched = data.reshape(n_channels, n_patches, patch_size)
        data_rearranged = np.transpose(data_patched, (1, 0, 2))
        data_flattened = data_rearranged.reshape(-1, patch_size)
        return torch.FloatTensor(data_flattened).unsqueeze(0)

    alpha_input = prepare_input(alpha_multi)
    beta_input = prepare_input(beta_multi)

    # Extract features
    with torch.no_grad():
        feat_alpha = model(alpha_input).squeeze(0).numpy()
        feat_beta = model(beta_input).squeeze(0).numpy()

    print(f"Feature shapes: {feat_alpha.shape}")

    # Check discrimination
    cos_sim = np.dot(feat_alpha.flatten(), feat_beta.flatten()) / (
        np.linalg.norm(feat_alpha.flatten()) * np.linalg.norm(feat_beta.flatten())
    )

    print(f"\nCosine similarity between alpha and beta: {cos_sim:.6f}")

    if cos_sim < 0.9:
        print("✅ Features are discriminative!")
    else:
        print("❌ Features are NOT discriminative!")


def main():
    """Run all verification and fixes."""
    # Step 1: Verify checkpoint architecture
    embed_dim, n_summary_tokens, n_blocks = verify_checkpoint_architecture()

    # Step 2: Test current loading
    test_current_loading()

    # Step 3: Create and test fixed model
    test_fixed_model_features()

    print("\n\n=== SUMMARY ===")
    print("\nThe issue is that create_eegpt_model already sets embed_dim=512 correctly!")
    print("The problem must be in the weight loading process with strict=False")
    print("which silently ignores mismatches.")

    print("\n🔧 THE FIX:")
    print("In eegpt_architecture.py, change line 484:")
    print("  FROM: model.load_state_dict(encoder_state, strict=False)")
    print("  TO:   model.load_state_dict(encoder_state, strict=True)")
    print("\nThis will reveal any shape mismatches immediately!")


if __name__ == "__main__":
    main()
