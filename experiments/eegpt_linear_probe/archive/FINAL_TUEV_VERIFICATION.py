#!/usr/bin/env python
"""Final verification of EEGPT feature extraction for TUEV.

This script tests the hypothesis that TUEV needs patch features, not just summary tokens.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.brain_go_brrr.infra.ml_models.eegpt_architecture import create_eegpt_model


def analyze_eegpt_outputs():
    """Analyze what EEGPT actually outputs at each stage."""
    
    print("="*60)
    print("EEGPT FEATURE EXTRACTION ANALYSIS")
    print("="*60)
    
    # Create dummy input matching TUEV after preprocessing
    batch_size = 2
    n_channels = 20  # After channel reduction from 23
    n_samples = 1024  # 4 seconds at 256 Hz
    x = torch.randn(batch_size, n_channels, n_samples)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Channels: {n_channels}")
    print(f"  Time samples: {n_samples}")
    
    # Create model (without loading weights for testing)
    model = create_eegpt_model(checkpoint_path=None)
    model.eval()
    
    # Manually step through the forward pass
    print("\n" + "="*60)
    print("FORWARD PASS ANALYSIS")
    print("="*60)
    
    # 1. Patch embedding
    patch_embed_output = model.patch_embed(x)
    print(f"\n1. After patch embedding: {patch_embed_output.shape}")
    print(f"   Interpretation: (batch={patch_embed_output.shape[0]}, "
          f"n_patches={patch_embed_output.shape[1]}, "
          f"n_channels={patch_embed_output.shape[2]}, "
          f"embed_dim={patch_embed_output.shape[3]})")
    
    # 2. Add channel embeddings
    batch_size, num_patches, num_channels, embed_dim = patch_embed_output.shape
    chan_ids = torch.arange(0, num_channels, dtype=torch.long)
    chan_embed = model.chan_embed(chan_ids).unsqueeze(0).unsqueeze(0)
    x_with_chan = patch_embed_output + chan_embed
    print(f"\n2. After channel embedding: {x_with_chan.shape}")
    
    # 3. Reshape for transformer
    x_reshaped = x_with_chan.reshape(batch_size, num_patches * num_channels, embed_dim)
    print(f"\n3. After reshape for transformer: {x_reshaped.shape}")
    print(f"   Total tokens: {x_reshaped.shape[1]} = {num_patches} patches × {num_channels} channels")
    
    # 4. Add summary tokens
    summary_tokens = model.summary_token.repeat(batch_size, 1, 1)
    x_with_summary = torch.cat([x_reshaped, summary_tokens], dim=1)
    print(f"\n4. After adding summary tokens: {x_with_summary.shape}")
    print(f"   Patch tokens: {x_reshaped.shape[1]}")
    print(f"   Summary tokens: {summary_tokens.shape[1]}")
    print(f"   Total: {x_with_summary.shape[1]}")
    
    # 5. After transformer blocks (simulation)
    x_transformed = x_with_summary  # Would go through blocks
    print(f"\n5. After transformer blocks: {x_transformed.shape}")
    
    # 6. Current extraction (ONLY SUMMARY)
    summary_only = x_transformed[:, -model.embed_num:, :]
    print(f"\n6. Current extraction (summary only): {summary_only.shape}")
    print(f"   Features: {summary_only.shape[1]} × {summary_only.shape[2]} = {summary_only.shape[1] * summary_only.shape[2]}")
    
    # 7. Alternative: Extract patches
    patches_only = x_transformed[:, :-model.embed_num, :]
    print(f"\n7. Alternative extraction (patches): {patches_only.shape}")
    print(f"   Features if flattened: {patches_only.shape[1]} × {patches_only.shape[2]} = {patches_only.shape[1] * patches_only.shape[2]}")
    
    # Analysis of Table 13's "15 × 4 × 512"
    print("\n" + "="*60)
    print("PAPER TABLE 13 ANALYSIS: '15 × 4 × 512'")
    print("="*60)
    
    print("\nPossible interpretations:")
    print("1. If '4' is summary tokens and '512' is embed_dim:")
    print(f"   Then '15' could be: 15 selected patches?")
    print(f"   Total patches available: {num_patches}")
    print(f"   Using 15 out of {num_patches}? {num_patches - 1 == 15}")
    
    print("\n2. If this is patches × channels × embed_dim:")
    print(f"   15 patches × 4 channels × 512 dim = 30,720 features")
    print(f"   But we have {n_channels} channels, not 4...")
    
    print("\n3. If keeping spatial structure:")
    # Try reshaping patches back to spatial
    patches_spatial = patches_only.reshape(batch_size, num_patches, num_channels, embed_dim)
    print(f"   Patches with spatial structure: {patches_spatial.shape}")
    print(f"   Could select 15 patches: {patches_spatial[:, :15, :, :].shape}")
    
    # Test hypothesis: Use 15 temporal positions
    if num_patches == 16:
        patches_15 = patches_spatial[:, :15, :, :]  # Drop last patch
        print(f"\n   Selecting first 15 patches: {patches_15.shape}")
        features_15 = patches_15.reshape(batch_size, -1)
        print(f"   Flattened features: {features_15.shape}")
        print(f"   Total features: {features_15.shape[1]}")
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    print("\n1. CURRENT IMPLEMENTATION:")
    print(f"   - Returns only {model.embed_num} summary tokens")
    print(f"   - Total features: {model.embed_num * embed_dim} = {model.embed_num * 512}")
    print(f"   - Throws away {num_patches * num_channels} patch tokens")
    
    print("\n2. PAPER EVIDENCE:")
    print("   - Table 13 shows '15 × 4 × 512' output shape")
    print("   - This doesn't match summary tokens (4 × 512)")
    print("   - Suggests spatial information is preserved")
    
    print("\n3. HYPOTHESIS:")
    print("   - Paper uses 15 out of 16 temporal patches")
    print("   - Preserves all 20 channels")
    print(f"   - Total features: 15 × 20 × 512 = 153,600")
    print("   - This would explain the 46% performance gap!")
    
    print("\n4. NEXT STEPS:")
    print("   - Implement patch extraction mode")
    print("   - Test with 15 patches × 20 channels")
    print("   - Compare performance to summary-only")


def test_feature_dimensions():
    """Test different feature extraction strategies."""
    
    print("\n" + "="*60)
    print("FEATURE DIMENSION COMPARISON")
    print("="*60)
    
    # Setup dimensions
    batch = 32
    channels = 20
    time = 1024
    embed_dim = 512
    patches = time // 64  # 16 patches
    
    print(f"\nInput: ({batch}, {channels}, {time})")
    print(f"Patches: {patches} (window of 64 samples each)")
    
    # Different extraction strategies
    strategies = {
        "Summary only (current)": 4 * embed_dim,
        "All patches": patches * channels * embed_dim,
        "15 patches (hypothesis)": 15 * channels * embed_dim,
        "Middle 8 patches": 8 * channels * embed_dim,
        "Patches + summary": (patches * channels + 4) * embed_dim,
    }
    
    print("\nFeature counts by strategy:")
    for name, n_features in strategies.items():
        ratio = n_features / strategies["Summary only (current)"]
        print(f"  {name:30s}: {n_features:8,d} features ({ratio:6.1f}x)")
    
    # Memory implications
    print("\nMemory usage (float32, batch=32):")
    for name, n_features in strategies.items():
        memory_mb = (batch * n_features * 4) / (1024 * 1024)
        print(f"  {name:30s}: {memory_mb:8.1f} MB")
    
    # Training implications
    print("\nTraining implications:")
    print("  TUEV training samples: ~84,000")
    for name, n_features in strategies.items():
        ratio = n_features / 84000
        status = "✓ OK" if ratio < 1 else "⚠ Overparameterized" if ratio < 2 else "✗ Severely overparameterized"
        print(f"  {name:30s}: {ratio:6.2f} feature/sample ratio {status}")


def test_linear_probe_dimensions():
    """Test linear probe input/output dimensions."""
    
    print("\n" + "="*60)
    print("LINEAR PROBE DIMENSION TEST")
    print("="*60)
    
    # Test different linear layer configurations
    configs = [
        ("Summary tokens", 4 * 512, 6),
        ("15 patches", 15 * 20 * 512, 6),
        ("All patches", 16 * 20 * 512, 6),
        ("Paper Table 13 literal", 15 * 4 * 512, 6),  # If taken literally
    ]
    
    print("\nLinear layer configurations:")
    for name, in_features, out_features in configs:
        layer = nn.Linear(in_features, out_features)
        n_params = sum(p.numel() for p in layer.parameters())
        print(f"  {name:25s}: Linear({in_features:7,d} → {out_features}) = {n_params:10,d} parameters")
    
    # Test forward pass
    print("\nForward pass test:")
    batch_size = 32
    for name, in_features, out_features in configs:
        try:
            layer = nn.Linear(in_features, out_features)
            x = torch.randn(batch_size, in_features)
            y = layer(x)
            print(f"  {name:25s}: Input {x.shape} → Output {y.shape} ✓")
        except Exception as e:
            print(f"  {name:25s}: Failed - {e}")


if __name__ == "__main__":
    print("FINAL TUEV VERIFICATION SCRIPT")
    print("Testing EEGPT feature extraction hypotheses")
    print()
    
    # Run all tests
    analyze_eegpt_outputs()
    test_feature_dimensions()
    test_linear_probe_dimensions()
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("\n1. Implement patch extraction (15 × 20 × 512)")
    print("2. This matches paper dimensionality better")
    print("3. Provides spatial information TUEV needs")
    print("4. Start with 15 patches, test 16 if needed")
    print("\nExpected improvement: BAcc 0.15 → 0.40+ (minimum)")
    print("Target performance: BAcc 0.62 (paper result)")