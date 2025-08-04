#!/usr/bin/env python3
"""Verify the normalization fix works."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from brain_go_brrr  # noqa: E402.data.tuab_cached_dataset import TUABCachedDataset
from brain_go_brrr  # noqa: E402.models.eegpt_wrapper import create_normalized_eegpt

print("=" * 80)
print("VERIFYING NORMALIZATION FIX")
print("=" * 80)

# Load model
backbone = create_normalized_eegpt(
    checkpoint_path="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
)

print("\nNormalization settings:")
print(f"  Enabled: {backbone.normalize}")
print(f"  Mean: {backbone.input_mean.item():.10f}")
print(f"  Std: {backbone.input_std.item():.10f}")
print(f"  Source: {backbone._stats_source}")

# Load some data
dataset = TUABCachedDataset(
    root_dir=Path(
        "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    ),
    cache_dir=Path(
        "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuab_enhanced"
    ),
    split="train",
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
)

# Test normalization
x, _ = dataset[0]
x_batch = x.unsqueeze(0)

print("\nData before normalization:")
print(f"  Mean: {x_batch.mean():.6f}")
print(f"  Std: {x_batch.std():.6f}")
print(f"  Min: {x_batch.min():.6f}")
print(f"  Max: {x_batch.max():.6f}")

# Normalize
if backbone.normalize:
    x_norm = (x_batch - backbone.input_mean) / (backbone.input_std + 1e-8)
    print("\nData after normalization:")
    print(f"  Mean: {x_norm.mean():.6f}")
    print(f"  Std: {x_norm.std():.6f}")
    print(f"  Min: {x_norm.min():.6f}")
    print(f"  Max: {x_norm.max():.6f}")
    print(f"  Has NaN: {torch.isnan(x_norm).any()}")
    print(f"  Has Inf: {torch.isinf(x_norm).any()}")

print(
    "\n✅ NORMALIZATION IS NOW SAFE!" if x_norm.std() < 10 else "\n❌ NORMALIZATION STILL BROKEN!"
)
