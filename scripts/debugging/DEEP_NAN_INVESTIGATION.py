#!/usr/bin/env python3
"""DEEP FUCKING INVESTIGATION OF NAN CRASH - CHECK EVERY FUCKING THING"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt

print("=" * 80)
print("DEEP NAN INVESTIGATION - CHECKING EVERY FUCKING COMPONENT")
print("=" * 80)

# 1. CHECK DATASET OUTPUT
print("\n1. CHECKING DATASET OUTPUT:")
print("-" * 40)

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

# Check first 10 samples
for i in range(min(10, len(dataset))):
    x, y = dataset[i]
    print(f"\nSample {i}:")
    print(f"  Shape: {x.shape}")
    print(f"  Label: {y}")
    print(f"  Min: {x.min():.6f}, Max: {x.max():.6f}")
    print(f"  Mean: {x.mean():.6f}, Std: {x.std():.6f}")
    print(f"  Has NaN: {torch.isnan(x).any()}")
    print(f"  Has Inf: {torch.isinf(x).any()}")

    # Check for extreme values
    if x.abs().max() > 1000:
        print("  WARNING: EXTREME VALUES DETECTED!")

    # Check channel-wise stats
    channel_means = x.mean(dim=1)
    channel_stds = x.std(dim=1)
    if (channel_stds < 1e-6).any():
        print(f"  WARNING: Near-zero std channels: {torch.where(channel_stds < 1e-6)[0].tolist()}")

# 2. CHECK CHANNEL ADAPTER
print("\n\n2. CHECKING CHANNEL ADAPTER:")
print("-" * 40)

probe = EEGPTTwoLayerProbe(
    backbone_dim=768, n_input_channels=19, n_classes=2, use_channel_adapter=True
)

# Test channel adapter with various inputs
test_inputs = [
    torch.randn(1, 19, 2048) * 0.1,  # Normal scale
    torch.randn(1, 19, 2048) * 100,  # Large scale
    torch.randn(1, 19, 2048) * 0.001,  # Small scale
    torch.zeros(1, 19, 2048),  # All zeros
]

for i, test_x in enumerate(test_inputs):
    print(f"\nTest input {i}:")
    print(f"  Input scale: mean={test_x.mean():.6f}, std={test_x.std():.6f}")

    with torch.no_grad():
        adapted = probe.adapt_channels(test_x)

    print(f"  After adapter: shape={adapted.shape}")
    print(f"  Output scale: mean={adapted.mean():.6f}, std={adapted.std():.6f}")
    print(f"  Has NaN: {torch.isnan(adapted).any()}")
    print(f"  Has Inf: {torch.isinf(adapted).any()}")

# 3. CHECK NORMALIZATION IN WRAPPER
print("\n\n3. CHECKING EEGPT NORMALIZATION:")
print("-" * 40)

backbone = create_normalized_eegpt(
    checkpoint_path="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
)

print(f"Normalization enabled: {backbone.normalize}")
print(f"Input mean: {backbone.input_mean.item():.6f}")
print(f"Input std: {backbone.input_std.item():.6f}")
print(f"Stats source: {backbone._stats_source}")

# Test with real data
x, _ = dataset[0]
x_batch = x.unsqueeze(0)  # Add batch dimension

print("\nBefore normalization:")
print(f"  Data scale: mean={x_batch.mean():.6f}, std={x_batch.std():.6f}")

# Check what happens in forward pass
backbone.eval()
with torch.no_grad():
    # Check normalization step
    if backbone.normalize:
        x_norm = (x_batch - backbone.input_mean) / (backbone.input_std + 1e-8)
        print("\nAfter normalization:")
        print(f"  Normalized scale: mean={x_norm.mean():.6f}, std={x_norm.std():.6f}")
        print(f"  Has NaN: {torch.isnan(x_norm).any()}")
        print(f"  Has Inf: {torch.isinf(x_norm).any()}")

# 4. CHECK LOSS COMPUTATION
print("\n\n4. CHECKING LOSS COMPUTATION:")
print("-" * 40)

# Test CrossEntropyLoss with various logits
criterion = nn.CrossEntropyLoss()

test_scenarios = [
    ("Normal logits", torch.randn(32, 2)),
    ("Large logits", torch.randn(32, 2) * 100),
    ("Small logits", torch.randn(32, 2) * 0.001),
    ("Extreme positive", torch.tensor([[1000.0, 0.0]] * 32)),
    ("Extreme negative", torch.tensor([[-1000.0, 0.0]] * 32)),
]

labels = torch.randint(0, 2, (32,))

for name, logits in test_scenarios:
    print(f"\n{name}:")
    print(f"  Logits scale: min={logits.min():.3f}, max={logits.max():.3f}")

    # Check softmax
    probs = F.softmax(logits, dim=1)
    print(f"  Softmax range: min={probs.min():.6f}, max={probs.max():.6f}")

    # Check loss
    loss = criterion(logits, labels)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss is NaN: {torch.isnan(loss).item()}")
    print(f"  Loss is Inf: {torch.isinf(loss).item()}")

# 5. CHECK MIXED PRECISION BEHAVIOR
print("\n\n5. CHECKING MIXED PRECISION (fp16):")
print("-" * 40)

if torch.cuda.is_available():
    # Test with fp16
    x_fp16 = x_batch.half().cuda()
    probe_fp16 = probe.half().cuda()

    print(f"FP16 input range: min={x_fp16.min():.6f}, max={x_fp16.max():.6f}")

    with torch.no_grad():
        adapted_fp16 = probe_fp16.adapt_channels(x_fp16)
        print(f"FP16 adapted range: min={adapted_fp16.min():.6f}, max={adapted_fp16.max():.6f}")
        print(f"FP16 has NaN: {torch.isnan(adapted_fp16).any()}")
        print(f"FP16 has Inf: {torch.isinf(adapted_fp16).any()}")
else:
    print("CUDA not available for fp16 testing")

# 6. CHECK GRADIENT FLOW
print("\n\n6. CHECKING GRADIENT FLOW:")
print("-" * 40)

# Simple test of gradient flow through probe
probe_test = EEGPTTwoLayerProbe(
    backbone_dim=768,
    n_input_channels=19,
    n_classes=2,
)

# Dummy backbone output
backbone_features = torch.randn(4, 16, 768, requires_grad=True)  # [B, patches, dim]
logits = probe_test(backbone_features)
loss = criterion(logits, torch.randint(0, 2, (4,)))

print(f"Test loss: {loss.item():.6f}")
loss.backward()

# Check gradients
for name, param in probe_test.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.6f}, has_nan={torch.isnan(param.grad).any()}")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
