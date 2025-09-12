#!/usr/bin/env python3
"""Quick test of SeizureTransformer with minimal data."""

import torch

from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer

# Test the model works
model = SeizureTransformer(
    in_channels=19,
    in_samples=15360,  # 60s @ 256Hz
    drop_rate=0.1,
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"SeizureTransformer parameters: {total_params:,}")

# Test forward pass
x = torch.randn(2, 19, 15360)  # Batch of 2
with torch.no_grad():
    out = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
print("\n✅ SeizureTransformer is working!")
print("Architecture includes:")
print("- Encoder (5 conv layers)")
print("- ResCNN stack (7 blocks)")
print("- Transformer (8 layers, 4 heads)")
print("- Decoder with skip connections")
