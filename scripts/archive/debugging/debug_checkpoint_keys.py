#!/usr/bin/env python3
"""Debug what keys are in the EEGPT checkpoint."""

from pathlib import Path

import torch

checkpoint_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

if not checkpoint_path.exists():
    print(f"Checkpoint not found at {checkpoint_path}")
    exit(1)

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print(f"\nTop-level keys: {list(checkpoint.keys())}")
print(f"\nTotal state_dict keys: {len(checkpoint['state_dict'])}")

# Look for summary token related keys
summary_keys = [k for k in checkpoint["state_dict"] if "summary" in k.lower()]
print(f"\nSummary token keys: {summary_keys}")

# Look for encoder keys
encoder_keys = [k for k in checkpoint["state_dict"] if k.startswith("encoder.")]
print(f"\nEncoder keys (first 10): {encoder_keys[:10]}")

# Check if summary_token is in the loaded weights
if "encoder.summary_token" in checkpoint["state_dict"]:
    print("\n✅ Found encoder.summary_token in checkpoint!")
    print(f"Shape: {checkpoint['state_dict']['encoder.summary_token'].shape}")
else:
    print("\n❌ encoder.summary_token NOT found in checkpoint!")

# Look for any token-related keys
token_keys = [k for k in checkpoint["state_dict"] if "token" in k.lower()]
print(f"\nToken-related keys: {token_keys[:10]}")
