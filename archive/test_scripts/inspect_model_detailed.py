#!/usr/bin/env python
"""Detailed inspection of the trained model."""

import torch
from pathlib import Path

print("=== DETAILED MODEL INSPECTION ===\n")

model_path = Path("experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt")

checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

print("Performance Metrics:")
print(f"  - Validation AUROC: {checkpoint.get('val_auroc', 'N/A')}")
print(f"  - Validation Balanced Accuracy: {checkpoint.get('val_bacc', 'N/A')}")
print(f"  - Training Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  - Global Step: {checkpoint.get('global_step', 'N/A')}")

print("\nProbe Architecture:")
if "probe_state_dict" in checkpoint:
    probe_dict = checkpoint["probe_state_dict"]
    print(f"  Number of layers: {len(probe_dict)}")
    for key, tensor in probe_dict.items():
        print(f"  - {key}: {tensor.shape}")
        
print("\nConfiguration:")
if "config" in checkpoint:
    config = checkpoint["config"]
    if isinstance(config, dict):
        print(f"  - Model type: {config.get('model', {}).get('probe', {}).get('type', 'N/A')}")
        print(f"  - Input dim: {config.get('model', {}).get('probe', {}).get('input_dim', 'N/A')}")
        print(f"  - Hidden dim: {config.get('model', {}).get('probe', {}).get('hidden_dim', 'N/A')}")
        print(f"  - Output classes: {config.get('model', {}).get('probe', {}).get('n_classes', 'N/A')}")
        print(f"  - Dropout: {config.get('model', {}).get('probe', {}).get('dropout', 'N/A')}")
        print(f"  - Target AUROC: {config.get('target_metrics', {}).get('auroc', 'N/A')}")

print("\n=== INTEGRATION PATH ===")
print("To use this model:")
print("1. Load EEGPT backbone (frozen)")
print("2. Load this probe on top")
print("3. Run inference on 4-second windows")
print("4. Window size: 4s @ 256Hz = 1024 samples")
print("5. Expected AUROC: ~0.79 (based on training)")

# Check if we need the base EEGPT model
eegpt_path = Path("data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
if not eegpt_path.exists():
    print("\n⚠️ WARNING: EEGPT base model not found!")
    print(f"   Expected at: {eegpt_path}")
    print("   This is required for inference!")
else:
    print(f"\n✅ EEGPT base model found at: {eegpt_path}")