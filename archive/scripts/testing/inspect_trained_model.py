#!/usr/bin/env python
"""Inspect the trained abnormality detection model."""

from pathlib import Path

import torch

print("=== INSPECTING TRAINED MODEL ===\n")

model_path = Path(
    "experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt"
)

if model_path.exists():
    print(f"Loading model from: {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024:.1f} KB\n")

    # Load with weights_only=False to handle numpy arrays
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    print("Checkpoint contents:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  - {key}")
            if key == "metrics":
                print(f"      {checkpoint[key]}")
            elif key == "epoch":
                print(f"      Epoch: {checkpoint[key]}")
            elif key == "model_state_dict" and isinstance(checkpoint[key], dict):
                print(f"      Layers: {len(checkpoint[key])} keys")
                # Show first few layer names
                for i, layer_key in enumerate(list(checkpoint[key].keys())[:5]):
                    print(f"        - {layer_key}: {checkpoint[key][layer_key].shape}")
    else:
        print(f"  Checkpoint is a {type(checkpoint)}")
        if hasattr(checkpoint, "keys"):
            for key in list(checkpoint.keys())[:10]:
                print(
                    f"    - {key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else type(checkpoint[key])}"
                )

    print("\n=== MODEL ARCHITECTURE ===")

    # Check if it's just the probe or includes EEGPT
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

        # Check for linear probe layers
        linear_layers = [
            k
            for k in state_dict.keys()
            if "linear" in k.lower()
            or "fc" in k.lower()
            or k.startswith("0.")
            or k.startswith("2.")
        ]
        if linear_layers:
            print(f"Linear probe layers found: {len(linear_layers)}")
            for layer in linear_layers[:5]:
                print(f"  - {layer}: {state_dict[layer].shape}")

        # Check for EEGPT backbone
        backbone_layers = [
            k for k in state_dict.keys() if "backbone" in k.lower() or "encoder" in k.lower()
        ]
        if backbone_layers:
            print(f"\nEEGPT backbone layers found: {len(backbone_layers)}")
        else:
            print("\nNo EEGPT backbone found (probe only)")

    print("\n=== TRAINING METRICS ===")
    if isinstance(checkpoint, dict) and "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                print(f"  - {metric_name}: {value}")

    print("\n=== READY FOR INTEGRATION ===")
    print("This model appears to be a trained linear probe for abnormality detection.")
    print("It should be loaded on top of the EEGPT backbone for inference.")

else:
    print(f"Model not found at: {model_path}")
