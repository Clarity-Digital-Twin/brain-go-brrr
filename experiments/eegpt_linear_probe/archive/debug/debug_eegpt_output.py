#!/usr/bin/env python
"""Debug EEGPT output shape."""

import sys
from pathlib import Path

import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper


def main():
    """Test EEGPT output shape."""

    # Load EEGPT
    checkpoint = "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    eegpt = EEGPTWrapper(checkpoint_path=checkpoint)
    eegpt.model.eval()

    # Create dummy input (batch=2, channels=20, time=1024)
    x = torch.randn(2, 20, 1024)

    print("Input shape:", x.shape)
    print()

    # Get features
    with torch.no_grad():
        features = eegpt.extract_features(x)

    print("Output shape:", features.shape)
    print("Expected shape: (2, 16, 4, 512) for 16 patches")
    print()

    # If it's flattened
    if len(features.shape) == 2:
        print("Features are already flattened!")
        print(f"Feature dim: {features.shape[1]}")
    elif len(features.shape) == 4:
        print("Features have patch structure")
        print(f"Patches: {features.shape[1]}, Tokens: {features.shape[2]}, Dim: {features.shape[3]}")
        flattened = features.view(features.size(0), -1)
        print(f"Flattened shape: {flattened.shape}")


if __name__ == "__main__":
    main()
