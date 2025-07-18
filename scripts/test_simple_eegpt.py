#!/usr/bin/env python3
"""Simple test of EEGPT model functionality."""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brain_go_brrr.models.eegpt_model import EEGPTModel


def test_simple_eegpt():
    """Test basic EEGPT functionality."""
    print("🧠 Simple EEGPT Test")
    print("=" * 50)

    # Load model
    model_path = project_root / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    print("\n1️⃣ Loading EEGPT model...")
    model = EEGPTModel(checkpoint_path=model_path)
    print("✅ Model loaded successfully")
    print(f"   Device: {model.device}")
    print(f"   Config: {model.config}")

    # Test feature extraction
    print("\n2️⃣ Testing feature extraction...")

    # Create a 4-second window
    window = np.random.randn(19, 1024) * 30e-6  # 19 channels, 4s at 256Hz
    channel_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
    ]

    features = model.extract_features(window, channel_names)
    print("✅ Features extracted")
    print(f"   Shape: {features.shape}")
    print("   Expected: (4, 512)")  # 4 summary tokens, 512 dims

    # Test batch processing
    print("\n3️⃣ Testing batch processing...")
    batch = np.random.randn(5, 19, 1024) * 30e-6  # 5 windows

    batch_features = model.extract_features_batch(batch, channel_names)
    print("✅ Batch features extracted")
    print(f"   Shape: {batch_features.shape}")
    print("   Expected: (5, 4, 512)")

    # Test abnormality head
    print("\n4️⃣ Testing abnormality prediction...")

    # Get prediction for single window
    with torch.no_grad():
        features_flat = torch.FloatTensor(features.flatten()).unsqueeze(0).to(model.device)
        logits = model.abnormality_head(features_flat)
        probs = torch.softmax(logits, dim=-1)

    print("✅ Abnormality prediction")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Normal prob: {probs[0, 0]:.3f}")
    print(f"   Abnormal prob: {probs[0, 1]:.3f}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_simple_eegpt()
