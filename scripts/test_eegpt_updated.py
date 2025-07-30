#!/usr/bin/env python3
"""Test updated EEGPT model with normalization."""

from pathlib import Path

import numpy as np

from brain_go_brrr.core.config import ModelConfig
from brain_go_brrr.models.eegpt_model import EEGPTModel


def test_updated_model():
    """Test EEGPT with normalization wrapper."""
    print("=== TESTING UPDATED EEGPT MODEL ===\n")

    # Create config
    config = ModelConfig()
    config.model_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    # Initialize model
    model = EEGPTModel(config=config)
    model.load_model()

    # Create test signals (19 channels, 1024 samples)
    t = np.arange(1024) / 256  # 4 seconds at 256 Hz

    # Different patterns
    alpha = np.sin(2 * np.pi * 10 * t) * 50e-6  # 10 Hz
    beta = np.sin(2 * np.pi * 25 * t) * 30e-6  # 25 Hz

    # Multi-channel
    signal1 = np.tile(alpha, (19, 1))
    signal2 = np.tile(beta, (19, 1))

    # Channel names
    channel_names = [f"EEG{i:03d}" for i in range(19)]

    # Extract features
    print("Extracting features...")
    features1 = model.extract_features(signal1, channel_names)
    features2 = model.extract_features(signal2, channel_names)

    print(f"\nFeature shapes: {features1.shape}")
    print(f"Feature 1 norm: {np.linalg.norm(features1):.3f}")
    print(f"Feature 2 norm: {np.linalg.norm(features2):.3f}")

    # Calculate similarity
    feat1_flat = features1.flatten()
    feat2_flat = features2.flatten()
    cosine_sim = np.dot(feat1_flat, feat2_flat) / (
        np.linalg.norm(feat1_flat) * np.linalg.norm(feat2_flat)
    )

    print(f"\nCosine similarity: {cosine_sim:.6f}")

    if cosine_sim < 0.9:
        print("✅ SUCCESS: Features are discriminative!")
    else:
        print("❌ PROBLEM: Features are not discriminative")

    # Test with real-like multi-channel data
    print("\n=== Testing with realistic multi-channel data ===")

    # Different signals per channel
    realistic_data = np.zeros((19, 1024))
    for i in range(19):
        freq = 8 + i * 0.5  # Different frequency per channel
        realistic_data[i] = np.sin(2 * np.pi * freq * t) * (30 + i) * 1e-6

    # Add some noise
    noise_data = realistic_data + np.random.randn(19, 1024) * 20e-6

    # Extract features
    features_real = model.extract_features(realistic_data, channel_names)
    features_noise = model.extract_features(noise_data, channel_names)

    # Similarity
    real_flat = features_real.flatten()
    noise_flat = features_noise.flatten()
    real_noise_sim = np.dot(real_flat, noise_flat) / (
        np.linalg.norm(real_flat) * np.linalg.norm(noise_flat)
    )

    print(f"Realistic vs Noisy similarity: {real_noise_sim:.6f}")

    # Cleanup
    model.cleanup()
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_updated_model()
