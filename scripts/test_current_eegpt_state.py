#!/usr/bin/env python3
"""Quick test to verify CURRENT EEGPT state - are features discriminative or not?"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_go_brrr.models.eegpt_model import EEGPTModel


def test_discrimination():
    """Test if EEGPT features discriminate between different patterns."""
    print("=== TESTING CURRENT EEGPT FEATURE DISCRIMINATION ===\n")

    # Load model
    model = EEGPTModel()
    if not model.is_loaded:
        model.load_model()

    # Generate test signals
    sampling_rate = 256
    duration = 4
    n_samples = sampling_rate * duration
    t = np.linspace(0, duration, n_samples)

    # Different patterns
    alpha = np.sin(2 * np.pi * 10 * t) * 50e-6  # 10 Hz
    beta = np.sin(2 * np.pi * 25 * t) * 50e-6  # 25 Hz
    noise = np.random.randn(n_samples) * 50e-6

    # Multi-channel (19 channels)
    alpha_multi = np.tile(alpha, (19, 1))
    beta_multi = np.tile(beta, (19, 1))
    noise_multi = np.tile(noise, (19, 1))

    # Channel names
    ch_names = [f"CH{i + 1}" for i in range(19)]

    # Extract features
    print("Extracting features...")
    feat_alpha = model.extract_features(alpha_multi, ch_names)
    feat_beta = model.extract_features(beta_multi, ch_names)
    feat_noise = model.extract_features(noise_multi, ch_names)

    print(f"Feature shape: {feat_alpha.shape}")
    print("Expected: (4, 512) for 4 summary tokens\n")

    # Check if all summary tokens are identical
    all_identical = True
    for i in range(4):
        for j in range(i + 1, 4):
            if not np.allclose(feat_alpha[i], feat_alpha[j]):
                all_identical = False
                break

    print(f"Are all 4 summary tokens identical? {all_identical}")

    # Compute similarities
    def cosine_sim(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

    sim_alpha_beta = cosine_sim(feat_alpha, feat_beta)
    sim_alpha_noise = cosine_sim(feat_alpha, feat_noise)
    sim_beta_noise = cosine_sim(feat_beta, feat_noise)

    print("\nCosine similarities:")
    print(f"  Alpha vs Beta:  {sim_alpha_beta:.6f}")
    print(f"  Alpha vs Noise: {sim_alpha_noise:.6f}")
    print(f"  Beta vs Noise:  {sim_beta_noise:.6f}")

    # Verdict
    if sim_alpha_beta > 0.99:
        print("\n❌ FEATURES ARE NOT DISCRIMINATIVE!")
        print("   Summary token fix may not be working correctly.")
    else:
        print("\n✅ FEATURES ARE DISCRIMINATIVE!")
        print("   Summary token fix is working correctly.")

    return sim_alpha_beta < 0.99


if __name__ == "__main__":
    test_discrimination()
