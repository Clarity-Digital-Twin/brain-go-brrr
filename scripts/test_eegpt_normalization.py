#!/usr/bin/env python3
"""Test if normalization fixes EEGPT feature discrimination."""

import numpy as np
import torch

from brain_go_brrr  # noqa: E402.models.eegpt_wrapper import create_normalized_eegpt


def test_normalization():
    """Test EEGPT with and without normalization."""
    print("=== TESTING EEGPT NORMALIZATION ===\n")

    checkpoint_path = "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    # Create test signals
    t = np.arange(1024) / 256  # 4 seconds at 256 Hz

    # Different frequency patterns
    alpha = np.sin(2 * np.pi * 10 * t) * 50e-6  # 10 Hz, 50 microvolts
    beta = np.sin(2 * np.pi * 25 * t) * 30e-6  # 25 Hz, 30 microvolts
    np.random.randn(1024) * 40e-6  # Random noise

    # Multi-channel signals
    signal1 = torch.FloatTensor(np.tile(alpha, (19, 1))).unsqueeze(0)
    signal2 = torch.FloatTensor(np.tile(beta, (19, 1))).unsqueeze(0)
    signal3 = torch.FloatTensor(np.random.randn(19, 1024) * 40e-6).unsqueeze(0)

    print("Input signals:")
    print(f"  Alpha (10 Hz): range [{signal1.min():.6f}, {signal1.max():.6f}]")
    print(f"  Beta (25 Hz):  range [{signal2.min():.6f}, {signal2.max():.6f}]")
    print(f"  Noise:         range [{signal3.min():.6f}, {signal3.max():.6f}]")

    # Test 1: Without normalization
    print("\n=== TEST 1: Without Normalization ===")
    model_raw = create_normalized_eegpt(checkpoint_path, normalize=False)
    model_raw.eval()

    with torch.no_grad():
        feat1_raw = model_raw(signal1)
        feat2_raw = model_raw(signal2)
        feat3_raw = model_raw(signal3)

    # Calculate similarities
    sim_12_raw = torch.nn.functional.cosine_similarity(
        feat1_raw.flatten().unsqueeze(0), feat2_raw.flatten().unsqueeze(0)
    ).item()
    sim_13_raw = torch.nn.functional.cosine_similarity(
        feat1_raw.flatten().unsqueeze(0), feat3_raw.flatten().unsqueeze(0)
    ).item()
    sim_23_raw = torch.nn.functional.cosine_similarity(
        feat2_raw.flatten().unsqueeze(0), feat3_raw.flatten().unsqueeze(0)
    ).item()

    print("Cosine similarities (without normalization):")
    print(f"  Alpha vs Beta:  {sim_12_raw:.6f}")
    print(f"  Alpha vs Noise: {sim_13_raw:.6f}")
    print(f"  Beta vs Noise:  {sim_23_raw:.6f}")

    # Test 2: With normalization
    print("\n=== TEST 2: With Normalization ===")
    model_norm = create_normalized_eegpt(checkpoint_path, normalize=True)

    # Estimate normalization from the signals
    all_signals = torch.cat([signal1, signal2, signal3], dim=0)
    model_norm.estimate_normalization_params(all_signals)
    model_norm.eval()

    with torch.no_grad():
        feat1_norm = model_norm(signal1)
        feat2_norm = model_norm(signal2)
        feat3_norm = model_norm(signal3)

    # Calculate similarities
    sim_12_norm = torch.nn.functional.cosine_similarity(
        feat1_norm.flatten().unsqueeze(0), feat2_norm.flatten().unsqueeze(0)
    ).item()
    sim_13_norm = torch.nn.functional.cosine_similarity(
        feat1_norm.flatten().unsqueeze(0), feat3_norm.flatten().unsqueeze(0)
    ).item()
    sim_23_norm = torch.nn.functional.cosine_similarity(
        feat2_norm.flatten().unsqueeze(0), feat3_norm.flatten().unsqueeze(0)
    ).item()

    print("\nCosine similarities (with normalization):")
    print(f"  Alpha vs Beta:  {sim_12_norm:.6f}")
    print(f"  Alpha vs Noise: {sim_13_norm:.6f}")
    print(f"  Beta vs Noise:  {sim_23_norm:.6f}")

    # Test 3: Standard scaling (mean=0, std=1)
    print("\n=== TEST 3: Standard Scaling ===")
    model_std = create_normalized_eegpt(checkpoint_path, normalize=True, mean=0.0, std=1.0)
    model_std.eval()

    # Scale inputs to ~N(0, 1)
    signal1_scaled = (signal1 - signal1.mean()) / signal1.std()
    signal2_scaled = (signal2 - signal2.mean()) / signal2.std()
    signal3_scaled = (signal3 - signal3.mean()) / signal3.std()

    with torch.no_grad():
        feat1_std = model_std(signal1_scaled)
        feat2_std = model_std(signal2_scaled)
        feat3_std = model_std(signal3_scaled)

    sim_12_std = torch.nn.functional.cosine_similarity(
        feat1_std.flatten().unsqueeze(0), feat2_std.flatten().unsqueeze(0)
    ).item()
    sim_13_std = torch.nn.functional.cosine_similarity(
        feat1_std.flatten().unsqueeze(0), feat3_std.flatten().unsqueeze(0)
    ).item()

    print("\nCosine similarities (standard scaling):")
    print(f"  Alpha vs Beta:  {sim_12_std:.6f}")
    print(f"  Alpha vs Noise: {sim_13_std:.6f}")

    # Summary
    print("\n=== SUMMARY ===")
    max_sim_raw = max(sim_12_raw, sim_13_raw, sim_23_raw)
    max_sim_norm = max(sim_12_norm, sim_13_norm, sim_23_norm)
    max_sim_std = max(sim_12_std, sim_13_std)

    print("\nMaximum similarity between different signals:")
    print(f"  Without normalization: {max_sim_raw:.6f}")
    print(f"  With normalization:    {max_sim_norm:.6f}")
    print(f"  Standard scaling:      {max_sim_std:.6f}")

    if max_sim_std < 0.95:
        print("\n✅ SUCCESS: Normalization makes features discriminative!")
    else:
        print("\n❌ PROBLEM: Features still not discriminative after normalization")


if __name__ == "__main__":
    test_normalization()
