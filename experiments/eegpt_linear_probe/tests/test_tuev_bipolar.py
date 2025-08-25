"""Test TUEV bipolar channel derivation."""

import numpy as np
from datasets.tuev_dataset import TCP_BIPOLAR_PAIRS, compute_bipolar_derivation


def test_bipolar_derivation():
    """Test that bipolar derivation computes correct differences."""
    # Create mock referential data
    n_samples = 1000
    ch_names = [
        'FP1-REF',
        'F7-REF',
        'T3-REF',
        'T5-REF',
        'O1-REF',
        'FP2-REF',
        'F8-REF',
        'T4-REF',
        'T6-REF',
        'O2-REF',
        'F3-REF',
        'C3-REF',
        'P3-REF',
        'F4-REF',
        'C4-REF',
        'P4-REF',
        'FZ-REF',
        'CZ-REF',
        'PZ-REF',
        'A1-REF',
        'A2-REF',
    ]

    # Create distinct signals for each channel
    data = np.zeros((len(ch_names), n_samples))
    for i in range(len(ch_names)):
        data[i] = np.sin(2 * np.pi * (i + 1) * np.arange(n_samples) / n_samples)

    # Compute bipolar derivations
    bipolar_data = compute_bipolar_derivation(data, ch_names, TCP_BIPOLAR_PAIRS[:23])

    # Check shape
    assert bipolar_data.shape == (23, n_samples), f"Wrong shape: {bipolar_data.shape}"

    # Check that we got non-zero derivations for most channels
    non_zero_channels = np.sum(np.abs(bipolar_data).mean(axis=1) > 0.01)
    assert non_zero_channels >= 20, f"Only {non_zero_channels}/23 channels have signal"

    # Spot check a specific derivation: FP1-F7 should be data[0] - data[1]
    expected_fp1_f7 = data[0] - data[1]  # FP1-REF minus F7-REF
    actual_fp1_f7 = bipolar_data[0]  # First TCP pair is FP1-F7

    np.testing.assert_allclose(
        actual_fp1_f7, expected_fp1_f7, rtol=1e-5, err_msg="FP1-F7 derivation incorrect"
    )

    print(f"✓ Bipolar derivation test passed: {non_zero_channels}/23 channels active")
    print("✓ FP1-F7 derivation correctly computed as FP1-REF minus F7-REF")


def test_missing_channels():
    """Test handling of missing channels."""
    # Only partial channels available
    ch_names = ['FP1', 'F7', 'T3', 'FP2', 'F8']  # Missing many channels
    data = np.random.randn(5, 1000) * 100e-6

    # Should still compute what it can and zero-pad the rest
    bipolar_data = compute_bipolar_derivation(data, ch_names, TCP_BIPOLAR_PAIRS[:23])

    assert bipolar_data.shape == (23, 1000), f"Wrong shape: {bipolar_data.shape}"

    # At least FP1-F7 and FP2-F8 should be computable
    non_zero_channels = np.sum(np.abs(bipolar_data).mean(axis=1) > 0)
    assert non_zero_channels >= 2, f"Should compute at least 2 pairs, got {non_zero_channels}"

    print(f"✓ Missing channels test passed: {non_zero_channels}/23 computed")


if __name__ == "__main__":
    test_bipolar_derivation()
    test_missing_channels()
    print("\n✅ All TUEV bipolar derivation tests passed!")
