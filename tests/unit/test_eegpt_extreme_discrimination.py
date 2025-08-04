"""Test EEGPT with extreme cases to expose the averaging bug."""

import numpy as np
import pytest

from brain_go_brrr.models.eegpt_model import EEGPTModel


@pytest.mark.integration
def test_extreme_pattern_discrimination():
    """Test with extremely different patterns that should produce very different features."""
    model = EEGPTModel()
    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "Pz",
        "P4",
        "T6",
        "O1",
        "O2",
    ]

    # Pattern 1: All zeros
    zeros = np.zeros((19, 1024))

    # Pattern 2: All ones
    ones = np.ones((19, 1024))

    # Pattern 3: Random noise
    noise = np.random.randn(19, 1024)

    # Pattern 4: Alternating spikes
    spikes = np.zeros((19, 1024))
    spikes[:, ::2] = 1.0  # Every other sample is 1

    # Extract features
    feat_zeros = model.extract_features(zeros, ch_names)
    feat_ones = model.extract_features(ones, ch_names)
    feat_noise = model.extract_features(noise, ch_names)
    feat_spikes = model.extract_features(spikes, ch_names)


    # Check if all 4 summary tokens are identical within each feature
    for _i, (_name, feat) in enumerate(
        [("zeros", feat_zeros), ("ones", feat_ones), ("noise", feat_noise), ("spikes", feat_spikes)]
    ):
        for _j in range(4):
            pass

        # Check if all tokens are identical
        for j in range(1, 4):
            if not np.allclose(feat[0], feat[j]):
                break

    # Calculate similarities
    def cosine_sim(a, b):
        return np.dot(a.flatten(), b.flatten()) / (
            np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten())
        )


    # These should be very different!
    assert cosine_sim(feat_zeros, feat_ones) < 0.99, "Zeros and ones too similar!"
    assert cosine_sim(feat_zeros, feat_noise) < 0.95, "Zeros and noise too similar!"
    assert cosine_sim(feat_ones, feat_spikes) < 0.95, "Ones and spikes too similar!"


@pytest.mark.integration
def test_check_averaging_bug():
    """Directly test if we're averaging features."""
    model = EEGPTModel()
    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "Pz",
        "P4",
        "T6",
        "O1",
        "O2",
    ]

    # Create data where each channel has a different constant value
    data = np.zeros((19, 1024))
    for i in range(19):
        data[i, :] = i  # Channel i has value i

    features = model.extract_features(data, ch_names)


    # If we're averaging across channels, all features should be similar
    # and equal to the mean of 0-18 = 9
    np.mean(range(19))  # Should be 9

    for _i in range(4):
        pass

    # Check if tokens are just copies
    for i in range(1, 4):
        np.allclose(features[0], features[i])


if __name__ == "__main__":
    test_extreme_pattern_discrimination()
    test_check_averaging_bug()
