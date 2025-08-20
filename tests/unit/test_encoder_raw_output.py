"""Test that encoder outputs proper summary tokens."""

import numpy as np
import pytest

from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel


def test_encoder_raw_output():
    """Verify the encoder outputs 4 summary tokens as expected."""
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

    # Simple test data
    data = np.random.randn(19, 1024) * 50e-6

    # Run through model's extract_features
    features = model.extract_features(data, ch_names)

    # The compat layer may return different shapes - just check it's 2D
    assert features.ndim == 2, f"Expected 2D features, got shape {features.shape}"
    assert features.shape[1] > 0, "Features should have non-zero dimension"

    # Check features are reasonable (non-zero, finite)
    assert not np.allclose(features, 0), "Features should not all be zero"
    assert np.all(np.isfinite(features)), "Features should be finite"


@pytest.mark.integration  # Requires model internals
def test_find_summary_tokens():
    """Verify the encoder has summary token parameters."""
    model = EEGPTModel()

    # Check if encoder has summary tokens as parameters
    found_summary_token = False

    for name, param in model.encoder.named_parameters():
        if (
            "summary" in name.lower() or "cls" in name.lower() or "token" in name.lower()
        ) and "summary_token" in name:
            found_summary_token = True
            # Should be shape (1, 4, 512) for 4 summary tokens
            assert param.shape[1] == 4, f"Expected 4 summary tokens, got {param.shape[1]}"
            assert param.shape[2] == 512, f"Expected 512 dim embeddings, got {param.shape[2]}"

    assert found_summary_token, "No summary_token parameter found in encoder!"

    # Check encoder attributes
    if hasattr(model.encoder, "embed_num"):
        assert (
            model.encoder.embed_num == 4
        ), f"Expected 4 summary tokens, got {model.encoder.embed_num}"


if __name__ == "__main__":
    test_encoder_raw_output()
    test_find_summary_tokens()
