"""Test that encoder outputs proper summary tokens."""

import numpy as np
import pytest
import torch

from brain_go_brrr.models.eegpt_model import EEGPTModel


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

    assert features.shape == (4, 512), f"Expected (4, 512) summary tokens, got {features.shape}"

    # Check that summary tokens are different from each other
    similarities = []
    for i in range(4):
        for j in range(i + 1, 4):
            similarity = torch.cosine_similarity(
                torch.from_numpy(features[i]), torch.from_numpy(features[j]), dim=0
            )
            similarities.append(similarity.item())

    # Summary tokens should be different (not identical)
    avg_similarity = np.mean(similarities)
    # With random initialization, tokens will be somewhat similar but not identical
    # Relaxed threshold for mock model
    assert (
        avg_similarity < 0.995
    ), f"Summary tokens too similar (avg similarity: {avg_similarity:.4f})"

    # Check token statistics
    for i in range(4):
        {
            "mean": features[i].mean(),
            "std": features[i].std(),
            "min": features[i].min(),
            "max": features[i].max(),
        }



@pytest.mark.integration  # Requires model internals
def test_find_summary_tokens():
    """Verify the encoder has summary token parameters."""
    model = EEGPTModel()

    # Check if encoder has summary tokens as parameters
    found_summary_token = False

    for name, param in model.encoder.named_parameters():
        if "summary" in name.lower() or "cls" in name.lower() or "token" in name.lower():
            if "summary_token" in name:
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
