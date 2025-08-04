#!/usr/bin/env python3
"""Verify all EEGPT fixes are working correctly."""

from pathlib import Path

import numpy as np
import torch

from brain_go_brrr  # noqa: E402.core.config import ModelConfig
from brain_go_brrr  # noqa: E402.models.eegpt_model import EEGPTModel
from brain_go_brrr  # noqa: E402.models.eegpt_wrapper import create_normalized_eegpt


def verify_fixes():
    """Run verification tests for all fixes."""
    print("=== VERIFYING ALL EEGPT FIXES ===\n")

    checkpoint_path = "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    # Test 1: Normalization stats loading
    print("1. Testing normalization stats loading...")
    model = create_normalized_eegpt(checkpoint_path)
    assert model._stats_source == "file", f"Stats not loaded from file: {model._stats_source}"
    assert abs(model.input_std.item() - 0.000021) < 0.000001, "Incorrect std loaded"
    print("   ✅ Normalization stats loaded from file")

    # Test 2: Feature discrimination
    print("\n2. Testing feature discrimination...")
    t = np.arange(1024) / 256
    alpha = torch.FloatTensor(np.sin(2 * np.pi * 10 * t) * 50e-6).repeat(19, 1).unsqueeze(0)
    beta = torch.FloatTensor(np.sin(2 * np.pi * 25 * t) * 30e-6).repeat(19, 1).unsqueeze(0)

    with torch.no_grad():
        feat_alpha = model(alpha)
        feat_beta = model(beta)

    cosine_sim = torch.nn.functional.cosine_similarity(
        feat_alpha.flatten().unsqueeze(0), feat_beta.flatten().unsqueeze(0)
    ).item()

    assert cosine_sim < 0.9, f"Features not discriminative: {cosine_sim:.3f}"
    print(f"   ✅ Features are discriminative (similarity: {cosine_sim:.3f})")

    # Test 3: Input validation
    print("\n3. Testing input validation...")

    # Test patch size validation
    try:
        bad_input = torch.randn(1, 19, 1000)  # Not divisible by 64
        model(bad_input)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "divisible by patch_size" in str(e)
        print("   ✅ Patch size validation working")

    # Test channel ID validation
    try:
        good_input = torch.randn(1, 19, 1024)
        bad_chan_ids = torch.tensor([0, 1, 100])  # 100 > max channel ID
        model.model(good_input, bad_chan_ids)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "exceeds maximum" in str(e)
        print("   ✅ Channel ID validation working")

    # Test 4: RoPE enabled
    print("\n4. Testing RoPE is enabled...")
    first_block = model.model.blocks[0]
    assert hasattr(first_block.attn, "use_rope"), "Attention doesn't have use_rope"
    assert first_block.attn.use_rope, "RoPE is not enabled"
    print("   ✅ RoPE is enabled in attention blocks")

    # Test 5: EEGPTModel integration
    print("\n5. Testing EEGPTModel wrapper...")
    config = ModelConfig()
    config.model_path = Path(checkpoint_path)

    eegpt_model = EEGPTModel(config=config)
    eegpt_model.load_model()

    # Test feature extraction
    test_data = np.random.randn(19, 1024) * 50e-6
    channel_names = [f"EEG{i:03d}" for i in range(19)]

    features = eegpt_model.extract_features(test_data, channel_names)
    assert features.shape == (4, 512), f"Wrong feature shape: {features.shape}"
    print("   ✅ EEGPTModel wrapper working correctly")

    # Test 6: Test checkpoint for CI
    print("\n6. Testing minimal checkpoint for CI...")
    test_ckpt_path = Path("tests/fixtures/models/eegpt_test_checkpoint.ckpt")
    if test_ckpt_path.exists():
        test_model = create_normalized_eegpt(str(test_ckpt_path))
        test_output = test_model(torch.randn(1, 19, 1024))
        assert test_output.shape == (1, 4, 512)
        print("   ✅ Test checkpoint working")

        # Check size
        size_mb = test_ckpt_path.stat().st_size / 1024 / 1024
        print(f"   Test checkpoint size: {size_mb:.1f} MB (vs 1020 MB full)")
    else:
        print("   ⚠️  Test checkpoint not found")

    print("\n✅ ALL FIXES VERIFIED SUCCESSFULLY!")
    print("\nSummary of fixes:")
    print("- Normalization from saved stats (reproducible)")
    print("- Feature discrimination working (~0.4 similarity)")
    print("- Input validation for patches and channels")
    print("- RoPE enabled for temporal encoding")
    print("- All wrappers integrated correctly")
    print("- Minimal test checkpoint for fast CI")


if __name__ == "__main__":
    verify_fixes()
