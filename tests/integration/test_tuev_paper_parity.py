"""Integration test for TUEV paper parity mode (23 channels + mapper)."""

import os
from pathlib import Path

import pytest
import torch

from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.utils import collate_tuev_parity_batch


@pytest.mark.integration
def test_paper_parity_pipeline():
    """Test complete pipeline with 23ch input + mapper."""
    # Skip if data not available
    data_root = os.environ.get('BGB_DATA_ROOT')
    if not data_root:
        pytest.skip("BGB_DATA_ROOT not set")

    tuev_path = Path(data_root) / 'datasets/tuev'
    if not tuev_path.exists():
        pytest.skip(f"TUEV dataset not found at {tuev_path}")

    # Create dataset with paper parity mode
    dataset = TUEVMNEDataset(
        root_dir=tuev_path,
        split='eval',
        use_paper_parity=True,  # 23 channels
    )

    # Get one sample
    x, y = dataset[0]
    assert x.shape[0] == 23, f"Expected 23 channels, got {x.shape[0]}"
    assert x.shape[1] == 1024, f"Expected 1024 samples, got {x.shape[1]}"

    # Test collate function accepts 23 channels
    batch = [(torch.from_numpy(x), y)]
    batch_x, batch_y = collate_tuev_parity_batch(batch)
    assert batch_x.shape == (1, 23, 1024), f"Unexpected batch shape: {batch_x.shape}"

    # Apply mapper
    mapper = TUEVChannelMapper(in_channels=23, out_channels=20)
    x_mapped = mapper(batch_x)
    assert x_mapped.shape == (1, 20, 1024), (
        f"Expected (1, 20, 1024) after mapping, got {x_mapped.shape}"
    )

    # Feed to EEGPT
    model = create_normalized_eegpt()
    model.eval()

    with torch.no_grad():
        features = model.extract_features(x_mapped, summary=False)

    # EEGPT outputs (B, 16, 4, 512) for 4-second windows
    assert features.shape == (1, 16, 4, 512), f"Unexpected feature shape: {features.shape}"

    print("✅ Paper parity pipeline test passed!")
    print(f"  - Dataset: 23 channels")
    print(f"  - Collate: Accepts 23 channels")
    print(f"  - Mapper: 23→20 channels")
    print(f"  - EEGPT: Features extracted successfully")


@pytest.mark.integration
def test_gradient_flow_through_mapper():
    """Test that gradients flow through the mapper in training."""
    data_root = os.environ.get('BGB_DATA_ROOT')
    if not data_root:
        pytest.skip("BGB_DATA_ROOT not set")

    # Create simple test data
    x_23ch = torch.randn(2, 23, 1024, requires_grad=True)

    # Create mapper
    mapper = TUEVChannelMapper(in_channels=23, out_channels=20)

    # Forward pass
    x_20ch = mapper(x_23ch)
    assert x_20ch.shape == (2, 20, 1024)

    # Simulate loss
    loss = x_20ch.mean()
    loss.backward()

    # Check gradients exist
    assert x_23ch.grad is not None, "No gradient for input"
    assert torch.any(x_23ch.grad != 0), "Zero gradients"

    # Check mapper parameters have gradients
    for name, param in mapper.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.any(param.grad != 0), f"Zero gradient for {name}"

    print("✅ Gradient flow test passed!")


@pytest.mark.integration
def test_cache_has_correct_units():
    """Verify cache was built with Volts (not mV)."""
    import json

    cache_dir = os.environ.get('BGB_CACHE_DIR')
    if not cache_dir:
        pytest.skip("BGB_CACHE_DIR not set")

    meta_file = Path(cache_dir) / 'tuev_23ch_paper_parity/train/META.json'

    if not meta_file.exists():
        pytest.skip("Cache not built yet - run build_tuev_23ch_cache.sh first")

    with open(meta_file) as f:
        meta = json.load(f)

    # Check units are Volts (SI units) per SSOT
    assert meta['unit'] == 'V', f"Cache has wrong units: {meta['unit']} (expected 'V')"
    assert meta['n_channels'] == 23, f"Cache has {meta['n_channels']} channels, expected 23"

    # Check channels are mixed-case
    channels = meta['channels']
    assert 'Fp1' in channels, "Should use mixed-case 'Fp1' not 'FP1'"
    assert 'Fz' in channels, "Should use mixed-case 'Fz' not 'FZ'"

    print("✅ Cache validation passed!")
    print(f"  - Units: {meta['unit']} (correct)")
    print(f"  - Channels: {meta['n_channels']}")
    print(f"  - Naming: Mixed-case (Fp1, Fz, etc.)")


if __name__ == "__main__":
    # Run tests directly
    test_paper_parity_pipeline()
    test_gradient_flow_through_mapper()
    test_cache_has_correct_units()
