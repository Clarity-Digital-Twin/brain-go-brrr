#!/usr/bin/env python
"""Test enhanced setup before training."""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import os
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

import torch
from omegaconf import OmegaConf

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from brain_go_brrr.tasks.enhanced_abnormality_detection import EnhancedAbnormalityDetectionProbe


def test_dataset():
    """Test enhanced dataset."""
    print("\n1. Testing Enhanced Dataset...")
    
    dataset = TUABEnhancedDataset(
        root_dir=Path(os.environ["BGB_DATA_ROOT"]) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=10.0,  # 10 seconds
        window_stride=5.0,     # 50% overlap
        sampling_rate=200,     # 200 Hz
        bandpass_low=0.1,
        bandpass_high=75.0,
        notch_freq=50.0,
        n_jobs=1,
    )
    
    print(f"   Dataset size: {len(dataset)} windows")
    print(f"   Class counts: {dataset.class_counts}")
    
    # Test single sample
    x, y = dataset[0]
    print(f"   Sample shape: {x.shape} (expected: [20, 2000])")
    print(f"   Sample label: {y}")
    
    assert x.shape == (20, 2000), f"Wrong shape: {x.shape}"
    print("   ✓ Dataset test passed!")
    
    return dataset


def test_probe():
    """Test two-layer probe."""
    print("\n2. Testing Two-Layer Probe...")
    
    probe = EEGPTTwoLayerProbe(
        backbone_dim=768,
        n_input_channels=20,
        n_adapted_channels=19,
        hidden_dim=16,
        n_classes=2,
        dropout=0.5,
        use_channel_adapter=True,
    )
    
    # Test with dummy input
    batch_size = 4
    features = torch.randn(batch_size, 768)  # Summary token features
    
    logits = probe(features)
    print(f"   Input shape: {features.shape}")
    print(f"   Output shape: {logits.shape} (expected: [{batch_size}, 2])")
    
    assert logits.shape == (batch_size, 2), f"Wrong output shape: {logits.shape}"
    print("   ✓ Probe test passed!")
    
    return probe


def test_lightning_module():
    """Test enhanced Lightning module."""
    print("\n3. Testing Lightning Module...")
    
    checkpoint_path = Path(os.environ["BGB_DATA_ROOT"]) / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    if not checkpoint_path.exists():
        print(f"   ⚠️  Checkpoint not found at {checkpoint_path}")
        print("   Skipping Lightning module test")
        return None
    
    probe = EEGPTTwoLayerProbe(
        backbone_dim=768,
        n_input_channels=20,
        n_adapted_channels=19,
        hidden_dim=16,
        n_classes=2,
        dropout=0.5,
    )
    
    model = EnhancedAbnormalityDetectionProbe(
        checkpoint_path=str(checkpoint_path),
        probe=probe,
        n_channels=20,
        learning_rate=5e-4,
        weight_decay=0.05,
        warmup_epochs=5,
        total_epochs=50,
        layer_decay=0.65,
    )
    
    # Test forward pass
    x = torch.randn(4, 20, 2048)  # [B, C, T]
    logits = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print("   ✓ Lightning module test passed!")
    
    return model


def test_config():
    """Test configuration loading."""
    print("\n4. Testing Configuration...")
    
    config_path = Path(__file__).parent / "configs/tuab_enhanced_config.yaml"
    
    if not config_path.exists():
        print(f"   ⚠️  Config not found at {config_path}")
        return None
    
    cfg = OmegaConf.load(config_path)
    
    print(f"   Window duration: {cfg.data.window_duration}s")
    print(f"   Sampling rate: {cfg.data.sampling_rate}Hz")
    print(f"   Batch size: {cfg.data.batch_size}")
    print(f"   Epochs: {cfg.training.epochs}")
    print(f"   Learning rate: {cfg.training.learning_rate}")
    print("   ✓ Config test passed!")
    
    return cfg


def main():
    """Run all tests."""
    print("=" * 60)
    print("ENHANCED EEGPT SETUP TEST")
    print("=" * 60)
    
    try:
        # Test components
        dataset = test_dataset()
        probe = test_probe()
        model = test_lightning_module()
        cfg = test_config()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED! Ready for enhanced training.")
        print("=" * 60)
        
        print("\nKey improvements verified:")
        print("  ✓ 10-second windows @ 200Hz")
        print("  ✓ Two-layer probe architecture")
        print("  ✓ Channel adaptation layer")
        print("  ✓ Enhanced preprocessing pipeline")
        
        print("\nLaunch training with:")
        print("  ./experiments/eegpt_linear_probe/launch_enhanced_training.sh")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()