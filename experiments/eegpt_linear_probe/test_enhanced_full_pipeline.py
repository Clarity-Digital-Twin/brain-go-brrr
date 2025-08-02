#!/usr/bin/env python
"""COMPREHENSIVE SMOKE TEST - Test ENTIRE pipeline before training."""

import sys
import os
from pathlib import Path
import traceback

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Import everything we'll use
from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from brain_go_brrr.tasks.enhanced_abnormality_detection import EnhancedAbnormalityDetectionProbe


def test_full_pipeline():
    """Test EVERYTHING before launching training."""
    print("="*80)
    print("COMPREHENSIVE ENHANCED EEGPT PIPELINE TEST")
    print("="*80)
    
    # Paths
    data_root = project_root / "data"
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    # Check checkpoint exists
    print("\n1. Checking checkpoint...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"✓ Checkpoint exists: {checkpoint_path}")
    
    # Load config
    print("\n2. Loading configuration...")
    config_path = Path(__file__).parent / "configs/tuab_enhanced_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    print("✓ Config loaded")
    
    # Test dataset creation
    print("\n3. Creating dataset...")
    try:
        dataset = TUABEnhancedDataset(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            split="train",
            window_duration=10.24,  # Must be divisible by 64 @ 200Hz
            sampling_rate=200,
            preload=False,
            channels=['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 
                     'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2'],
            use_old_naming=True,
            bandpass_low=0.1,
            bandpass_high=75.0,
            notch_freq=50.0,
            window_stride=5.12,
            cache_dir=data_root / "cache/tuab_enhanced_test",
        )
        print(f"✓ Dataset created: {len(dataset)} windows")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test data loading
    print("\n4. Loading sample data...")
    try:
        x, y = dataset[0]
        print(f"✓ Sample loaded: shape={x.shape}, label={y}")
        print(f"  Expected shape: [19, 2048]")
        assert x.shape == torch.Size([19, 2048]), f"Wrong shape: {x.shape}"
        assert torch.isfinite(x).all(), "Data contains NaN/Inf!"
        print("✓ Data is clean (no NaN/Inf)")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        traceback.print_exc()
        return False
    
    # Test sample weights
    print("\n5. Testing sample weights...")
    try:
        weights = dataset.get_sample_weights()
        print(f"✓ Sample weights computed: shape={weights.shape}")
        assert len(weights) == len(dataset), f"Wrong weight count: {len(weights)} vs {len(dataset)}"
        assert torch.isfinite(weights).all(), "Weights contain NaN/Inf!"
        print("✓ Weights are valid")
    except Exception as e:
        print(f"✗ Sample weights failed: {e}")
        traceback.print_exc()
        return False
    
    # Test data loader
    print("\n6. Creating data loader...")
    try:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights[:1000], num_samples=100, replacement=True)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)
        batch_x, batch_y = next(iter(loader))
        print(f"✓ Batch loaded: shape={batch_x.shape}, labels={batch_y}")
        assert batch_x.shape == torch.Size([2, 19, 2048]), f"Wrong batch shape: {batch_x.shape}"
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        traceback.print_exc()
        return False
    
    # Test model creation
    print("\n7. Creating EEGPT backbone...")
    try:
        backbone = create_normalized_eegpt(checkpoint_path=str(checkpoint_path))
        backbone = backbone.cuda()
        backbone.eval()
        print("✓ Backbone loaded and moved to GPU")
    except Exception as e:
        print(f"✗ Backbone creation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test backbone forward pass
    print("\n8. Testing backbone forward pass...")
    try:
        with torch.no_grad():
            x_gpu = batch_x.cuda()
            features = backbone(x_gpu)
            print(f"✓ Features extracted: shape={features.shape}")
            assert features.shape == torch.Size([2, 4, 512]), f"Wrong feature shape: {features.shape}"
            assert torch.isfinite(features).all(), "Features contain NaN/Inf!"
            print("✓ Features are clean")
    except Exception as e:
        print(f"✗ Backbone forward failed: {e}")
        traceback.print_exc()
        return False
    
    # Test probe creation
    print("\n9. Creating two-layer probe...")
    try:
        probe = EEGPTTwoLayerProbe(
            backbone_dim=768,
            n_input_channels=19,
            n_adapted_channels=19,
            hidden_dim=16,
            n_classes=2,
            dropout=0.5,
            use_channel_adapter=True,
        )
        probe = probe.cuda()
        probe.eval()  # Disable dropout for testing
        print("✓ Probe created")
    except Exception as e:
        print(f"✗ Probe creation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test probe forward
    print("\n10. Testing probe forward pass...")
    try:
        with torch.no_grad():
            logits = probe(features)
            print(f"✓ Logits computed: shape={logits.shape}")
            assert logits.shape == torch.Size([2, 2]), f"Wrong logit shape: {logits.shape}"
            assert torch.isfinite(logits).all(), "Logits contain NaN/Inf!"
            print("✓ Logits are clean")
    except Exception as e:
        print(f"✗ Probe forward failed: {e}")
        traceback.print_exc()
        return False
    
    # Test Lightning module
    print("\n11. Creating Lightning module...")
    try:
        lightning_model = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=str(checkpoint_path),
            probe=probe,
            n_channels=19,
            learning_rate=5e-4,
            weight_decay=0.05,
            warmup_epochs=5,
            total_epochs=50,
            layer_decay=0.65,
            scheduler_type="onecycle",
            freeze_backbone=True,
        )
        print("✓ Lightning module created")
    except Exception as e:
        print(f"✗ Lightning module failed: {e}")
        traceback.print_exc()
        return False
    
    # Test training step
    print("\n12. Testing training step...")
    try:
        lightning_model = lightning_model.cuda()
        loss = lightning_model.training_step((batch_x.cuda(), batch_y.cuda()), 0)
        print(f"✓ Training step completed: loss={loss.item():.4f}")
        assert torch.isfinite(loss), "Loss is NaN/Inf!"
        print("✓ Loss is valid")
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        traceback.print_exc()
        return False
    
    # Test validation step
    print("\n13. Testing validation step...")
    try:
        lightning_model.validation_step((batch_x.cuda(), batch_y.cuda()), 0)
        print("✓ Validation step completed")
    except Exception as e:
        print(f"✗ Validation step failed: {e}")
        traceback.print_exc()
        return False
    
    # Test optimizer configuration
    print("\n14. Testing optimizer configuration...")
    try:
        # Mock trainer for optimizer test
        class MockTrainer:
            def __init__(self):
                self.estimated_stepping_batches = 1000
        
        lightning_model.trainer = MockTrainer()
        opt_config = lightning_model.configure_optimizers()
        print(f"✓ Optimizer configured: {type(opt_config['optimizer']).__name__}")
        print(f"✓ Scheduler configured: {type(opt_config['lr_scheduler']['scheduler']).__name__}")
    except Exception as e:
        print(f"✗ Optimizer config failed: {e}")
        traceback.print_exc()
        return False
    
    # Test directory creation
    print("\n15. Testing log directory creation...")
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / f"test_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Log directory created: {log_dir}")
        # Clean up
        log_dir.rmdir()
    except Exception as e:
        print(f"✗ Directory creation failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - READY FOR TRAINING!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    if not success:
        print("\n❌ PIPELINE TEST FAILED - DO NOT LAUNCH TRAINING!")
        sys.exit(1)
    else:
        print("\n✅ Pipeline validated. Safe to launch training.")
        print("\nTo launch training:")
        print("tmux new -s eegpt_enhanced -d \\")
        print("  \"uv run python experiments/eegpt_linear_probe/train_enhanced.py \\")
        print("    2>&1 | tee logs/enhanced_$(date +%Y%m%d_%H%M%S).log\"")