#!/usr/bin/env python
"""Smoke test for enhanced EEGPT training with NaN checks."""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe


def smoke_test():
    """Run minimal smoke test with NaN checks."""
    print("Starting enhanced EEGPT smoke test...")
    
    # Paths
    data_root = project_root / "data"
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    # 1. Test dataset instantiation
    print("\n1. Testing dataset instantiation...")
    dataset = TUABEnhancedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=10.24,  # 10.24 seconds = 2048 samples @ 200Hz
        sampling_rate=200,      # 200 Hz
        preload=False,
        channels=['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 
                  'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2'],  # Remove FPZ
        use_old_naming=True,  # TUAB uses old names
    )
    print(f"✓ Dataset created: {len(dataset)} windows")
    
    # 2. Load one sample
    print("\n2. Loading one sample...")
    x, y = dataset[0]
    print(f"✓ Sample loaded: shape={x.shape}, label={y}")
    print(f"  Data range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Check for NaNs in input
    assert torch.isfinite(x).all(), "Input data contains NaN or Inf!"
    print("✓ No NaNs in input data")
    
    # 3. Load EEGPT backbone
    print("\n3. Loading EEGPT backbone...")
    backbone = create_normalized_eegpt(
        checkpoint_path=str(checkpoint_path),
    )
    backbone = backbone.cuda()
    backbone.eval()
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    print("✓ Backbone loaded and frozen")
    
    # 4. Test forward pass through backbone
    print("\n4. Testing backbone forward pass...")
    with torch.no_grad():
        # Add batch dimension and move to GPU
        x_batch = x.unsqueeze(0).cuda()
        
        # Use fp32 for stability
        x_batch = x_batch.float()
        
        # Forward through backbone
        features = backbone(x_batch)
        
        print(f"✓ Features extracted: shape={features.shape}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Check for NaNs
        assert torch.isfinite(features).all(), "Backbone outputs contain NaN or Inf!"
        print("✓ No NaNs in backbone output")
    
    # 5. Test probe
    print("\n5. Testing two-layer probe...")
    probe = EEGPTTwoLayerProbe(
        backbone_dim=768,
        n_input_channels=20,
        n_adapted_channels=19,
        hidden_dim=256,
        n_classes=2,
        dropout=0.5,
        use_channel_adapter=True,
    ).cuda()
    probe.eval()  # Disable dropout for testing
    
    # Forward through probe
    with torch.no_grad():
        logits = probe(features)
        print(f"✓ Logits computed: shape={logits.shape}")
        print(f"  Logit values: {logits[0].cpu().numpy()}")
        
        # Check for NaNs
        assert torch.isfinite(logits).all(), "Probe outputs contain NaN or Inf!"
        print("✓ No NaNs in probe output")
    
    # 6. Test loss computation
    print("\n6. Testing loss computation...")
    criterion = nn.CrossEntropyLoss()
    y_batch = torch.tensor([y]).cuda()
    loss = criterion(logits, y_batch)
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    assert torch.isfinite(loss), "Loss contains NaN or Inf!"
    print("✓ No NaNs in loss")
    
    print("\n" + "="*50)
    print("✅ SMOKE TEST PASSED - No NaNs detected!")
    print("Ready to launch full training.")
    print("="*50)
    
    return True


if __name__ == "__main__":
    smoke_test()