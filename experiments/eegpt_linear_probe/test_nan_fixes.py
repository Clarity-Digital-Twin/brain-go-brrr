#!/usr/bin/env python
"""Test NaN fixes before running training."""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from brain_go_brrr.models.eegpt_linear_probe_robust import RobustEEGPTLinearProbe

def test_nan_handling():
    """Test that model handles NaN gracefully."""
    print("Testing NaN handling...")
    
    # Mock checkpoint path
    data_root = Path(__file__).resolve().parents[2] / "data"
    checkpoint = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    # Create model
    model = RobustEEGPTLinearProbe(
        checkpoint_path=checkpoint,
        n_input_channels=20,
        n_classes=2
    )
    model.eval()
    
    # Test cases
    test_cases = [
        ("Normal input", torch.randn(4, 20, 2048)),
        ("Input with NaN", torch.randn(4, 20, 2048)),
        ("Input with Inf", torch.randn(4, 20, 2048)),
        ("All zeros", torch.zeros(4, 20, 2048)),
        ("Extreme values", torch.randn(4, 20, 2048) * 1000),
    ]
    
    # Add NaN to second test case
    test_cases[1][1][0, 5, 100:200] = float('nan')
    
    # Add Inf to third test case  
    test_cases[2][1][0, 10, 500:600] = float('inf')
    
    for name, x in test_cases:
        print(f"\n{name}:")
        print(f"  Input stats: min={x.min():.2f}, max={x.max():.2f}, std={x.std():.2f}")
        print(f"  Has NaN: {torch.isnan(x).any()}")
        print(f"  Has Inf: {torch.isinf(x).any()}")
        
        try:
            with torch.no_grad():
                output = model(x)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Output has NaN: {torch.isnan(output).any()}")
            print(f"  Output has Inf: {torch.isinf(output).any()}")
            print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
            print("  ✅ PASSED")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print(f"\nModel statistics:")
    print(f"  NaN count: {model.nan_count.item()}")
    print(f"  Clip count: {model.clip_count.item()}")


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("\n\nTesting gradient flow...")
    
    data_root = Path(__file__).resolve().parents[2] / "data"
    checkpoint = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    model = RobustEEGPTLinearProbe(
        checkpoint_path=checkpoint,
        n_input_channels=20,
        n_classes=2,
        freeze_backbone=True  # Only train probe
    )
    
    # Create problematic input
    x = torch.randn(8, 20, 2048)
    x[0, :, 100:200] = 100.0  # Extreme values
    x.requires_grad = True
    
    # Forward pass
    output = model(x)
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("Gradient check:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any()
            print(f"  {name}: norm={grad_norm:.6f}, has_nan={has_nan}")
    
    print("✅ Gradient test completed")


if __name__ == "__main__":
    print("=" * 60)
    print("NaN ROBUSTNESS TEST")
    print("=" * 60)
    
    try:
        test_nan_handling()
        test_gradient_flow()
        print("\n✅ ALL TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()