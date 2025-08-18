#!/usr/bin/env python
"""Test TUEV model shapes end-to-end."""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train_tuev_aligned import TUEVLinearProbe
from tuev_dataset_padded import TUEVCachedDatasetPadded


def main():
    print("=" * 60)
    print("TUEV SHAPE TEST")
    print("=" * 60)
    
    # Load config
    config = OmegaConf.load("configs/tuev_table13_aligned.yaml")
    
    # Create model
    print("\n1. Creating model...")
    model = TUEVLinearProbe(
        eegpt_checkpoint=config.model.eegpt_checkpoint,
        device='cpu'  # Use CPU for testing
    )
    
    # Check model layers
    print("\n2. Model architecture:")
    print(f"   Channel reducer: {model.channel_reducer}")
    print(f"   Temporal conv: {model.temporal_conv}")
    print(f"   Classifier: {model.classifier}")
    
    # Load one sample from cache
    print("\n3. Loading cached sample...")
    dataset = TUEVCachedDatasetPadded(
        cache_dir=Path(config.data.cache_dir),
        split='train',
        padding='edge'
    )
    
    x, y = dataset[0]
    print(f"   Input shape: {x.shape} (should be 23×1024)")
    print(f"   Label: {y}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    x_batch = x.unsqueeze(0)  # Add batch dimension
    print(f"   Batch input: {x_batch.shape}")
    
    with torch.no_grad():
        # Test each layer
        x1 = model.channel_reducer(x_batch)
        print(f"   After channel reduction: {x1.shape} (should be 1×20×1024)")
        
        x2 = model.temporal_conv(x1)
        print(f"   After temporal conv: {x2.shape} (should be 1×20×1024)")
        
        # Test EEGPT
        features = model.eegpt.extract_features(x2)
        print(f"   EEGPT features: {features.shape}")
        
        # Full forward
        logits = model(x_batch)
        print(f"   Final logits: {logits.shape} (should be 1×6)")
    
    print("\n5. Sanity check:")
    if logits.shape == (1, 6):
        print("   ✅ Model architecture is correct!")
    else:
        print("   ❌ Model architecture is broken!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()