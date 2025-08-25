#!/usr/bin/env python3
"""Quick check to see if loss=0 is real or display issue."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.eegpt_linear_probe.datasets.tuab_dataset import TUABMemoryMappedDataset
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
from torch.utils.data import DataLoader
from experiments.eegpt_linear_probe.utils.custom_collate_fixed import collate_eeg_batch_fixed

# Simple probe
class SimpleProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, 2)  # 4*512 = 2048 input features
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.reshape(x.size(0), -1)
        return self.fc(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    data_root = os.environ.get("BGB_DATA_ROOT", 
                              "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
    
    dataset = TUABMemoryMappedDataset(
        root_dir=f"{data_root}/datasets/external/tuab",
        cache_dir=f"{data_root}/cache/tuab_4s_final",
        split="train",
        n_channels=20,
        sampling_rate=256,
        window_duration=4.0,
        window_stride=2.0
    )
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_eeg_batch_fixed
    )
    
    # Load model
    checkpoint = f"{data_root}/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    backbone = EEGPTWrapper(checkpoint_path=checkpoint)
    backbone.to(device)
    backbone.eval()
    
    probe = SimpleProbe().to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001)
    
    print("\n" + "="*60)
    print("CHECKING FIRST 50 BATCHES")
    print("="*60)
    
    losses = []
    all_labels = []
    
    for i, (data, labels) in enumerate(loader):
        if i >= 50:
            break
            
        data = data.to(device)
        labels = labels.to(device)
        
        # Extract features
        with torch.no_grad():
            features = backbone.extract_features(data, summary=False)
        
        # Forward
        logits = probe(features)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Store actual loss value
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Check labels
        unique_labels = torch.unique(labels).tolist()
        all_labels.extend(labels.cpu().numpy())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = torch.stack([p.grad.abs().mean() for p in probe.parameters()]).mean().item()
        
        optimizer.step()
        
        # Print every 10 batches
        if i % 10 == 0:
            print(f"\nBatch {i}:")
            print(f"  Loss (actual): {loss_val:.10e}")
            print(f"  Loss (formatted .4f): {loss_val:.4f}")
            print(f"  Loss == 0? {loss_val == 0.0}")
            print(f"  Loss < 1e-4? {loss_val < 1e-4}")
            print(f"  Unique labels: {unique_labels}")
            print(f"  Gradient norm: {grad_norm:.8e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    losses = np.array(losses)
    print(f"Loss stats: mean={losses.mean():.8e}, std={losses.std():.8e}")
    print(f"Loss range: [{losses.min():.8e}, {losses.max():.8e}]")
    print(f"Exactly zero: {np.sum(losses == 0)}/{len(losses)}")
    print(f"Less than 1e-4: {np.sum(losses < 1e-4)}/{len(losses)}")
    print(f"Less than 1e-6: {np.sum(losses < 1e-6)}/{len(losses)}")
    
    # Label distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({100*count/len(all_labels):.1f}%)")
    
    print("\n" + "="*60)
    if np.all(losses == 0):
        print("❌ CRITICAL: All losses are EXACTLY ZERO - there's a bug!")
    elif losses.mean() < 1e-6:
        print("⚠️ Loss is extremely small - likely just display rounding")
        print("FIX: Change tqdm format from .4f to .6e or .8f")
    else:
        print("✅ Loss values are normal - just a display issue")
        print(f"Actual loss: {losses.mean():.8e}")

if __name__ == "__main__":
    main()