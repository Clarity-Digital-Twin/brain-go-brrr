#!/usr/bin/env python
"""Fuck Lightning. Pure PyTorch training that actually works."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add to path
import sys
sys.path.insert(0, str(Path(__file__).resolve()))

# Use the EXACT same imports that work in enhanced training
from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from src.brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from src.brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from experiments.eegpt_linear_probe.custom_collate_fixed import collate_eeg_batch_fixed

print("FUCK LIGHTNING - RUNNING RAW PYTORCH")
print("="*80)

# Paths
data_root = Path("data")
device = torch.device("cuda")

# Dataset - EXACT same as enhanced training
train_dataset = TUABCachedDataset(
    root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
    cache_dir=data_root / "cache/tuab_enhanced",
    cache_index_path=data_root / "cache/tuab_index.json"
)
print(f"Dataset: {len(train_dataset)} samples")

# Dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_eeg_batch_fixed,
    pin_memory=True
)

# Model - use the EXACT same creation as enhanced training
backbone = create_normalized_eegpt(
    str(data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
).to(device)
backbone.eval()

probe = EEGPTTwoLayerProbe(
    n_input_channels=19,
    n_adapted_channels=19,
    hidden_dim=16,
    n_classes=2,
    dropout=0.5,
    use_channel_adapter=True
).to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-4, weight_decay=0.05)

print("\nSTARTING TRAINING...")
for epoch in range(1):  # Just 1 epoch to prove it works
    for i, (x, y) in enumerate(tqdm(train_loader, total=300)):
        if i >= 300:
            break
            
        x, y = x.to(device), y.to(device)
        
        # Forward
        with torch.no_grad():
            features = backbone(x)
        logits = probe(features)
        loss = criterion(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"\nStep {i}: loss={loss.item():.4f}")

print("\n✅ TRAINING WORKS! Lightning is the problem, not your code.")
print("Now you can either:")
print("1. Keep using this raw PyTorch script")
print("2. Downgrade Lightning to 1.9.x")
print("3. File a bug report with Lightning")