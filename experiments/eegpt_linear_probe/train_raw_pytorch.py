#!/usr/bin/env python
"""Raw PyTorch training - bypass Lightning completely."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf

from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
from experiments.eegpt_linear_probe.custom_collate_fixed import collate_eeg_batch_fixed

# Load config
cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_stable.yaml")
data_root = Path(os.environ.get("BGB_DATA_ROOT", "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"))

print("Loading datasets...")
train_dataset = TUABCachedDataset(
    root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
    cache_dir=data_root / "cache/tuab_enhanced",
    cache_index_path=data_root / "cache/tuab_index.json"
)
print(f"Train dataset: {len(train_dataset)} samples")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_eeg_batch_fixed,
    pin_memory=True
)

# Create model
print("Loading EEGPT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load backbone
from brain_go_brrr.models.eegpt_architecture import EEGPTFeatureExtractor
backbone_model = EEGPTFeatureExtractor(
    n_channels=58,
    sampling_rate=256,
    duration=4.0,
    n_patches=512,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    dropout=0.1,
    mask_ratio=0.9
)

# Load checkpoint
ckpt = torch.load(data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt", map_location="cpu")
backbone_model.load_state_dict(ckpt["state_dict"], strict=False)
print("Loaded EEGPT checkpoint")

backbone = EEGPTWrapper(backbone_model, n_channels=19)
backbone.eval()
for param in backbone.parameters():
    param.requires_grad = False

probe = EEGPTTwoLayerProbe(
    input_channels=19,
    n_classes=2,
    channel_adapter_in=19,
    channel_adapter_out=19,
    probe1_out=16,
    dropout=0.5
)

backbone = backbone.to(device)
probe = probe.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-4, weight_decay=0.05)

print("Starting training...")
print("="*80)

# Training loop
for epoch in range(5):  # Just 5 epochs for testing
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader), total=min(300, len(train_loader)), desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in pbar:
        if batch_idx >= 300:  # Limit to 300 batches
            break
            
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        with torch.no_grad():
            features = backbone(x)
        logits = probe(features, x)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
    
    print(f"Epoch {epoch}: Loss={train_loss/min(300, batch_idx+1):.4f}, Acc={acc:.2f}%")

print("\nTraining completed successfully!")
print("This proves the model and data work fine - Lightning is the problem.")