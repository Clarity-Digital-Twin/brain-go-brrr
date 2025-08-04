#!/usr/bin/env python
"""
PyTorch Training Script for EEGPT Linear Probe
==============================================

This script replaces PyTorch Lightning due to a critical bug in Lightning 2.5.2
where training hangs indefinitely at "Loading train_dataloader to estimate number 
of stepping batches" when using large cached datasets.

Bug Details:
- Lightning 2.5.2 hangs during dataloader length estimation
- Occurs with datasets > 100k samples using cached data
- Cannot be fixed with any Lightning settings (deterministic, max_steps, etc.)
- See: experiments/eegpt_linear_probe/LIGHTNING_BUG_REPORT.md

Author: Brain-Go-Brrr Team
Date: August 2025
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Project imports
from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from src.brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from src.brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt
from experiments.eegpt_linear_probe.custom_collate_fixed import collate_eeg_batch_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataloaders(data_root: Path, batch_size: int = 32, num_workers: int = 0):
    """Create train and validation dataloaders.
    
    WARNING: Do NOT use num_workers > 0 until multiprocessing issues are resolved.
    Lightning was hanging due to worker spawning issues.
    """
    # Training dataset
    train_dataset = TUABCachedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=8.0,
        window_stride=4.0,
        sampling_rate=256,
        cache_dir=data_root / "cache/tuab_enhanced",
        cache_index_path=data_root / "cache/tuab_index.json"
    )
    
    # Validation dataset
    val_dataset = TUABCachedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="eval",
        window_duration=8.0,
        window_stride=8.0,  # No overlap for validation
        sampling_rate=256,
        cache_dir=data_root / "cache/tuab_enhanced",
        cache_index_path=data_root / "cache/tuab_index.json"
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} windows")
    logger.info(f"Val dataset: {len(val_dataset)} windows")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # WARNING: Keep at 0 to avoid hangs
        collate_fn=collate_eeg_batch_fixed,
        pin_memory=True,
        persistent_workers=False  # Must be False when num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eeg_batch_fixed,
        pin_memory=True,
        persistent_workers=False
    )
    
    return train_loader, val_loader


def create_model(checkpoint_path: Path, device: torch.device):
    """Create EEGPT backbone and linear probe.
    
    Architecture matches the paper:
    - Frozen EEGPT backbone (25.3M params)
    - Two-layer probe with dropout (34.2K params)
    - Channel adapter for 19 -> 22 -> 19 channels
    """
    # Load pretrained EEGPT backbone
    backbone = create_normalized_eegpt(str(checkpoint_path))
    backbone = backbone.to(device)
    backbone.eval()  # Always in eval mode (frozen)
    
    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Create two-layer probe
    probe = EEGPTTwoLayerProbe(
        n_input_channels=19,      # TUAB has 19 channels
        n_adapted_channels=19,    # Output 19 channels
        hidden_dim=16,            # Hidden layer size
        n_classes=2,              # Binary classification
        dropout=0.5,              # Dropout rate
        use_channel_adapter=True  # Enable channel adaptation
    ).to(device)
    
    logger.info(f"Model created - Backbone: frozen, Probe: {sum(p.numel() for p in probe.parameters() if p.requires_grad)} params")
    
    return backbone, probe


def validate(backbone, probe, val_loader, criterion, device):
    """Run validation and compute metrics."""
    probe.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            features = backbone(x)
            logits = probe(features)
            loss = criterion(logits, y)
            
            val_loss += loss.item()
            
            # Store predictions
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of abnormal
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Compute metrics
    val_loss /= len(val_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds > 0.5)
    
    probe.train()
    return val_loss, auroc, balanced_acc


def train_epoch(backbone, probe, train_loader, criterion, optimizer, device, epoch, max_batches=None):
    """Train for one epoch."""
    probe.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(enumerate(train_loader), total=max_batches or len(train_loader), 
                desc=f"Epoch {epoch}")
    
    for batch_idx, (x, y) in pbar:
        if max_batches and batch_idx >= max_batches:
            break
            
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        with torch.no_grad():
            features = backbone(x)
        logits = probe(features)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        epoch_loss += loss.item()
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")
    
    epoch_loss /= (batch_idx + 1)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """Main training loop."""
    # Configuration
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training hyperparameters (matching paper)
    epochs = 50
    batch_size = 32
    learning_rate = 2e-4
    weight_decay = 0.05
    warmup_epochs = 5
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/eegpt_linear_probe/runs/pytorch_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("EEGPT Linear Probe Training (Pure PyTorch)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_root, batch_size)
    
    # Create model
    backbone, probe = create_model(checkpoint_path, device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_auroc = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_balanced_acc": []}
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            backbone, probe, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_auroc, val_balanced_acc = validate(
            backbone, probe, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_balanced_acc"].append(val_balanced_acc)
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_balanced_acc': val_balanced_acc,
            }, output_dir / 'best_model.pt')
            logger.info(f"Saved best model with AUROC: {val_auroc:.4f}")
    
    # Save final results
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"Training completed! Best AUROC: {best_auroc:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()