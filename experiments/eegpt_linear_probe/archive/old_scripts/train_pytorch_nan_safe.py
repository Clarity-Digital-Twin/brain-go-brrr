#!/usr/bin/env python
"""
NaN-Safe PyTorch Training Script for EEGPT Linear Probe
========================================================

This script includes COMPREHENSIVE NaN protection:
- Input data validation
- Gradient clipping
- Anomaly detection
- Safe numerical operations
- Checkpoint recovery from NaN crashes

Author: Brain-Go-Brrr Team
Date: August 2025
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import json
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Project imports - use absolute imports
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt

# Import collate from same directory
sys.path.insert(0, str(Path(__file__).parent))
from custom_collate_fixed import collate_eeg_batch_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaNDetector:
    """Utility class to detect NaN/Inf in training."""
    
    @staticmethod
    def check_tensor(tensor, name="tensor"):
        """Check if tensor contains NaN or Inf values."""
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            raise RuntimeError(
                f"NaN/Inf detected in {name}: "
                f"NaN count={nan_count}, Inf count={inf_count}, "
                f"shape={tensor.shape}, dtype={tensor.dtype}"
            )
        
        # Also check for extreme values
        max_val = tensor.abs().max().item()
        if max_val > 1e6:
            logger.warning(f"Large values in {name}: max={max_val:.2e}")
        
        return True
    
    @staticmethod
    def check_gradients(model, max_grad_norm=10.0):
        """Check if gradients are finite and not too large."""
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_count += 1
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                if not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"NaN/Inf gradient in {name}")
                
                if param_norm > max_grad_norm:
                    logger.warning(f"Large gradient in {name}: norm={param_norm:.2e}")
        
        total_norm = total_norm ** 0.5
        return total_norm


def create_dataloaders(data_root, batch_size=32, num_workers=2):
    """Create train and validation dataloaders with NaN-safe settings."""
    logger.info("Creating NaN-safe dataloaders...")
    
    # Train dataset
    train_dataset = TUABCachedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=8.0,
        window_stride=4.0,
        sampling_rate=256,
        preload=False,
        normalize=True,
        cache_dir=data_root / "cache/tuab_enhanced",
        cache_index_path=data_root / "cache/tuab_index.json"  # Specify exact path
    )
    
    # Val dataset
    val_dataset = TUABCachedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="eval",
        window_duration=8.0,
        window_stride=8.0,  # No overlap for validation
        sampling_rate=256,
        preload=False,
        normalize=True,
        cache_dir=data_root / "cache/tuab_enhanced",
        cache_index_path=data_root / "cache/tuab_index.json"  # Specify exact path
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders with safe settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),  # Only if we have workers
        drop_last=True,
        collate_fn=collate_eeg_batch_fixed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),  # Only if we have workers
        collate_fn=collate_eeg_batch_fixed
    )
    
    # Validate first batch
    logger.info("Validating first batch...")
    first_batch = next(iter(train_loader))
    x, y = first_batch
    NaNDetector.check_tensor(x, "first_batch_data")
    logger.info(f"First batch validated: shape={x.shape}, labels={y}")
    
    return train_loader, val_loader


def create_model(checkpoint_path, device):
    """Create EEGPT backbone and probe with NaN-safe initialization."""
    logger.info("Creating NaN-safe model...")
    
    # Create backbone
    backbone = create_normalized_eegpt(
        checkpoint_path=str(checkpoint_path),
        normalize=True  # TUAB data is already normalized
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Create probe
    probe = EEGPTTwoLayerProbe(
        backbone_dim=768,  # EEGPT feature dimension
        hidden_dim=16,
        n_classes=2,
        dropout=0.5,
        n_input_channels=20,  # Some samples have 20 channels
        n_adapted_channels=19,  # EEGPT expects 19
        use_channel_adapter=True
    )
    probe = probe.to(device)
    
    # Check model initialization
    for name, param in probe.named_parameters():
        NaNDetector.check_tensor(param, f"init_{name}")
    
    logger.info("Model created and validated")
    return backbone, probe


def safe_forward(backbone, probe, x, check_intermediate=False):
    """Forward pass with NaN checking at each step."""
    # Check input
    NaNDetector.check_tensor(x, "model_input")
    
    # Get features from backbone
    with torch.no_grad():
        features = backbone(x)
    
    if check_intermediate:
        NaNDetector.check_tensor(features, "backbone_features")
    
    # Get predictions from probe
    logits = probe(features)
    
    if check_intermediate:
        NaNDetector.check_tensor(logits, "probe_logits")
    
    return logits


def train_epoch(backbone, probe, train_loader, criterion, optimizer, device, epoch, 
                gradient_clip=1.0, max_batches=None):
    """Train for one epoch with comprehensive NaN protection."""
    probe.train()
    epoch_loss = 0
    correct = 0
    total = 0
    nan_batches = 0
    
    # Progress bar
    pbar = tqdm(enumerate(train_loader), total=max_batches or len(train_loader), 
                desc=f"Epoch {epoch}")
    
    for batch_idx, (x, y) in pbar:
        if max_batches and batch_idx >= max_batches:
            break
        
        try:
            x, y = x.to(device), y.to(device)
            
            # Forward pass with NaN checking
            logits = safe_forward(backbone, probe, x, check_intermediate=(batch_idx % 100 == 0))
            
            # Compute loss with numerical stability
            loss = criterion(logits, y)
            NaNDetector.check_tensor(loss, "loss")
            
            # Backward pass with anomaly detection
            optimizer.zero_grad()
            
            if batch_idx == 0:  # Extra careful on first batch
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            
            # Check gradients before clipping
            grad_norm = NaNDetector.check_gradients(probe)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(probe.parameters(), gradient_clip)
            
            # Optimizer step
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total,
                'grad_norm': f'{grad_norm:.2f}'
            })
            
        except RuntimeError as e:
            if "NaN" in str(e) or "Inf" in str(e):
                nan_batches += 1
                logger.error(f"NaN/Inf detected in batch {batch_idx}: {e}")
                if nan_batches > 5:
                    raise RuntimeError("Too many NaN batches, stopping training")
                continue
            else:
                raise
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    if nan_batches > 0:
        logger.warning(f"Epoch {epoch} had {nan_batches} NaN batches")
    
    return avg_loss, accuracy


def validate(backbone, probe, val_loader, criterion, device):
    """Validate with NaN protection."""
    probe.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            
            try:
                logits = safe_forward(backbone, probe, x)
                loss = criterion(logits, y)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of abnormal
                
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
            except RuntimeError as e:
                if "NaN" in str(e) or "Inf" in str(e):
                    logger.error(f"NaN/Inf in validation: {e}")
                    continue
                else:
                    raise
    
    avg_loss = val_loss / len(val_loader)
    
    # Compute metrics safely
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Filter out NaN predictions
    valid_mask = np.isfinite(all_preds)
    if not valid_mask.all():
        logger.warning(f"Filtering {(~valid_mask).sum()} NaN predictions")
        all_preds = all_preds[valid_mask]
        all_labels = all_labels[valid_mask]
    
    auroc = roc_auc_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds > 0.5)
    
    return avg_loss, auroc, balanced_acc


def save_checkpoint(probe, optimizer, epoch, best_auroc, output_dir, is_best=False):
    """Save model checkpoint with NaN checking."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': probe.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auroc': best_auroc,
    }
    
    # Check for NaN in state dict
    for key, value in checkpoint['model_state_dict'].items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"NaN in checkpoint: {key}")
    
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = output_dir / 'best_model.pt'
        shutil.copy(checkpoint_path, best_path)
        logger.info(f"Saved best model with AUROC: {best_auroc:.4f}")


def main():
    """Main training function with NaN protection."""
    # Configuration
    data_root = Path(os.environ.get("BGB_DATA_ROOT", "data"))
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    # Training parameters with SAFE defaults
    batch_size = 32
    epochs = 50
    learning_rate = 2e-4  # Reduced from 5e-4
    weight_decay = 0.05
    gradient_clip = 1.0
    warmup_epochs = 5
    num_workers = 2  # Not 0 to avoid the persistent_workers bug
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("output") / f"nan_safe_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("NaN-Safe EEGPT Linear Probe Training")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Gradient clip: {gradient_clip}")
    logger.info(f"Num workers: {num_workers}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_root, batch_size, num_workers)
    
    # Test first batch with anomaly detection
    logger.info("Testing first batch with anomaly detection...")
    test_batch = next(iter(train_loader))
    test_x, test_y = test_batch[0].to(device), test_batch[1].to(device)
    
    # Create model
    backbone, probe = create_model(checkpoint_path, device)
    
    # Test forward pass
    with torch.autograd.detect_anomaly():
        test_logits = safe_forward(backbone, probe, test_x, check_intermediate=True)
        test_loss = nn.CrossEntropyLoss()(test_logits, test_y)
        logger.info(f"Test batch passed: loss={test_loss.item():.4f}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate schedule with warmup
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], 
                            milestones=[warmup_epochs])
    
    # Training loop
    best_auroc = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_balanced_acc": []}
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_loss, train_acc = train_epoch(
            backbone, probe, train_loader, criterion, optimizer, device, epoch,
            gradient_clip=gradient_clip
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
        
        # Save checkpoint
        is_best = val_auroc > best_auroc
        if is_best:
            best_auroc = val_auroc
        
        save_checkpoint(probe, optimizer, epoch, best_auroc, output_dir, is_best)
        
        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Early stopping if loss is NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            logger.error("NaN loss detected, stopping training")
            break
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()