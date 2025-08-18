"""TUEV Training Script - Following Table 13 Architecture Exactly.

Based on TUEV_UNIFIED_SPECS.md:
- Architecture from Table 13: 23→20 channels, kernel 55, dropout 0.5
- Batch size: 500, Learning rate: 5e-4 (constant)
- 6-class event detection
- Run 3 times with different seeds (paper protocol)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

from tuev_dataset import TUEVDataset
from tuev_dataset_cached import TUEVCachedDatasetPadded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TUEVLinearProbe(nn.Module):
    """TUEV Linear Probe - EXACT Table 13 Architecture."""
    
    def __init__(self, eegpt_checkpoint: str, device: str = 'cuda'):
        """Initialize with frozen EEGPT backbone and TUEV-specific layers.
        
        Architecture from Table 13 (lines 606-613):
        1. Conv1d: 23→20 channels (kernel=1)
        2. BatchNorm + GELU
        3. Conv1d: Depthwise temporal (kernel=55, groups=20, padding=27)
        4. BatchNorm + GELU
        5. Dropout(0.5)
        6. EEGPT encoder (frozen)
        7. Linear: 15×4×512 → 6 classes
        """
        super().__init__()
        
        # Load frozen EEGPT backbone
        self.eegpt = EEGPTWrapper(checkpoint_path=eegpt_checkpoint)
        self.eegpt.model.eval()
        for param in self.eegpt.model.parameters():
            param.requires_grad = False
        self.eegpt.model = self.eegpt.model.to(device)
        
        # Layer 1: Channel reduction (23 → 20)
        self.channel_reducer = nn.Conv1d(
            in_channels=23,
            out_channels=20,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm1d(20)
        
        # Layer 2: Temporal convolution (depthwise)
        self.temporal_conv = nn.Conv1d(
            in_channels=20,
            out_channels=20,
            kernel_size=55,  # CRITICAL: 55, not 15!
            stride=1,
            groups=20,  # Depthwise convolution
            padding=27  # Maintains size
        )
        self.bn2 = nn.BatchNorm1d(20)
        
        # Dropout - CRITICAL: 0.5 for TUEV, not 0.25!
        self.dropout = nn.Dropout(0.5)
        
        # Linear probe (16×4×512 → 6) for 4s windows at 256Hz
        # 1024 samples / 64 = 16 temporal patches
        # 16 patches × 4 summary tokens × 512 dim = 32,768 features
        self.classifier = nn.Linear(16 * 4 * 512, 6)
        
        self.device = device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass following Table 13 exactly.
        
        Args:
            x: Input tensor of shape (batch, 23, 1024)
            
        Returns:
            Logits of shape (batch, 6)
        """
        # Verify input shape  
        assert x.shape[1:] == (23, 1024), f"Wrong input shape: {x.shape}, expected (batch, 23, 1024)"
        
        # Channel reduction: 23 → 20
        x = self.channel_reducer(x)  # (batch, 20, 1024)
        x = F.gelu(self.bn1(x))
        
        # Temporal convolution (depthwise)
        x = self.temporal_conv(x)  # (batch, 20, 1024)
        x = F.gelu(self.bn2(x))
        
        # Dropout
        x = self.dropout(x)  # (batch, 20, 1024)
        
        # EEGPT encoder (frozen) - get all temporal features
        with torch.no_grad():
            # EEGPT expects (batch, channels, time)
            features = self.eegpt.extract_features(x, return_all_temporal=True)  # (batch, 16, 4, 512)
        
        # Flatten all temporal and summary features
        features = features.view(features.size(0), -1)  # (batch, 32768)
        logits = self.classifier(features)  # (batch, 6)
        
        return logits


def train_epoch(
    model: TUEVLinearProbe,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    # Keep EEGPT frozen
    model.eegpt.model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train")
    
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(data)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track predictions
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{(preds == labels).float().mean():.3f}"
        })
    
    # Compute metrics
    metrics = {
        'loss': total_loss / len(train_loader),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds)
    }
    
    return metrics


def evaluate(
    model: TUEVLinearProbe,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d} Eval")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(data)
            loss = criterion(logits, labels)
            
            # Track predictions
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(preds == labels).float().mean():.3f}"
            })
    
    # Compute metrics
    metrics = {
        'loss': total_loss / len(val_loader),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'cohen_kappa': cohen_kappa_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    class_names = ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG']
    for i, name in enumerate(class_names):
        metrics[f'f1_{name}'] = per_class_f1[i] if i < len(per_class_f1) else 0.0
    
    return metrics


def main(args):
    """Main training loop."""
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/tuev_{timestamp}_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting TUEV training with seed {args.seed}")
    logger.info(f"Config: {config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Critical assertions from Table 13
    assert config.data.batch_size == 500, "Batch size must be 500 (paper line 587)"
    assert config.training.learning_rate == 5e-4, "LR must be 5e-4 (paper line 587)"
    
    # Create datasets
    if args.use_cache:
        logger.info("Using cached dataset with padding to 1024 samples")
        train_dataset = TUEVCachedDatasetPadded(
            cache_dir=Path(config.data.cache_dir),
            split='train',
            padding='edge'  # Repeat last samples for padding
        )
        val_dataset = TUEVCachedDatasetPadded(
            cache_dir=Path(config.data.cache_dir),
            split='eval',
            padding='edge'
        )
    else:
        logger.info("Loading dataset from EDF files")
        train_dataset = TUEVDataset(
            root_dir=Path(config.data.root_dir),
            split='train'
        )
        val_dataset = TUEVDataset(
            root_dir=Path(config.data.root_dir),
            split='eval'
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(val_dataset)}")
    
    # Create model
    model = TUEVLinearProbe(
        eegpt_checkpoint=config.model.eegpt_checkpoint,
        device=args.device
    )
    
    # Setup loss (weighted for class imbalance)
    class_weights = train_dataset.get_class_weights().to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class weights: {class_weights}")
    
    # Setup optimizer 
    # [Paper] says "same optimizer" and lr=5e-4 but doesn't name it
    # [Decision] Use AdamW (from pretraining) with constant LR (schedule not specified)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Training loop
    best_balanced_acc = 0
    best_epoch = 0
    history = []
    
    logger.info("=" * 50)
    logger.info("TUEV Training - Target Performance (from paper):")
    logger.info("  Balanced Accuracy: 0.6232 ± 0.0114")
    logger.info("  Weighted F1:       0.8187 ± 0.0063")
    logger.info("  Cohen's Kappa:     0.6351 ± 0.0134")
    logger.info("=" * 50)
    
    for epoch in range(1, config.training.n_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, criterion, args.device, epoch
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch:03d}:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"BAcc: {train_metrics['balanced_acc']:.4f}, "
                   f"F1: {train_metrics['weighted_f1']:.4f}, "
                   f"Kappa: {train_metrics['cohen_kappa']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"BAcc: {val_metrics['balanced_acc']:.4f}, "
                   f"F1: {val_metrics['weighted_f1']:.4f}, "
                   f"Kappa: {val_metrics['cohen_kappa']:.4f}")
        
        # Save best model
        if val_metrics['balanced_acc'] > best_balanced_acc:
            best_balanced_acc = val_metrics['balanced_acc']
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': OmegaConf.to_container(config)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"  → New best model! BAcc: {best_balanced_acc:.4f}")
        
        # Store history
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Final report
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Best Balanced Accuracy: {best_balanced_acc:.4f} (epoch {best_epoch})")
    logger.info(f"Target from paper:      0.6232")
    logger.info(f"Achievement rate:       {best_balanced_acc/0.6232*100:.1f}%")
    logger.info("=" * 50)
    
    return best_balanced_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TUEV Linear Probe")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-cache', action='store_true', help='Use cached dataset')
    
    args = parser.parse_args()
    
    # Run training
    main(args)