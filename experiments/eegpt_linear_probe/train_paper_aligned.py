#!/usr/bin/env python
"""Train EEGPT linear probe with paper-aligned settings for TUAB abnormality detection."""

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

# Import custom dataset and collate
sys.path.insert(0, str(Path(__file__).parent))
from tuab_simple_cached import TUABSimpleCachedDataset
from custom_collate_fixed import collate_eeg_batch_fixed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Two-layer linear probe with channel adapter."""
    
    def __init__(self, config):
        super().__init__()
        
        # Channel adapter (1x1 conv)
        if config['probe']['use_channel_adapter']:
            self.channel_adapter = nn.Conv1d(
                config['probe']['channel_adapter_in'],
                config['probe']['channel_adapter_out'],
                kernel_size=1
            )
        else:
            self.channel_adapter = None
            
        # Two-layer probe
        self.probe = nn.Sequential(
            nn.Linear(config['probe']['input_dim'], config['probe']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['probe']['dropout']),
            nn.Linear(config['probe']['hidden_dim'], config['probe']['n_classes'])
        )
        
    def forward(self, features):
        """Forward pass through probe."""
        # features: (batch_size, n_summary_tokens, embed_dim)
        # Average pool over summary tokens
        x = features.mean(dim=1)  # (batch_size, embed_dim)
        return self.probe(x)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables
    def resolve_env_vars(obj):
        if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        elif isinstance(obj, dict):
            return {k: resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_env_vars(item) for item in obj]
        return obj
    
    return resolve_env_vars(config)


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    # Resolve environment variables in paths
    data_root = os.environ.get('BGB_DATA_ROOT', '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data')
    cache_dir = Path(data_root) / "cache" / "tuab_4s_final"
    
    # Create datasets using the simple cached dataset
    train_dataset = TUABSimpleCachedDataset(
        cache_dir=cache_dir,
        split='train'
    )
    
    # Validation dataset
    val_dataset = TUABSimpleCachedDataset(
        cache_dir=cache_dir,
        split='eval'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=config['data'].get('persistent_workers', False) if config['data'].get('num_workers', 0) > 0 else False,
        prefetch_factor=config['data'].get('prefetch_factor', 2) if config['data'].get('num_workers', 0) > 0 else None,
        collate_fn=collate_eeg_batch_fixed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=config['data'].get('persistent_workers', False) if config['data'].get('num_workers', 0) > 0 else False,
        prefetch_factor=config['data'].get('prefetch_factor', 2) if config['data'].get('num_workers', 0) > 0 else None,
        collate_fn=collate_eeg_batch_fixed
    )
    
    return train_loader, val_loader


def train_epoch(model, probe, train_loader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    model.eval()  # Backbone stays frozen
    probe.train()
    
    losses = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward through frozen backbone
        with torch.no_grad():
            features = model(data)
            
        # Forward through probe
        logits = probe(features)
        
        # Compute loss
        if config['training'].get('weighted_loss', False):
            # Compute class weights
            class_counts = torch.bincount(labels)
            class_weights = 1.0 / (class_counts.float() + 1e-5)
            class_weights = class_weights / class_weights.sum()
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip_val'] > 0:
            torch.nn.utils.clip_grad_norm_(
                probe.parameters(), 
                config['training']['gradient_clip_val']
            )
            
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        losses.append(loss.item())
        preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
    # Compute epoch metrics
    auroc = roc_auc_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return {
        'loss': np.mean(losses),
        'auroc': auroc,
        'bacc': bacc
    }


def validate(model, probe, val_loader, device):
    """Validate the model."""
    model.eval()
    probe.eval()
    
    losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc='Validation'):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward
            features = model(data)
            logits = probe(features)
            loss = F.cross_entropy(logits, labels)
            
            # Track metrics
            losses.append(loss.item())
            preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return {
        'loss': np.mean(losses),
        'auroc': auroc,
        'bacc': bacc
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='configs/tuab_4s_paper_aligned.yaml',
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"output/{config['experiment']['name']}_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Update steps per epoch in config
    config['training']['scheduler']['steps_per_epoch'] = len(train_loader)
    
    # Create model
    # Resolve model checkpoint path
    model_checkpoint = config['model']['backbone']['checkpoint_path']
    if '${BGB_DATA_ROOT}' in model_checkpoint:
        data_root = os.environ.get('BGB_DATA_ROOT', '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data')
        model_checkpoint = model_checkpoint.replace('${BGB_DATA_ROOT}', data_root)
    
    backbone = EEGPTWrapper(
        checkpoint_path=model_checkpoint
    )
    backbone.to(device)
    backbone.eval()  # Freeze backbone
    
    probe = LinearProbe(config['model'])
    probe.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Create scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(config['training']['scheduler']['max_lr']),
        epochs=config['training']['scheduler']['epochs'],
        steps_per_epoch=config['training']['scheduler']['steps_per_epoch'],
        pct_start=config['training']['scheduler']['pct_start'],
        anneal_strategy=config['training']['scheduler']['anneal_strategy'],
        div_factor=config['training']['scheduler']['div_factor'],
        final_div_factor=config['training']['scheduler']['final_div_factor']
    )
    
    # Training loop
    best_val_auroc = 0
    patience_counter = 0
    
    for epoch in range(config['training']['max_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            backbone, probe, train_loader, optimizer, scheduler, device, config
        )
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"AUROC: {train_metrics['auroc']:.4f}, "
                   f"BACC: {train_metrics['bacc']:.4f}")
        
        # Validate
        if (epoch + 1) % 2 == 0:  # Validate every 2 epochs
            val_metrics = validate(backbone, probe, val_loader, device)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"AUROC: {val_metrics['auroc']:.4f}, "
                       f"BACC: {val_metrics['bacc']:.4f}")
            
            # Save checkpoint if best
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'probe_state_dict': probe.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_auroc': val_metrics['auroc'],
                    'val_bacc': val_metrics['bacc'],
                    'config': config
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                logger.info(f"Saved best model with AUROC: {val_metrics['auroc']:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info("Early stopping triggered")
                break
                
    logger.info(f"\nTraining complete! Best AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    main()