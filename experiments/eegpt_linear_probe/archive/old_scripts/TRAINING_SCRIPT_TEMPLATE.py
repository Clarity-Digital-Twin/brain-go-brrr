#!/usr/bin/env python
"""
Professional Training Script Template for EEGPT Linear Probes
=============================================================

This template includes all fixes and best practices learned from debugging.
Copy this file and modify for new experiments.

Usage:
    python train_[experiment_name].py --config configs/[config_name].yaml

Author: [Your Name]
Date: [Creation Date]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Project imports
from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from custom_collate_fixed import collate_eeg_batch_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Log to file will be added after output_dir is created
    ]
)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Linear probe for EEGPT features."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        probe_config = config['probe']
        
        # Optional channel adapter
        if probe_config.get('use_channel_adapter', False):
            self.channel_adapter = nn.Conv1d(
                probe_config['channel_adapter_in'],
                probe_config['channel_adapter_out'],
                kernel_size=1
            )
        else:
            self.channel_adapter = None
        
        # Probe layers
        layers = []
        input_dim = probe_config['input_dim']
        
        if probe_config['type'] == 'linear':
            layers.append(nn.Linear(input_dim, probe_config['n_classes']))
        elif probe_config['type'] == 'two_layer':
            layers.extend([
                nn.Linear(input_dim, probe_config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(probe_config['dropout']),
                nn.Linear(probe_config['hidden_dim'], probe_config['n_classes'])
            ])
        else:
            raise ValueError(f"Unknown probe type: {probe_config['type']}")
        
        self.probe = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, n_summary_tokens, embed_dim)
        Returns:
            logits: (batch_size, n_classes)
        """
        # Average pool over summary tokens
        x = features.mean(dim=1)  # (batch_size, embed_dim)
        return self.probe(x)


class Trainer:
    """Professional trainer class with all fixes applied."""
    
    def __init__(self, config: Dict, output_dir: Path, device: torch.device):
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.start_time = time.time()
        
        # Setup file logging
        log_file = output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Initialize tracking
        self.history = {
            'train_loss': [],
            'train_auroc': [],
            'val_loss': [],
            'val_auroc': [],
            'val_bacc': [],
            'lr': []
        }
        
    def save_checkpoint(self, 
                       epoch: int,
                       model: nn.Module,
                       probe: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'probe_state_dict': probe.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train_epoch(self,
                   model: nn.Module,
                   probe: nn.Module,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.eval()  # Keep backbone frozen
        probe.train()
        
        losses = []
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                features = model(data)
            
            logits = probe(features)
            
            # Compute loss
            if self.config['training'].get('weighted_loss', False):
                # Handle class imbalance
                class_counts = torch.bincount(labels)
                class_weights = 1.0 / (class_counts.float() + 1e-5)
                class_weights = class_weights / class_weights.sum() * 2
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip_val'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    probe.parameters(),
                    self.config['training']['gradient_clip_val']
                )
            
            optimizer.step()
            
            # Update scheduler
            if scheduler and self.config['training']['scheduler']['name'] == 'OneCycleLR':
                scheduler.step()
            
            # Track metrics
            losses.append(loss.item())
            preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Compute epoch metrics
        epoch_loss = np.mean(losses)
        epoch_auroc = roc_auc_score(all_labels, all_preds)
        epoch_bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)
        
        return {
            'loss': epoch_loss,
            'auroc': epoch_auroc,
            'bacc': epoch_bacc,
            'lr': optimizer.param_groups[0]['lr']
        }
    
    def validate(self,
                model: nn.Module,
                probe: nn.Module,
                val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        model.eval()
        probe.eval()
        
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc='Validation'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                features = model(data)
                logits = probe(features)
                loss = F.cross_entropy(logits, labels)
                
                # Track metrics
                losses.append(loss.item())
                preds = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        val_loss = np.mean(losses)
        val_auroc = roc_auc_score(all_labels, all_preds)
        val_bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)
        
        return {
            'loss': val_loss,
            'auroc': val_auroc,
            'bacc': val_bacc
        }
    
    def train(self,
             model: nn.Module,
             probe: nn.Module,
             train_loader: DataLoader,
             val_loader: DataLoader,
             optimizer: torch.optim.Optimizer,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]):
        """Main training loop."""
        best_val_auroc = 0
        patience_counter = 0
        
        for epoch in range(self.config['training']['max_epochs']):
            # Train
            train_metrics = self.train_epoch(
                model, probe, train_loader, optimizer, scheduler, epoch
            )
            
            # Log training metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['max_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train AUROC: {train_metrics['auroc']:.4f}, "
                f"LR: {train_metrics['lr']:.2e}"
            )
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auroc'].append(train_metrics['auroc'])
            self.history['lr'].append(train_metrics['lr'])
            
            # Validate
            if (epoch + 1) % self.config['training'].get('val_check_interval', 1) == 0:
                val_metrics = self.validate(model, probe, val_loader)
                
                logger.info(
                    f"Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"AUROC: {val_metrics['auroc']:.4f}, "
                    f"BACC: {val_metrics['bacc']:.4f}"
                )
                
                # Update history
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_auroc'].append(val_metrics['auroc'])
                self.history['val_bacc'].append(val_metrics['bacc'])
                
                # Save checkpoint
                is_best = val_metrics['auroc'] > best_val_auroc
                if is_best:
                    best_val_auroc = val_metrics['auroc']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                self.save_checkpoint(
                    epoch, model, probe, optimizer, scheduler,
                    val_metrics, is_best
                )
                
                # Early stopping
                if patience_counter >= self.config['training']['early_stopping']['patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Update scheduler (epoch-based)
            if scheduler and self.config['training']['scheduler']['name'] != 'OneCycleLR':
                scheduler.step()
            
            # Save history
            with open(self.output_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Training complete
        elapsed_time = (time.time() - self.start_time) / 3600
        logger.info(
            f"Training complete! Best AUROC: {best_val_auroc:.4f}, "
            f"Time: {elapsed_time:.2f} hours"
        )


def resolve_env_vars(obj):
    """Recursively resolve environment variables in config."""
    if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        env_var = obj[2:-1]
        return os.environ.get(env_var, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with all fixes applied."""
    # Resolve environment variables
    data_root = os.environ.get('BGB_DATA_ROOT', '/default/path')
    cache_index_path = Path(data_root) / "cache" / "tuab_index.json"
    
    # Resolve paths in config
    root_dir = config['data']['root_dir']
    if '${BGB_DATA_ROOT}' in root_dir:
        root_dir = root_dir.replace('${BGB_DATA_ROOT}', data_root)
    
    cache_dir = config['data']['cache_dir']
    if '${BGB_DATA_ROOT}' in cache_dir:
        cache_dir = cache_dir.replace('${BGB_DATA_ROOT}', data_root)
    
    # Validate paths
    if not Path(root_dir).exists():
        raise ValueError(f"Dataset root not found: {root_dir}")
    if not cache_index_path.exists():
        raise ValueError(f"Cache index not found: {cache_index_path}")
    
    # Create datasets
    logger.info(f"Loading datasets from {root_dir}")
    logger.info(f"Using cache at {cache_dir}")
    
    train_dataset = TUABCachedDataset(
        root_dir=Path(root_dir),
        split='train',
        window_duration=config['data']['window_duration'],
        window_stride=config['data']['window_stride'],
        sampling_rate=config['data']['sampling_rate'],
        preload=False,
        normalize=True,
        cache_dir=Path(cache_dir),
        cache_index_path=cache_index_path
    )
    
    val_dataset = TUABCachedDataset(
        root_dir=Path(root_dir),
        split='eval',
        window_duration=config['data']['window_duration'],
        window_stride=config['data']['window_duration'],  # No overlap for validation
        sampling_rate=config['data']['sampling_rate'],
        preload=False,
        normalize=True,
        cache_dir=Path(cache_dir),
        cache_index_path=cache_index_path
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers'],
        prefetch_factor=config['data']['prefetch_factor'],
        collate_fn=collate_eeg_batch_fixed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers'],
        prefetch_factor=config['data']['prefetch_factor'],
        collate_fn=collate_eeg_batch_fixed
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train EEGPT Linear Probe')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables
    config = resolve_env_vars(config)
    
    # Override seed if provided
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    
    # Set random seeds
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"output/{config['experiment']['name']}_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Update steps per epoch for OneCycleLR
    if config['training']['scheduler']['name'] == 'OneCycleLR':
        config['training']['scheduler']['steps_per_epoch'] = len(train_loader)
    
    # Create model
    model_checkpoint = config['model']['backbone']['checkpoint_path']
    if '${BGB_DATA_ROOT}' in model_checkpoint:
        data_root = os.environ.get('BGB_DATA_ROOT', '/default/path')
        model_checkpoint = model_checkpoint.replace('${BGB_DATA_ROOT}', data_root)
    
    if not Path(model_checkpoint).exists():
        raise ValueError(f"Model checkpoint not found: {model_checkpoint}")
    
    logger.info(f"Loading EEGPT from {model_checkpoint}")
    backbone = EEGPTWrapper(checkpoint_path=model_checkpoint)
    backbone.to(device)
    backbone.eval()  # Freeze backbone
    
    # Test forward pass
    logger.info("Testing model forward pass...")
    with torch.no_grad():
        dummy_input = torch.randn(2, 20, 1024).to(device)  # (batch, channels, time)
        dummy_output = backbone(dummy_input)
        logger.info(f"Model output shape: {dummy_output.shape}")
        
        # Update config if needed
        actual_embed_dim = dummy_output.shape[-1]
        if config['model']['probe']['input_dim'] != actual_embed_dim:
            logger.warning(
                f"Updating probe input_dim from {config['model']['probe']['input_dim']} "
                f"to {actual_embed_dim}"
            )
            config['model']['probe']['input_dim'] = actual_embed_dim
    
    # Create probe
    probe = LinearProbe(config['model'])
    probe.to(device)
    logger.info(f"Created probe: {probe}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Create scheduler
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler_config = config['training']['scheduler']
        if scheduler_config['name'] == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=float(scheduler_config['max_lr']),
                epochs=scheduler_config['epochs'],
                steps_per_epoch=scheduler_config['steps_per_epoch'],
                pct_start=scheduler_config['pct_start'],
                anneal_strategy=scheduler_config['anneal_strategy'],
                div_factor=scheduler_config['div_factor'],
                final_div_factor=scheduler_config['final_div_factor']
            )
        elif scheduler_config['name'] == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config.get('eta_min', 0)
            )
    
    # Create trainer and start training
    trainer = Trainer(config, output_dir, device)
    trainer.train(backbone, probe, train_loader, val_loader, optimizer, scheduler)


if __name__ == "__main__":
    main()