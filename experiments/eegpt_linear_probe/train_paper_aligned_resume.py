"""Training script for paper-aligned EEGPT linear probe - WITH RESUME SUPPORT!"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import argparse
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
from src.brain_go_brrr.models.eegpt_model import EEGPTModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WSL SAFETY: Set multiprocessing strategy
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy("file_system")


class LinearProbe(nn.Module):
    """Linear probe matching the original training structure."""
    
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=2, dropout=0.1, probe_type='linear'):
        super().__init__()
        
        if probe_type == 'linear':
            # Single layer probe
            self.probe = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, n_classes)
            )
        else:
            # Two-layer probe - THIS IS WHAT WAS USED!
            self.probe = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )
        
    def forward(self, x):
        # Average pool if needed (depends on input shape)
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Average over summary tokens
        return self.probe(x)


def collate_eeg_batch_fixed(batch):
    """Collate function that handles variable-length sequences."""
    # Stack data and labels
    data_list = []
    label_list = []
    
    for data, label in batch:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(label, (int, np.integer)):
            label = torch.tensor(label, dtype=torch.long)
        elif isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
            
        data_list.append(data)
        label_list.append(label)
    
    # Stack into batches
    data_batch = torch.stack(data_list)
    label_batch = torch.stack(label_list) if label_list[0].dim() > 0 else torch.tensor(label_list)
    
    return data_batch, label_batch


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    from tuab_mmap_dataset_safe import TUABMemoryMappedDatasetSafe
    
    # Expand environment variables
    cache_dir = Path(os.path.expandvars(config['data']['cache_dir']))
    
    # Create datasets
    train_dataset = TUABMemoryMappedDatasetSafe(
        cache_dir=cache_dir,
        split='train'
    )
    
    # Validation dataset
    val_dataset = TUABMemoryMappedDatasetSafe(
        cache_dir=cache_dir,
        split='eval'
    )
    
    # Create dataloaders - WSL SAFE with num_workers=0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,  # CRITICAL FOR WSL!
        pin_memory=False,  # WSL stability
        persistent_workers=False,
        collate_fn=collate_eeg_batch_fixed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,  # CRITICAL FOR WSL!
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_eeg_batch_fixed
    )
    
    return train_loader, val_loader


def train_epoch(model, probe, train_loader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    # model is EEGPTModel - doesn't need .eval(), already frozen
    probe.train()
    
    losses = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        # Extract features with EEGPT
        with torch.no_grad():
            features = model.extract_features_batch(data)
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).to(device)
        
        # Forward through probe
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), config['training']['gradient_clip_val'])
        optimizer.step()
        
        # Store for metrics
        losses.append(loss.item())
        preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        # Log periodically
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    # Step scheduler
    scheduler.step()
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return {
        'loss': np.mean(losses),
        'auroc': auroc,
        'bacc': bacc
    }


def validate(model, probe, val_loader, device):
    """Validate the model."""
    # model is EEGPTModel - doesn't need .eval()
    probe.eval()
    
    losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            
            # Extract features
            features = model.extract_features_batch(data)
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).to(device)
            
            # Forward through probe
            logits = probe(features)
            loss = F.cross_entropy(logits, labels)
            
            # Store for metrics
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
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/tuab_4s_wsl_safe.yaml')
    parser.add_argument('--resume', type=str, default='output/tuab_4s_paper_target_20250806_132743/best_model.pt',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Set environment
    os.environ['BGB_DATA_ROOT'] = os.environ.get('BGB_DATA_ROOT', 
        '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data')
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/{config['experiment']['name']}_resume_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize model
    logger.info("Loading EEGPT backbone...")
    checkpoint_path = Path(os.path.expandvars(config['model']['backbone']['checkpoint_path']))
    model = EEGPTModel(checkpoint_path=checkpoint_path, device=device)
    # EEGPTModel doesn't have .eval() - it's already frozen
    
    # Initialize probe - matching the saved model structure
    probe = LinearProbe(
        input_dim=config['model']['probe']['input_dim'],
        hidden_dim=config['model']['probe']['hidden_dim'],
        n_classes=config['model']['probe']['n_classes'],
        dropout=config['model']['probe']['dropout'],
        probe_type='two_layer'  # The saved model used two-layer probe!
    )
    probe = probe.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['scheduler']['max_lr'],
        total_steps=len(train_loader) * config['training']['max_epochs'],
        pct_start=config['training']['scheduler']['pct_start'],
        anneal_strategy=config['training']['scheduler']['anneal_strategy'],
        div_factor=config['training']['scheduler']['div_factor'],
        final_div_factor=config['training']['scheduler']['final_div_factor']
    )
    
    # 🚀 RESUME LOGIC - THE MAGIC HAPPENS HERE!
    start_epoch = 0
    best_auroc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        logger.info(f"🔥 RESUMING FROM CHECKPOINT: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load probe weights
        probe.load_state_dict(checkpoint['probe_state_dict'])
        logger.info("✅ Loaded probe weights")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Loaded optimizer state")
        
        # Load scheduler state (might need to adjust)
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✅ Loaded scheduler state")
            except:
                logger.warning("⚠️ Could not load scheduler state, starting fresh")
        
        # Set starting epoch (checkpoint has epoch 15, so we start at 16)
        start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint.get('val_auroc', 0.0)
        
        logger.info(f"🎯 RESUMING FROM EPOCH {start_epoch}")
        logger.info(f"📊 Best AUROC so far: {best_auroc:.4f}")
        logger.info(f"🎯 Target AUROC: {config['target_metrics']['auroc']}")
        logger.info(f"📈 Progress: {best_auroc/config['target_metrics']['auroc']*100:.1f}% of target")
    
    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 STARTING TRAINING FROM EPOCH {start_epoch}")
    logger.info(f"{'='*60}\n")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['max_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['max_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, probe, train_loader, optimizer, scheduler, device, config)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"AUROC: {train_metrics['auroc']:.4f}, "
                   f"BACC: {train_metrics['bacc']:.4f}")
        
        # Validate every epoch
        if (epoch + 1) % config['training']['val_check_interval'] == 0:
            val_metrics = validate(model, probe, val_loader, device)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"AUROC: {val_metrics['auroc']:.4f}, "
                       f"BACC: {val_metrics['bacc']:.4f}")
            
            # Save best model
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
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
                logger.info(f"🎉 Saved best model with AUROC: {best_auroc:.4f}")
                logger.info(f"📈 Progress: {best_auroc/config['target_metrics']['auroc']*100:.1f}% of target")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Check if we hit target
            if best_auroc >= config['target_metrics']['auroc']:
                logger.info(f"🎊 TARGET REACHED! AUROC: {best_auroc:.4f} >= {config['target_metrics']['auroc']}")
                break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 TRAINING COMPLETE!")
    logger.info(f"Best Val AUROC: {best_auroc:.4f}")
    logger.info(f"Target AUROC: {config['target_metrics']['auroc']}")
    logger.info(f"Final Progress: {best_auroc/config['target_metrics']['auroc']*100:.1f}%")
    logger.info(f"Model saved at: {output_dir / 'best_model.pt'}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()