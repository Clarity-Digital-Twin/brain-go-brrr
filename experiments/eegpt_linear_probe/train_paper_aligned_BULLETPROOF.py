"""BULLETPROOF Training script - All issues fixed after deep audit.

COMPLETE FIX LIST:
1. ✅ OneCycleLR steps per batch (not per epoch)
2. ✅ Gradient accumulation aware total_steps  
3. ✅ Global step tracking for proper resume
4. ✅ No start_epoch reset bug
5. ✅ Optimizer LR logging (not scheduler.get_last_lr)
6. ✅ cycle_momentum=False for AdamW (CRITICAL!)
7. ✅ RNG state saving/loading for perfect reproducibility
8. ✅ Extensive sanity checks for LR progression
9. ✅ Proper checkpoint saving with all states
10. ✅ Non-blocking data transfers for speed
"""

import os
import math
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
import random

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
    """Linear probe matching the paper structure."""
    
    def __init__(self, input_dim=512, hidden_dim=128, n_classes=2, dropout=0.1, probe_type='two_layer'):
        super().__init__()
        
        if probe_type == 'linear':
            # Single layer probe
            self.probe = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, n_classes)
            )
        else:
            # Two-layer probe (PAPER DEFAULT)
            self.probe = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )
        
    def forward(self, x):
        # Average pool if needed
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Average over summary tokens
        return self.probe(x)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_eeg_batch_fixed(batch):
    """Collate function that handles variable-length sequences."""
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
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_eeg_batch_fixed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,  # CRITICAL FOR WSL!
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_eeg_batch_fixed
    )
    
    return train_loader, val_loader


def train_epoch(model, probe, train_loader, optimizer, scheduler, device, config, global_step=0):
    """Train for one epoch with all fixes applied."""
    # model is EEGPTModel - doesn't need .eval(), already frozen
    probe.train()
    
    losses = []
    all_preds = []
    all_labels = []
    
    # Get gradient accumulation steps
    accum_steps = int(config['training'].get('gradient_accumulation_steps', 1))
    
    # Expected LR at different phases for sanity checks
    max_lr = float(config['training']['scheduler']['max_lr'])
    div_factor = float(config['training']['scheduler']['div_factor'])
    final_div_factor = float(config['training']['scheduler']['final_div_factor'])
    initial_lr = max_lr / div_factor
    final_lr = max_lr / final_div_factor
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, labels) in enumerate(pbar):
        # Non-blocking transfers for speed
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Extract features with EEGPT
        with torch.no_grad():
            features = model.extract_features_batch(data)
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).to(device)
        
        # Forward through probe
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / accum_steps
        
        # Backward
        loss.backward()
        
        # Only step optimizer when accumulation is complete
        should_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
        
        if should_step:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                probe.parameters(), 
                float(config['training']['gradient_clip_val'])
            )
            
            # Check for gradient explosion
            if grad_norm > 100:
                logger.warning(f"Large gradient norm: {grad_norm:.2f} at step {global_step}")
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # CRITICAL: Step scheduler ONLY when optimizer steps
            scheduler.step()
            global_step += 1
            
            # Get ACTUAL learning rate from optimizer (not scheduler)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Extensive sanity checks
            if global_step == 1:
                expected = initial_lr
                if abs(current_lr - expected) > expected * 0.1:
                    logger.warning(f"Initial LR mismatch: {current_lr:.6f} vs expected {expected:.6f}")
            
            if global_step == 100:
                # Should be in warmup, LR increasing
                if current_lr <= initial_lr:
                    logger.warning(f"⚠️ Warmup not working! LR should be > {initial_lr:.6f}, got {current_lr:.6f}")
            
            if global_step > 10 and global_step % 100 == 0:
                if not hasattr(train_epoch, 'last_lr'):
                    train_epoch.last_lr = current_lr
                elif abs(current_lr - train_epoch.last_lr) < 1e-10:
                    logger.warning(f"⚠️ LR not changing! Stuck at {current_lr:.2e}")
                train_epoch.last_lr = current_lr
        
        # Store for metrics (unscaled loss)
        losses.append((loss * accum_steps).item())
        preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar with OPTIMIZER LR
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{(loss * accum_steps).item():.4f}',
            'lr': f'{current_lr:.2e}',
            'step': global_step
        })
        
        # Enhanced logging
        if global_step > 0 and global_step % config['logging']['log_every_n_steps'] == 0:
            logger.info(f"Step {global_step} - Loss: {(loss * accum_steps).item():.4f} - LR: {current_lr:.2e} - Grad norm: {grad_norm:.2f}")
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return {
        'loss': np.mean(losses),
        'auroc': auroc,
        'bacc': bacc,
        'global_step': global_step
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
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
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
    parser.add_argument('--config', type=str, default='configs/tuab_4s_paper_aligned.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dry_run', action='store_true', help='Test scheduler for 100 steps')
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
    
    # Set seed for reproducibility
    seed = config['experiment']['seed']
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/{config['experiment']['name']}_BULLETPROOF_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize model
    logger.info("Loading EEGPT backbone...")
    checkpoint_path = Path(os.path.expandvars(config['model']['backbone']['checkpoint_path']))
    model = EEGPTModel(checkpoint_path=checkpoint_path, device=device)
    # EEGPTModel doesn't have .eval() - it's already frozen
    
    # Initialize probe
    probe = LinearProbe(
        input_dim=config['model']['probe']['input_dim'],
        hidden_dim=config['model']['probe']['hidden_dim'],
        n_classes=config['model']['probe']['n_classes'],
        dropout=config['model']['probe']['dropout'],
        probe_type='two_layer'  # Paper uses two-layer
    )
    probe = probe.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config['training']['optimizer']['lr']),
        weight_decay=float(config['training']['optimizer']['weight_decay']),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # CRITICAL: Calculate total steps with gradient accumulation awareness
    accum_steps = int(config['training'].get('gradient_accumulation_steps', 1))
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * config['training']['max_epochs']
    
    # Sanity check
    assert total_steps > 0, f"Invalid total_steps: {total_steps}"
    assert steps_per_epoch > 0, f"Invalid steps_per_epoch: {steps_per_epoch}"
    
    # Initialize tracking variables BEFORE resume logic
    start_epoch = 0
    global_step = 0
    best_auroc = 0.0
    
    # Resume logic (BEFORE scheduler creation)
    if args.resume and os.path.exists(args.resume):
        logger.info(f"🔥 RESUMING FROM CHECKPOINT: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load states
        probe.load_state_dict(checkpoint['probe_state_dict'])
        logger.info("✅ Loaded probe weights")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Loaded optimizer state")
        
        # Load tracking variables (NO RESET!)
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
        best_auroc = checkpoint.get('val_auroc', 0.0)
        
        # Restore RNG states for perfect reproducibility
        if 'rng_states' in checkpoint:
            torch.set_rng_state(checkpoint['rng_states']['torch'])
            torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])
            np.random.set_state(checkpoint['rng_states']['numpy'])
            random.setstate(checkpoint['rng_states']['python'])
            logger.info("✅ Restored RNG states")
        
        logger.info(f"🎯 RESUMING FROM EPOCH {start_epoch}, STEP {global_step}")
        logger.info(f"📊 Best AUROC so far: {best_auroc:.4f}")
    
    # Create scheduler with ALL fixes
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config['training']['scheduler']['max_lr']),
        total_steps=int(total_steps),
        pct_start=float(config['training']['scheduler']['pct_start']),
        anneal_strategy=str(config['training']['scheduler']['anneal_strategy']),
        div_factor=float(config['training']['scheduler']['div_factor']),
        final_div_factor=float(config['training']['scheduler']['final_div_factor']),
        cycle_momentum=False,  # CRITICAL FOR ADAMW!
        last_epoch=global_step - 1 if global_step > 0 else -1
    )
    
    # Load scheduler state if resuming
    if args.resume and os.path.exists(args.resume) and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✅ Loaded scheduler state")
        except Exception as e:
            logger.warning(f"⚠️ Could not load scheduler state: {e}")
    
    # Log configuration
    logger.info(f"\n{'='*60}")
    logger.info(f"OneCycleLR Scheduler Configuration (BULLETPROOF):")
    logger.info(f"  Total steps: {total_steps} ({steps_per_epoch} optimizer steps/epoch * {config['training']['max_epochs']} epochs)")
    logger.info(f"  Gradient accumulation: {accum_steps} steps")
    logger.info(f"  Max LR: {config['training']['scheduler']['max_lr']:.6f}")
    logger.info(f"  Initial LR: {config['training']['scheduler']['max_lr']/config['training']['scheduler']['div_factor']:.6f}")
    logger.info(f"  Final LR: {config['training']['scheduler']['max_lr']/config['training']['scheduler']['final_div_factor']:.6f}")
    logger.info(f"  Warmup: {config['training']['scheduler']['pct_start']*100:.1f}% ({int(total_steps*config['training']['scheduler']['pct_start'])} steps)")
    logger.info(f"  cycle_momentum: False (AdamW compatibility)")
    if global_step > 0:
        logger.info(f"  Progress: {global_step}/{total_steps} ({global_step/total_steps*100:.1f}% complete)")
    logger.info(f"{'='*60}\n")
    
    # Dry run test
    if args.dry_run:
        logger.info("🧪 DRY RUN: Testing scheduler for 100 steps...")
        dummy_loss = torch.tensor(0.0, requires_grad=True)
        for step in range(100):
            dummy_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            if step % 10 == 0:
                logger.info(f"  Step {step}: LR = {lr:.6f}")
        logger.info("✅ Dry run complete. Check if LR is changing properly.")
        return
    
    # Training loop
    logger.info(f"🚀 STARTING TRAINING FROM EPOCH {start_epoch}")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['max_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['max_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, probe, train_loader, optimizer, scheduler, device, config, global_step)
        global_step = train_metrics['global_step']  # Update global step
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"AUROC: {train_metrics['auroc']:.4f}, "
                   f"BACC: {train_metrics['bacc']:.4f}, "
                   f"Global Step: {global_step}/{total_steps}")
        
        # Sanity check
        if global_step > total_steps:
            logger.warning(f"⚠️ Global step {global_step} exceeded total_steps {total_steps}")
        
        # Validate
        if (epoch + 1) % config['training']['val_check_interval'] == 0:
            val_metrics = validate(model, probe, val_loader, device)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"AUROC: {val_metrics['auroc']:.4f}, "
                       f"BACC: {val_metrics['bacc']:.4f}")
            
            # Save checkpoint with ALL states
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auroc': val_metrics['auroc'],
                'val_bacc': val_metrics['bacc'],
                'config': config,
                'rng_states': {
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all(),
                    'numpy': np.random.get_state(),
                    'python': random.getstate()
                }
            }
            
            # Save best model
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                torch.save(checkpoint, output_dir / 'best_model.pt')
                logger.info(f"🎉 Saved best model with AUROC: {best_auroc:.4f}")
                logger.info(f"📈 Progress: {best_auroc/config['target_metrics']['auroc']*100:.1f}% of target")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Always save latest checkpoint for resume
            torch.save(checkpoint, output_dir / 'latest_checkpoint.pt')
            
            # Early stopping
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # Check if we hit target
            if best_auroc >= config['target_metrics']['auroc']:
                logger.info(f"🎊 TARGET REACHED! AUROC: {best_auroc:.4f} >= {config['target_metrics']['auroc']}")
                break
    
    # Final sanity check
    final_lr = optimizer.param_groups[0]['lr']
    expected_final = config['training']['scheduler']['max_lr'] / config['training']['scheduler']['final_div_factor']
    if abs(final_lr - expected_final) > expected_final * 0.5:
        logger.warning(f"Final LR mismatch: {final_lr:.6f} vs expected {expected_final:.6f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🏁 TRAINING COMPLETE!")
    logger.info(f"Best Val AUROC: {best_auroc:.4f}")
    logger.info(f"Target AUROC: {config['target_metrics']['auroc']}")
    logger.info(f"Final Progress: {best_auroc/config['target_metrics']['auroc']*100:.1f}%")
    logger.info(f"Model saved at: {output_dir / 'best_model.pt'}")
    logger.info(f"Final LR: {final_lr:.6f} (expected ~{expected_final:.6f})")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()