#!/usr/bin/env python3
"""
TUEV training script with MNE+Autoreject preprocessing.
Multi-class event detection (6 classes) targeting Table 13 performance.
Expected: 62.32% balanced accuracy, 81.87% weighted F1, 0.635 Cohen's kappa.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.eegpt_linear_probe.datasets.tuev_mne_dataset import TUEVMNEDataset
from experiments.eegpt_linear_probe.utils.collate_tuev import collate_tuev_batch
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TUEVLinearProbe(nn.Module):
    """Linear probe for 6-class TUEV event detection (Table 13 architecture)."""

    def __init__(self, config):
        super().__init__()

        # Channel reduction: 23 → 20 (handled in preprocessing)
        # Temporal convolution (kernel=55 for TUEV)
        self.temporal_conv = nn.Conv1d(
            in_channels=20,  # After channel reduction
            out_channels=20,
            kernel_size=config["model"]["temporal_conv"]["kernel_size"],  # 55
            padding=config["model"]["temporal_conv"]["padding"],  # 27
            groups=config["model"]["temporal_conv"]["groups"],  # 20 (depthwise)
        )

        # Classification head
        self.probe = nn.Sequential(
            nn.LazyLinear(256),  # Hidden layer
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),  # 0.5 for TUEV
            nn.Linear(256, 6),  # 6 classes
        )

    def forward(self, features):
        """Forward pass through probe.

        Args:
            features: EEGPT features (B, 4, 512) or flattened (B, 2048)

        Returns:
            Logits for 6-class classification (B, 6)
        """
        # Flatten if needed (B, 4, 512) -> (B, 2048)
        if features.dim() == 3:
            features = features.flatten(1)

        return self.probe(features)


def train_epoch(model, probe, train_loader, optimizer, scheduler, criterion, device, epoch):
    """Train for one epoch."""
    probe.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Extract EEGPT features (frozen backbone)
        with torch.no_grad():
            features = model.extract_features(x, summary=False)  # (B, 4, 512)

        # Forward through probe
        logits = probe(features)  # (B, 6)

        # Compute loss
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

        # Update progress bar
        if batch_idx % 10 == 0:
            current_acc = balanced_accuracy_score(all_labels, all_preds)
            pbar.set_postfix(
                {
                    'loss': f'{loss.item():.4f}',
                    'bal_acc': f'{current_acc:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                }
            )

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    return avg_loss, balanced_acc, weighted_f1, kappa


def evaluate(model, probe, eval_loader, criterion, device):
    """Evaluate model."""
    probe.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(eval_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            # Extract features
            features = model.extract_features(x, summary=False)

            # Forward through probe
            logits = probe(features)

            # Compute loss
            loss = criterion(logits, y)
            total_loss += loss.item()

            # Track predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(eval_loader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    class_names = ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG']
    per_class_results = dict(zip(class_names, per_class_f1, strict=False))

    return avg_loss, balanced_acc, weighted_f1, kappa, all_preds, all_labels, per_class_results


def resolve_env_vars(obj):
    """Recursively resolve environment variables in config."""
    import re

    if isinstance(obj, str):
        # Handle ${VAR} or ${VAR}/path patterns
        def replacer(match):
            env_var = match.group(1)
            return os.environ.get(env_var, match.group(0))

        return re.sub(r'\$\{([^}]+)\}', replacer, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description='Train TUEV with MNE preprocessing')
    parser.add_argument(
        '--config', type=str, default='configs/tuev.yaml', help='Path to config file'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None, help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_mne_preprocessed',
        help='MNE preprocessed cache directory',
    )
    parser.add_argument(
        '--resume', type=str, default=None, help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Load config and resolve environment variables
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = resolve_env_vars(config)

    # Setup output directory
    if args.output_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"output/tuev_mne_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup logging to file
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Starting TUEV training with MNE+Autoreject preprocessing")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Target metrics (Table 13):")
    logger.info("  - Balanced Accuracy: 62.32%")
    logger.info("  - Weighted F1: 81.87%")
    logger.info("  - Cohen's Kappa: 0.635")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading MNE-preprocessed datasets...")
    train_dataset = TUEVMNEDataset(
        root_dir=Path(config['data']['root_dir']), split='train', cache_dir=Path(args.cache_dir)
    )

    eval_dataset = TUEVMNEDataset(
        root_dir=Path(config['data']['root_dir']), split='eval', cache_dir=Path(args.cache_dir)
    )

    logger.info(f"Train dataset: {len(train_dataset)} windows")
    logger.info(f"Eval dataset: {len(eval_dataset)} windows")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False),
        collate_fn=collate_tuev_batch,  # TUEV-specific: strict 20ch enforcement
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', False),
        collate_fn=collate_tuev_batch,  # TUEV-specific: strict 20ch enforcement
    )

    # Load EEGPT model
    logger.info("Loading EEGPT model...")
    eegpt_checkpoint = Path(config['model']['eegpt_checkpoint'])
    if not eegpt_checkpoint.exists():
        raise FileNotFoundError(f"EEGPT checkpoint not found at {eegpt_checkpoint}")

    model = EEGPTWrapper(checkpoint_path=eegpt_checkpoint)
    model = model.to(device)
    model.eval()  # Freeze EEGPT backbone

    # Create probe
    probe = TUEVLinearProbe(config).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['learning_rate'],  # 5e-4 for TUEV
        weight_decay=config['training']['weight_decay'],
    )

    # Setup scheduler
    total_steps = len(train_loader) * config['training']['n_epochs']
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=total_steps,
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos',
    )

    # Setup loss (weighted for class imbalance)
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_balanced_acc = 0
    best_kappa = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        probe.load_state_dict(checkpoint['probe_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_balanced_acc = checkpoint.get('best_balanced_acc', 0)
        best_kappa = checkpoint.get('best_kappa', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best balanced acc: {best_balanced_acc:.4f}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['n_epochs']):
        # Train
        train_loss, train_acc, train_f1, train_kappa = train_epoch(
            model, probe, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Evaluate
        eval_loss, eval_acc, eval_f1, eval_kappa, _, _, per_class = evaluate(
            model, probe, eval_loader, criterion, device
        )

        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(
            f"  Train - Loss: {train_loss:.4f}, Bal Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Kappa: {train_kappa:.4f}"
        )
        logger.info(
            f"  Eval  - Loss: {eval_loss:.4f}, Bal Acc: {eval_acc:.4f}, F1: {eval_f1:.4f}, Kappa: {eval_kappa:.4f}"
        )
        logger.info(f"  Per-class F1: {per_class}")

        # Save checkpoint if best
        if eval_acc > best_balanced_acc:
            best_balanced_acc = eval_acc
            best_kappa = eval_kappa
            checkpoint = {
                'epoch': epoch,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_balanced_acc': best_balanced_acc,
                'best_kappa': best_kappa,
                'config': config,
            }
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(
                f"Saved best model - Bal Acc: {best_balanced_acc:.4f}, Kappa: {best_kappa:.4f}"
            )

        # Save regular checkpoint
        save_every = config.get('training', {}).get('save_every', 5)
        if epoch % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_balanced_acc': best_balanced_acc,
                'best_kappa': best_kappa,
                'config': config,
            }
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best Balanced Accuracy: {best_balanced_acc:.4f} (target: 0.6232)")
    logger.info(f"Best Cohen's Kappa: {best_kappa:.4f} (target: 0.6351)")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
