#!/usr/bin/env python3
"""
TUAB training script with MNE+Autoreject preprocessing.
Parallel implementation to train_tuab.py but using clean data.
Expected to achieve 75-87% AUROC (vs 56% without preprocessing).
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
from experiments.eegpt_linear_probe.datasets.tuab_mne_dataset import TUABMNEDataset
from experiments.eegpt_linear_probe.utils.collate_tuab import collate_tuab_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Linear probe for binary classification (matching EEGPT paper)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        # Two-layer probe using LazyLinear to infer input dimension
        self.probe = nn.Sequential(
            nn.LazyLinear(config["model"]["probe"]["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["model"]["probe"]["dropout"]),
            nn.Linear(config["model"]["probe"]["hidden_dim"], 1),  # Binary output
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through probe.

        Args:
            features: EEGPT features (B, 4, 512) or flattened (B, 2048)

        Returns:
            Logits for binary classification (B, 1)
        """
        # Flatten if needed (B, 4, 512) -> (B, 2048)
        if features.dim() == 3:
            features = features.flatten(1)

        return self.probe(features)


def train_epoch(
    model: nn.Module,
    probe: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple[float, float]:
    """Train for one epoch."""
    probe.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Log first batch shapes for diagnostics
        if batch_idx == 0 and epoch == 0:
            logger.info(f"First batch - x.shape: {x.shape}, y.dtype: {y.dtype}, y.shape: {y.shape}")

        # Extract EEGPT features (frozen backbone)
        with torch.no_grad():
            features = model.extract_features(x, summary=False)  # (B, 4, 512)

        # Forward through probe
        logits = probe(features).squeeze(-1)  # (B,)

        # Compute loss
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

        # Update progress bar
        if batch_idx % 10 == 0:
            current_auroc = (
                roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
            )
            pbar.set_postfix(
                {
                    'loss': f'{loss.item():.4f}',
                    'auroc': f'{current_auroc:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                }
            )

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    epoch_auroc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, epoch_auroc


def evaluate(
    model: nn.Module,
    probe: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float, list[float], list[float]]:
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
            logits = probe(features).squeeze(-1)

            # Compute loss
            loss = criterion(logits, y)
            total_loss += loss.item()

            # Track predictions
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(eval_loader)
    auroc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, auroc, all_preds, all_labels


def resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve environment variables in config.

    Handles both ${VAR} and ${VAR}/path patterns.
    """
    if isinstance(obj, str):
        # Handle ${VAR} or ${VAR}/path patterns
        def replacer(match: re.Match) -> str:
            env_var = match.group(1)
            return os.environ.get(env_var, match.group(0))

        return re.sub(r'\$\{([^}]+)\}', replacer, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description='Train TUAB with MNE preprocessing')
    parser.add_argument(
        '--config', type=str, default='configs/tuab.yaml', help='Path to config file'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None, help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuab_mne_preprocessed',
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
        args.output_dir = f"output/tuab_mne_{timestamp}"

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
    logger.info("Starting TUAB training with MNE+Autoreject preprocessing")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading MNE-preprocessed datasets...")
    train_dataset = TUABMNEDataset(
        root_dir=Path(config['data']['root_dir']), split='train', cache_dir=Path(args.cache_dir)
    )

    eval_dataset = TUABMNEDataset(
        root_dir=Path(config['data']['root_dir']), split='eval', cache_dir=Path(args.cache_dir)
    )

    logger.info(f"Train dataset: {len(train_dataset)} windows")
    logger.info(f"Eval dataset: {len(eval_dataset)} windows")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),  # Default 0 for WSL
        pin_memory=config['data'].get('pin_memory', False),  # Respect config (False for WSL)
        collate_fn=collate_tuab_batch,  # TUAB-specific: handles 19ch + workaround for 20ch
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),  # Default 0 for WSL
        pin_memory=config['data'].get('pin_memory', False),  # Respect config (False for WSL)
        collate_fn=collate_tuab_batch,  # TUAB-specific: handles 19ch + workaround for 20ch
    )

    # Load EEGPT model
    logger.info("Loading EEGPT model...")
    eegpt_checkpoint = Path(config['model']['backbone']['checkpoint_path'])
    if not eegpt_checkpoint.exists():
        raise FileNotFoundError(f"EEGPT checkpoint not found at {eegpt_checkpoint}")

    model = EEGPTWrapper(checkpoint_path=eegpt_checkpoint)
    model = model.to(device)
    model.eval()  # Freeze EEGPT backbone

    # Create probe
    probe = LinearProbe(config).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
    )

    # Setup scheduler
    total_steps = len(train_loader) * config['training']['max_epochs']
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['scheduler']['max_lr'],
        total_steps=total_steps,
        pct_start=config['training']['scheduler']['pct_start'],
        anneal_strategy='cos',
    )

    # Setup loss with class weighting if configured
    if config['training'].get('weighted_loss', False):
        # Compute class weights from training dataset
        logger.info("Computing class weights for balanced loss...")
        class_counts = {0: 0, 1: 0}
        for sample_info in train_dataset.samples:
            label = sample_info['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        # pos_weight = neg_count / pos_count for BCEWithLogitsLoss
        pos_weight = class_counts[0] / class_counts[1]
        logger.info(f"Class distribution - Normal: {class_counts[0]}, Abnormal: {class_counts[1]}")
        logger.info(f"Using pos_weight={pos_weight:.3f} for BCEWithLogitsLoss")

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_auroc = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        probe.load_state_dict(checkpoint['probe_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint.get('best_auroc', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best AUROC: {best_auroc:.4f}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['max_epochs']):
        # Train
        train_loss, train_auroc = train_epoch(
            model, probe, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Evaluate
        eval_loss, eval_auroc, _, _ = evaluate(model, probe, eval_loader, criterion, device)

        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}")
        logger.info(f"Epoch {epoch}: Eval Loss: {eval_loss:.4f}, Eval AUROC: {eval_auroc:.4f}")

        # Save checkpoint if best
        if eval_auroc > best_auroc:
            best_auroc = eval_auroc
            checkpoint = {
                'epoch': epoch,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auroc': best_auroc,
                'config': config,
            }
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model with AUROC: {best_auroc:.4f}")

        # Save regular checkpoint
        save_every = config.get('training', {}).get('save_every', 2)  # Default 2 if not in config
        if epoch % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auroc': best_auroc,
                'config': config,
            }
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
