#!/usr/bin/env python3
"""
TUEV training script with MNE+Autoreject preprocessing.
Multi-class event detection (6 classes) targeting Table 13 performance.
Expected: 62.32% balanced accuracy, 81.87% weighted F1, 0.635 Cohen's kappa.

Now with full operational parity to TUAB script:
- Deterministic training with seeding
- Mid-epoch resume capability
- Intra-epoch checkpointing
- Heartbeat monitoring
- Class-weighted loss for imbalanced data
- Optimized DataLoader settings
"""

import argparse
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
from brain_go_brrr.infra.ml_models.linear_probe import TwoLayerProbe
from brain_go_brrr.utils import collate_tuev_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


"""
NOTE: Probe implementation moved to src/brain_go_brrr/infra/ml_models/linear_probe.py
      This script now uses TwoLayerProbe from src to avoid parallel implementations.
"""


def create_deterministic_dataloader(
    dataset,
    batch_size: int,
    epoch: int,
    seed: int,
    start_idx: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    collate_fn=None,
    epoch_indices: torch.Tensor | None = None,
) -> tuple[DataLoader, torch.Tensor]:
    """Create a DataLoader with deterministic sampling order for reproducible resume."""
    # Use provided indices or generate new ones
    if epoch_indices is None:
        # Create deterministic generator for this epoch
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        # Generate deterministic permutation for this epoch
        epoch_indices = torch.randperm(len(dataset), generator=generator)

    # If resuming mid-epoch, use subset of indices
    subset_indices = epoch_indices[start_idx:].tolist() if start_idx > 0 else epoch_indices.tolist()

    # Create a Subset dataset with the exact indices we want
    subset_dataset = Subset(dataset, subset_indices)

    # Build DataLoader kwargs conditionally to avoid prefetch_factor=None crash
    dl_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,  # CRITICAL: Don't shuffle - preserve our deterministic order
        'num_workers': num_workers if num_workers > 0 else 0,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn,
    }

    # Only add persistent_workers and prefetch_factor if we have workers
    if num_workers > 0:
        dl_kwargs['persistent_workers'] = persistent_workers
        dl_kwargs['prefetch_factor'] = prefetch_factor

    # Create DataLoader without any sampler - preserve deterministic order
    loader = DataLoader(subset_dataset, **dl_kwargs)

    return loader, epoch_indices


def update_heartbeat(output_dir: Path, epoch: int, batch_idx: int, global_step: int):
    """Update heartbeat file for crash detection."""
    heartbeat_file = output_dir / 'heartbeat.json'
    heartbeat_data = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'batch_idx': batch_idx,
        'global_step': global_step,
    }
    with open(heartbeat_file, 'w') as f:
        json.dump(heartbeat_data, f)


def train_epoch(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    output_dir=None,
    global_step=0,
    config=None,
    epoch_indices=None,
    start_batch=0,
):
    """Train for one epoch with mid-epoch checkpointing."""
    probe.train()

    total_loss = 0
    all_preds = []
    all_labels = []
    batches_processed = 0
    samples_seen = 0  # Track cumulative samples for accurate checkpointing

    # If resuming mid-epoch, calculate samples already processed
    if start_batch > 0:
        # This is approximate, but conservative (assumes full batches)
        samples_seen = start_batch * config['data']['batch_size']

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        # Skip already-processed batches when resuming mid-epoch
        if batch_idx < start_batch:
            continue

        x, y = x.to(device), y.to(device)
        global_step += 1
        samples_seen += x.shape[0]  # Track actual samples processed

        # Log first batch shapes for diagnostics
        if batch_idx == 0 and epoch == 0:
            logger.info(f"First batch - x.shape: {x.shape}, y.dtype: {y.dtype}, y.shape: {y.shape}")

        # Extract EEGPT features (frozen backbone)
        with torch.no_grad():
            features = model.extract_features(x, summary=False)  # (B, 4, 512)
            features = features.flatten(1)  # (B, 2048) - CRITICAL FIX!

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
        batches_processed += 1

        # Update progress bar
        if batch_idx % 10 == 0 and all_labels:
            current_acc = balanced_accuracy_score(all_labels, all_preds)
            pbar.set_postfix(
                {
                    'loss': f'{loss.item():.4f}',
                    'bal_acc': f'{current_acc:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                }
            )

        # Heartbeat for crash detection
        if batch_idx % 100 == 0 and output_dir:
            update_heartbeat(output_dir, epoch, batch_idx, global_step)

        # Intra-epoch checkpointing
        if output_dir and batch_idx > 0 and batch_idx % 500 == 0:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'global_step': global_step,
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch_indices': epoch_indices,
                'sample_offset': samples_seen,  # Exact cumulative samples processed
                'config': config,
            }
            checkpoint_path = output_dir / f'checkpoint_epoch{epoch}_batch{batch_idx}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved intra-epoch checkpoint at epoch {epoch}, batch {batch_idx}")

    # Calculate epoch metrics
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) if all_labels else 0
    weighted_f1 = (
        f1_score(all_labels, all_preds, average='weighted', zero_division=0) if all_labels else 0
    )
    kappa = cohen_kappa_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0

    return avg_loss, balanced_acc, weighted_f1, kappa, global_step


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
            features = features.flatten(1)

            # Forward through probe
            logits = probe(features)

            # Compute loss
            loss = criterion(logits, y)
            total_loss += loss.item()

            # Track predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics (guard against empty eval set)
    avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) if all_labels else 0
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Per-class F1 with zero_division handling
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
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
        default=os.environ.get('BGB_CACHE_DIR', 'data/cache/tuev_mne_v2'),
        help='MNE preprocessed cache directory (env: BGB_CACHE_DIR)',
    )
    parser.add_argument(
        '--resume', type=str, default=None, help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Load config and resolve environment variables
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = resolve_env_vars(config)

    # Setup determinism for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    logger.info(f"Random seed: {seed}")
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

    # Calculate batches per epoch for scheduler (use ceil for correct count)
    batches_per_epoch = math.ceil(len(train_dataset) / config['data']['batch_size'])
    logger.info(f"Batches per epoch: {batches_per_epoch}")

    # Note: train_loader will be created per epoch with deterministic sampling

    # Build eval loader with conditional kwargs to avoid prefetch_factor=None crash
    num_workers = config['data'].get('num_workers', 4)
    eval_dl_kwargs = {
        'batch_size': config['data']['batch_size'],
        'shuffle': False,
        'num_workers': num_workers if num_workers > 0 else 0,
        'pin_memory': config['data'].get('pin_memory', True),
        'collate_fn': collate_tuev_batch,  # TUEV-specific: strict 20ch enforcement
    }

    # Only add persistent_workers and prefetch_factor if we have workers
    if num_workers > 0:
        eval_dl_kwargs['persistent_workers'] = config['data'].get('persistent_workers', True)
        eval_dl_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)

    eval_loader = DataLoader(eval_dataset, **eval_dl_kwargs)

    # Load EEGPT model
    logger.info("Loading EEGPT model...")
    eegpt_checkpoint = Path(config['model']['eegpt_checkpoint'])
    if not eegpt_checkpoint.exists():
        raise FileNotFoundError(f"EEGPT checkpoint not found at {eegpt_checkpoint}")

    model = EEGPTWrapper(checkpoint_path=eegpt_checkpoint)
    model = model.to(device)
    model.eval()  # Freeze EEGPT backbone

    # Create probe from src to avoid duplication
    probe = TwoLayerProbe(
        input_dim=2048,
        hidden_dim=256,
        output_dim=6,
        dropout=config["model"]["dropout"],
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['learning_rate'],  # 5e-4 for TUEV
        weight_decay=config['training']['weight_decay'],
    )

    # Setup scheduler using the correct PyTorch API
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        epochs=config['training']['n_epochs'],
        steps_per_epoch=batches_per_epoch,
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos',
    )

    # Setup loss with class weighting for imbalanced data
    if config.get('training', {}).get('weighted_loss', True):
        # Compute class weights from training dataset efficiently
        logger.info("Computing class weights for balanced loss...")
        # Check if dataset has precomputed class counts (much faster)
        if hasattr(train_dataset, 'class_counts'):
            class_counts = np.array(list(train_dataset.class_counts.values()))
        else:
            # Fallback: iterate through dataset (slow but works)
            all_labels = []
            for _, label in train_dataset:
                all_labels.append(label)
            class_counts = np.bincount(all_labels, minlength=6)

        # Guard against divide-by-zero if any class is missing
        total_samples = class_counts.sum()
        if np.any(class_counts == 0):
            logger.warning(f"WARNING: Some classes have zero samples: {class_counts.tolist()}")
            logger.warning("Using uniform weights to avoid divide-by-zero")
            class_weights = torch.ones(6).to(device)
        else:
            # Compute inverse frequency weights
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights = torch.FloatTensor(class_weights).to(device)
            logger.info(f"Class counts: {class_counts.tolist()}")
            logger.info(f"Class weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_balanced_acc = 0
    best_kappa = 0
    global_step = 0
    epoch_indices = None
    start_batch = 0
    sample_offset = None

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)  # nosec:weights_only - Full checkpoint with optimizer state
        probe.load_state_dict(checkpoint['probe_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler state (preferred) or set position manually
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        else:
            # Fallback: set scheduler position based on global step
            global_step_resume = checkpoint.get('global_step', 0)
            scheduler.last_epoch = global_step_resume - 1

        # Proper mid-epoch resume logic
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint.get('batch_idx', 0)
        best_balanced_acc = checkpoint.get('best_balanced_acc', 0)
        best_kappa = checkpoint.get('best_kappa', 0)
        global_step = checkpoint.get('global_step', 0)
        epoch_indices = checkpoint.get('epoch_indices', None)
        sample_offset = checkpoint.get('sample_offset', None)

        # Check if we need to advance to the next epoch
        if start_batch > 0:
            # We were mid-epoch, check if we should resume or advance
            if sample_offset is not None:
                samples_processed = sample_offset
            else:
                samples_processed = (start_batch + 1) * config['data']['batch_size']

            if samples_processed >= len(train_dataset):
                logger.info(
                    f"Epoch {start_epoch} already completed, advancing to epoch {start_epoch + 1}"
                )
                start_epoch = start_epoch + 1
                start_batch = 0
                epoch_indices = None
                sample_offset = None
            else:
                logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")
        else:
            # Completed epoch, start fresh
            start_epoch = checkpoint['epoch'] + 1

        logger.info(f"Resuming: epoch {start_epoch}, best balanced acc: {best_balanced_acc:.4f}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['n_epochs']):
        # Create deterministic dataloader for this epoch
        resume_batch = 0
        if epoch == start_epoch and start_batch > 0 and epoch_indices is not None:
            # Resume mid-epoch with saved indices
            if sample_offset is not None:
                start_idx = sample_offset
            else:
                start_idx = (start_batch + 1) * config['data']['batch_size']

            train_loader, current_epoch_indices = create_deterministic_dataloader(
                train_dataset,
                batch_size=config['data']['batch_size'],
                epoch=epoch,
                seed=seed,
                start_idx=start_idx,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=config['data'].get('pin_memory', True),
                persistent_workers=config['data'].get('persistent_workers', True),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                collate_fn=collate_tuev_batch,
                epoch_indices=epoch_indices,
            )
            resume_batch = 0  # Already sliced dataloader
        else:
            # Create new deterministic loader for this epoch
            train_loader, current_epoch_indices = create_deterministic_dataloader(
                train_dataset,
                batch_size=config['data']['batch_size'],
                epoch=epoch,
                seed=seed,
                start_idx=0,
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=config['data'].get('pin_memory', True),
                persistent_workers=config['data'].get('persistent_workers', True),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                collate_fn=collate_tuev_batch,
                epoch_indices=None,
            )
            resume_batch = 0

        # Train
        train_loss, train_acc, train_f1, train_kappa, global_step = train_epoch(
            model,
            probe,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            output_dir=output_dir,
            global_step=global_step,
            config=config,
            epoch_indices=current_epoch_indices,
            start_batch=resume_batch,
        )

        # Reset for next epoch
        start_batch = 0
        sample_offset = None

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
                'global_step': global_step,
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
                'global_step': global_step,
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
