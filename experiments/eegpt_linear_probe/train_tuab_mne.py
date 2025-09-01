#!/usr/bin/env python3
"""
TUAB training script with MNE+Autoreject preprocessing.
Parallel implementation to train_tuab.py but using clean data.
Expected to achieve 75-87% AUROC (vs 56% without preprocessing).
"""

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from brain_go_brrr.infra.data.tuab_dataset import TUABDataset as TUABMNEDataset
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
from brain_go_brrr.infra.ml_models.linear_probe import TwoLayerProbe
from brain_go_brrr.utils import collate_tuab_batch

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
    # Create a subset that starts from the correct position (or full if start_idx=0)
    subset_indices = epoch_indices[start_idx:].tolist() if start_idx > 0 else epoch_indices.tolist()

    # Create a Subset dataset with the exact indices we want
    # This preserves order - no random shuffling!
    subset_dataset = Subset(dataset, subset_indices)

    # Create DataLoader without any sampler - just iterate in order
    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: Don't shuffle - preserve our deterministic order
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    return loader, epoch_indices


def train_epoch(
    model: nn.Module,
    probe: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: OneCycleLR,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    output_dir: Path | None = None,
    checkpoint_every: int = 500,
    best_auroc: float = 0.0,
    start_batch: int = 0,
    global_step: int = 0,
    epoch_indices: torch.Tensor | None = None,
    batches_per_epoch: int | None = None,
) -> tuple[float, float, int]:
    """Train for one epoch with deterministic resume support."""
    probe.train()

    total_loss = 0
    all_preds = []
    all_labels = []
    batches_processed = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        # Note: with SubsetRandomSampler, we don't skip - sampler handles it

        x, y = x.to(device), y.to(device)
        global_step += 1

        # Log first batch shapes for diagnostics
        if batch_idx == 0 and epoch == 0:
            logger.info(f"First batch - x.shape: {x.shape}, y.dtype: {y.dtype}, y.shape: {y.shape}")

        # Extract EEGPT features (frozen backbone)
        with torch.no_grad():
            features = model.extract_features(x, summary=False)  # (B, 4, 512)
            features = features.flatten(1)  # match original behavior

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
        batches_processed += 1
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

        # Write heartbeat for monitoring
        if output_dir and batch_idx % 50 == 0:
            batch_size = x.shape[0]  # Actual batch size from data
            heartbeat = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch,
                'batch_idx': batch_idx + start_batch,  # Report absolute batch index
                'total_batches': batches_per_epoch
                or (len(train_loader) + start_batch),  # Exact if provided
                'loss': float(loss.item()),
                'auroc': float(current_auroc) if 'current_auroc' in locals() else 0.5,
                'lr': float(scheduler.get_last_lr()[0]),
                'global_step': global_step,
                'samples_seen': global_step * batch_size,
                'gpu_memory_gb': torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0,
                'alive': True,
            }
            heartbeat_file = output_dir / 'heartbeat.json'
            with open(heartbeat_file, 'w') as f:
                json.dump(heartbeat, f, indent=2)

        # CRITICAL: Save checkpoint every N batches to avoid losing progress
        # Use absolute batch index for uniform intervals across restarts
        abs_batch_idx = batch_idx + start_batch
        if (
            output_dir
            and checkpoint_every
            and abs_batch_idx > 0
            and abs_batch_idx % checkpoint_every == 0
        ):
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx + start_batch,  # Save absolute batch index
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auroc': best_auroc,
                'train_loss': total_loss / batches_processed if batches_processed > 0 else 0,
                'global_step': global_step,
                'epoch_indices': epoch_indices,  # Save deterministic order
                'sample_offset': (batch_idx + start_batch + 1)
                * batch_size,  # Exact samples processed in epoch
            }
            checkpoint_path = (
                output_dir / f'checkpoint_epoch{epoch}_batch{batch_idx + start_batch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(
                f"Saved intra-epoch checkpoint at epoch {epoch}, batch {batch_idx + start_batch}"
            )

    # Calculate epoch metrics
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0
    epoch_auroc = (
        roc_auc_score(all_labels, all_preds)
        if len(set(all_labels)) > 1 and len(all_labels) > 0
        else 0.5
    )

    return avg_loss, epoch_auroc, global_step


def evaluate(
    model: nn.Module,
    probe: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
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
            features = features.flatten(1)

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
        default='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuab_mne_v2',
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

    # Set seeds for reproducibility
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable fully deterministic algorithms for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    logger.info(f"Random seed: {seed}")

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

    # NOTE: train_loader will be created per-epoch for deterministic sampling
    # This is necessary for proper mid-epoch resume functionality
    batches_per_epoch = (len(train_dataset) + config['data']['batch_size'] - 1) // config['data'][
        'batch_size'
    ]
    logger.info(f"Batches per epoch: {batches_per_epoch}")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),  # Use 4 workers for parallel loading
        pin_memory=config['data'].get('pin_memory', True),  # Enable GPU transfer optimization
        persistent_workers=config['data'].get('persistent_workers', True)
        if config['data'].get('num_workers', 4) > 0
        else False,
        prefetch_factor=config['data'].get('prefetch_factor', 2)
        if config['data'].get('num_workers', 4) > 0
        else None,
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

    # Create probe from src to avoid duplication
    probe = TwoLayerProbe(
        input_dim=2048,
        hidden_dim=config["model"]["probe"]["hidden_dim"],
        output_dim=1,
        dropout=config["model"]["probe"]["dropout"],
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
    )

    # Setup scheduler
    total_steps = batches_per_epoch * config['training']['max_epochs']
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

    # Resume from checkpoint if specified (handles mid-epoch checkpoints correctly)
    start_epoch = 0
    start_batch = 0
    best_auroc = 0
    global_step = 0
    epoch_indices = None  # Will store deterministic order if resuming

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)  # nosec:weights_only - Full checkpoint with optimizer state
        probe.load_state_dict(checkpoint['probe_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Proper mid-epoch resume logic
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint.get('batch_idx', 0)
        best_auroc = checkpoint.get('best_auroc', 0)
        global_step = checkpoint.get('global_step', 0)
        epoch_indices = checkpoint.get('epoch_indices', None)  # Restore epoch order if available
        sample_offset = checkpoint.get('sample_offset', None)  # Exact sample count if available

        # Check for edge case: resuming from last batch of epoch
        # The checkpoint saves the batch we just completed, so to resume:
        # - If saved batch + 1 >= batches_per_epoch, we've finished the epoch
        # - OR if we have epoch_indices and processed all samples
        epoch_complete = start_batch + 1 >= batches_per_epoch
        if epoch_indices is not None and sample_offset is not None:
            # More precise check using actual data
            epoch_complete = epoch_complete or sample_offset >= len(epoch_indices)

        if epoch_complete:
            logger.info(
                f"Checkpoint was at last batch of epoch {start_epoch}, moving to next epoch"
            )
            start_epoch = start_epoch + 1
            start_batch = 0
            epoch_indices = None  # Generate new indices for new epoch
        elif start_batch > 0:
            # Mid-epoch resume
            logger.info(
                f"Resuming from epoch {start_epoch}, batch {start_batch}/{batches_per_epoch}, best AUROC: {best_auroc:.4f}"
            )
            logger.info(f"Global step: {global_step}")
            if epoch_indices is None:
                logger.warning(
                    "No epoch_indices in checkpoint - will use new random order (may cause data duplication)"
                )
        else:
            # Completed epoch, move to next
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}, best AUROC: {best_auroc:.4f}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['max_epochs']):
        # Determine if we're resuming mid-epoch (only for first epoch after resume)
        if epoch == start_epoch and start_batch > 0:
            # Resume mid-epoch with saved indices
            # Use exact sample offset if available, otherwise derive from batch
            if sample_offset is not None:
                start_idx = sample_offset  # Most accurate - exact samples processed
            else:
                # Fallback: derive from batch (assumes full batches)
                start_idx = (start_batch + 1) * config['data']['batch_size']

            # Check if we've processed all data in this epoch
            # Use epoch_indices length if available for more accurate check
            dataset_size = len(epoch_indices) if epoch_indices is not None else len(train_dataset)
            if start_idx >= dataset_size:
                logger.info(f"All data processed in epoch {start_epoch}, moving to next epoch")
                start_epoch = start_epoch + 1
                start_batch = 0
                epoch_indices = None
                continue  # Skip to next epoch in the loop

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
                collate_fn=collate_tuab_batch,
                epoch_indices=epoch_indices,  # Use saved indices if available
            )
            resume_batch = start_batch
        else:
            # Create new deterministic loader for this epoch
            train_loader, current_epoch_indices = create_deterministic_dataloader(
                train_dataset,
                batch_size=config['data']['batch_size'],
                epoch=epoch,
                seed=seed,
                start_idx=0,  # Starting fresh epoch
                num_workers=config['data'].get('num_workers', 4),
                pin_memory=config['data'].get('pin_memory', True),
                persistent_workers=config['data'].get('persistent_workers', True),
                prefetch_factor=config['data'].get('prefetch_factor', 2),
                collate_fn=collate_tuab_batch,
            )
            resume_batch = 0  # No resume for new epochs
            start_batch = 0  # Reset for next iteration

        # Train with intra-epoch checkpointing
        train_loss, train_auroc, global_step = train_epoch(
            model,
            probe,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch,
            output_dir=output_dir,
            checkpoint_every=config.get('training', {}).get('checkpoint_every', 500),
            best_auroc=best_auroc,
            start_batch=resume_batch,
            global_step=global_step,
            epoch_indices=current_epoch_indices,
            batches_per_epoch=batches_per_epoch,
        )

        # Reset start_batch after first epoch
        start_batch = 0

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
                'epoch_indices': current_epoch_indices,  # Save for deterministic resume
                'sample_offset': len(train_dataset),  # All samples in epoch processed
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
                'epoch_indices': current_epoch_indices,  # Save for deterministic resume
                'sample_offset': len(train_dataset),  # All samples in epoch processed
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
