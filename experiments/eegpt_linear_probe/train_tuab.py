#!/usr/bin/env python3
"""
FIXED TUAB training script matching EEGPT paper exactly.
Uses BCEWithLogitsLoss for binary classification without class weights.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.eegpt_linear_probe.datasets.tuab_cached_dataset import TUABCachedDataset
from experiments.eegpt_linear_probe.utils.custom_collate_fixed import collate_eeg_batch_fixed
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Linear probe for binary classification (matching EEGPT)."""

    def __init__(self, config):
        super().__init__()

        # Two-layer probe using LazyLinear to infer input dimension
        self.probe = nn.Sequential(
            nn.LazyLinear(config["probe"]["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["probe"]["dropout"]),
            nn.Linear(config["probe"]["hidden_dim"], 1),  # Binary output!
        )

    def forward(self, features):
        """Forward pass through probe."""
        batch_size = features.shape[0]

        # Handle both (B, 4, 512) and (B, 512) shapes
        if features.ndim == 3:
            # (B, 4, 512) -> flatten to (B, 2048)
            x = features.reshape(batch_size, -1)
        else:
            x = features

        return self.probe(x).squeeze(-1)  # (B, 1) -> (B,) for BCEWithLogitsLoss


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variables
    def resolve_env_vars(obj):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
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
    data_root = os.environ.get(
        "BGB_DATA_ROOT", "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"
    )
    cache_dir = Path(data_root) / "cache" / "tuab_4s_final"

    # Create cached datasets that load .pt files
    logger.info("Creating cached datasets...")
    train_dataset = TUABCachedDataset(cache_dir=cache_dir, split="train")
    val_dataset = TUABCachedDataset(cache_dir=cache_dir, split="eval")

    # Create dataloaders with proper settings for WSL
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,  # Shuffle for training
        num_workers=0,  # WSL compatibility
        pin_memory=False,  # WSL compatibility
        collate_fn=collate_eeg_batch_fixed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_eeg_batch_fixed,
    )

    return train_loader, val_loader


def train_epoch(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    device,
    config,
    epoch,
    output_dir,
    start_batch,
    global_step,
    total_batches=None,
    initial_batch=0,
):
    """Train for one epoch with batch-level resume support."""
    model.eval()  # Backbone stays frozen
    probe.train()

    losses = []
    all_preds = []
    all_labels = []

    # Micro-batching configuration
    micro_batch_size = 16  # Process 16 samples at a time for feature extraction

    # Binary classification criterion matching EEGPT
    criterion = nn.BCEWithLogitsLoss()

    # Optional: Add FIXED pos_weight for class imbalance (NOT dynamic!)
    # Based on TUAB's ~80% normal, ~20% abnormal distribution
    # pos_weight = torch.tensor([4.0]).to(device)  # Weight for positive class
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use total_batches if provided (for subset loaders), else use train_loader length
    total = total_batches if total_batches is not None else len(train_loader)

    pbar = tqdm(
        enumerate(train_loader), desc="Training", total=len(train_loader), initial=initial_batch
    )
    for batch_idx, (data, labels) in pbar:
        try:
            data = data.to(device)
            labels = labels.float().to(device)  # Float for BCEWithLogitsLoss
            batch_size = data.size(0)

            # Forward through frozen backbone with temporal features using micro-batching
            with torch.no_grad():
                # Process in smaller chunks to reduce memory pressure
                features_list = []
                for i in range(0, batch_size, micro_batch_size):
                    end_idx = min(i + micro_batch_size, batch_size)
                    micro_batch = data[i:end_idx]
                    # Get all 4 summary tokens, NOT averaged (summary=False)
                    micro_features = model.extract_features(micro_batch, summary=False)
                    features_list.append(micro_features)

                # Concatenate all micro-batch features
                features = torch.cat(features_list, dim=0)

                # Log shape on first batch for verification
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"EEGPT features shape: {features.shape}")
                    logger.info(f"Expected shape: (batch_size={batch_size}, 4 tokens, 512 dims)")
                    logger.info(f"Flattened probe input: ({batch_size}, {4*512})")

            # Forward through probe
            logits = probe(features)  # (B,) for binary

            # Compute loss - BCEWithLogitsLoss as in EEGPT paper
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config["training"]["gradient_clip_val"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    probe.parameters(), config["training"]["gradient_clip_val"]
                )

            optimizer.step()
            scheduler.step()
            global_step += 1

            # Write heartbeat for signal handler (atomic write for safety)
            if output_dir:
                heartbeat = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "walltime": time.time(),
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                }
                # Atomic write: write to temp file then rename
                heartbeat_path = output_dir / "heartbeat.json"
                heartbeat_tmp = output_dir / "heartbeat.json.tmp"
                with open(heartbeat_tmp, 'w') as f:
                    json.dump(heartbeat, f)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                os.replace(heartbeat_tmp, heartbeat_path)  # Atomic rename

            # Track metrics
            losses.append(loss.item())
            preds = torch.sigmoid(logits).detach().cpu().numpy()  # Sigmoid for binary
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar (use scientific notation for tiny losses)
            pbar.set_postfix(
                {"loss": f"{loss.item():.6e}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            )

            # Periodic memory cleanup and checkpointing
            if batch_idx % 100 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
                logger.info(f"Batch {batch_idx}/{total}: loss={loss.item():.4f}, clearing cache")

            if batch_idx % 500 == 0 and batch_idx > 0 and output_dir:
                checkpoint_path = output_dir / f"checkpoint_epoch{epoch}_batch{batch_idx}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "probe_state_dict": probe.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss.item(),
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint at batch {batch_idx}")

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            # Continue training on error - don't crash the whole run
            continue

    # Calculate epoch metrics
    epoch_loss = np.mean(losses)
    if len(set(all_labels)) > 1:  # Only compute if we have both classes
        epoch_auroc = roc_auc_score(all_labels, all_preds)
    else:
        epoch_auroc = 0.5  # Default if only one class

    logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, AUROC={epoch_auroc:.4f}")

    return {"loss": epoch_loss, "auroc": epoch_auroc}, global_step


def validate(model, probe, val_loader, device):
    """Validate the model."""
    model.eval()
    probe.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validation"):
            data = data.to(device)
            labels = labels.float().to(device)

            # Extract features
            features = model.extract_features(data, summary=False)

            # Get predictions
            logits = probe(features)
            preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    if len(set(all_labels)) > 1:
        auroc = roc_auc_score(all_labels, all_preds)
    else:
        auroc = 0.5

    accuracy = np.mean((np.array(all_preds) > 0.5) == np.array(all_labels))

    return {"auroc": auroc, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/tuab.yaml", help="Path to config file"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output/tuab_fixed_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Output directory: {output_dir}")
    logger.info("Using BCEWithLogitsLoss (matching EEGPT paper)")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Update steps per epoch in config
    config["training"]["scheduler"]["steps_per_epoch"] = len(train_loader)

    # Create model
    # Resolve model checkpoint path
    model_checkpoint = config["model"]["backbone"]["checkpoint_path"]
    if "${BGB_DATA_ROOT}" in model_checkpoint:
        data_root = os.environ.get(
            "BGB_DATA_ROOT", "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"
        )
        model_checkpoint = model_checkpoint.replace("${BGB_DATA_ROOT}", data_root)

    backbone = EEGPTWrapper(checkpoint_path=model_checkpoint)
    backbone.to(device)
    backbone.eval()  # Freeze backbone

    probe = LinearProbe(config["model"])
    probe.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config["training"]["optimizer"]["lr"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
    )

    # Create scheduler with proper configuration
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config["training"]["max_epochs"]

    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["scheduler"]["max_lr"]),
        total_steps=total_steps,
        pct_start=config["training"]["scheduler"]["pct_start"],
        anneal_strategy=config["training"]["scheduler"]["anneal_strategy"],
        div_factor=config["training"]["scheduler"]["div_factor"],
        final_div_factor=config["training"]["scheduler"]["final_div_factor"],
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_val_auroc = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        probe.load_state_dict(checkpoint["probe_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["batch_idx"] + 1  # Start from next batch
        global_step = checkpoint["global_step"]
        logger.info(f"Resumed from epoch {start_epoch}, batch {start_batch}")

    # Training loop
    for epoch in range(start_epoch, config["training"]["max_epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")

        # Train
        train_metrics, global_step = train_epoch(
            backbone,
            probe,
            train_loader,
            optimizer,
            scheduler,
            device,
            config,
            epoch,
            output_dir,
            start_batch if epoch == start_epoch else 0,
            global_step,
        )

        # Validate
        val_metrics = validate(backbone, probe, val_loader, device)
        logger.info(f"Validation: AUROC={val_metrics['auroc']:.4f}, Acc={val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            torch.save(
                {
                    "epoch": epoch,
                    "probe_state_dict": probe.state_dict(),
                    "val_auroc": best_val_auroc,
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"Saved best model with AUROC={best_val_auroc:.4f}")

        # Save epoch checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "batch_idx": 0,
                "global_step": global_step,
                "probe_state_dict": probe.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_auroc": val_metrics["auroc"],
            },
            output_dir / f"checkpoint_epoch{epoch}.pt",
        )

    logger.info(f"\nTraining complete! Best AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    main()
