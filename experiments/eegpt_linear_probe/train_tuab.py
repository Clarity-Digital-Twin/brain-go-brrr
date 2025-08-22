#!/usr/bin/env python
"""Train EEGPT linear probe with paper-aligned settings for TUAB abnormality detection."""

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

# Import custom dataset and collate (use utils module to avoid duplication)
sys.path.insert(0, str(Path(__file__).parent))
from utils.custom_collate_fixed import collate_eeg_batch_fixed
from tuab_dataset import TUABMemoryMappedDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Two-layer linear probe with channel adapter."""

    def __init__(self, config):
        super().__init__()

        # Channel adapter (1x1 conv)
        if config["probe"].get("use_channel_adapter", False):
            self.channel_adapter = nn.Conv1d(
                config["probe"]["channel_adapter_in"],
                config["probe"]["channel_adapter_out"],
                kernel_size=1,
            )
        else:
            self.channel_adapter = None

        # Two-layer probe using LazyLinear to infer input dimension
        self.probe = nn.Sequential(
            nn.LazyLinear(config["probe"]["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(config["probe"]["dropout"]),
            nn.Linear(config["probe"]["hidden_dim"], config["probe"]["n_classes"]),
        )

    def forward(self, features):
        """Forward pass through probe."""
        # features: (batch_size, n_temporal, n_summary_tokens, embed_dim)
        # For TUAB with 4s windows: (B, 16, 4, 512)
        # Flatten all features: 16 * 4 * 512 = 32,768 features
        batch_size = features.shape[0]
        x = features.reshape(batch_size, -1)  # Flatten to (batch_size, 32768)
        return self.probe(x)


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

    # Create memory-mapped datasets (NO RAM usage - streams from disk)
    logger.info("Creating memory-mapped datasets...")
    train_dataset = TUABMemoryMappedDataset(cache_dir=cache_dir, split="train")

    # Validation dataset
    val_dataset = TUABMemoryMappedDataset(cache_dir=cache_dir, split="eval")

    # Create dataloaders - SIMPLE AND RELIABLE FOR WSL
    # Force single-threaded for WSL stability with memory-mapped data
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Always 0 for WSL stability
        pin_memory=False,  # Disable for WSL with mmap
        collate_fn=collate_eeg_batch_fixed,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=0,  # Always 0 for WSL stability
        pin_memory=False,  # Disable for WSL with mmap
        collate_fn=collate_eeg_batch_fixed,
    )

    return train_loader, val_loader


def train_epoch(model, probe, train_loader, optimizer, scheduler, device, config, epoch=0):
    """Train for one epoch."""
    model.eval()  # Backbone stays frozen
    probe.train()

    losses = []
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)

        # Forward through frozen backbone with temporal features
        with torch.no_grad():
            features = model.extract_features(data, return_all_temporal=True)
            # Verify patch count matches expected
            n_patches = features.shape[1]
            expected_patches = data.shape[-1] // 64
            assert n_patches == expected_patches, f"Patch count mismatch: got {n_patches}, expected {expected_patches} from {data.shape[-1]} samples"
            # Log shape on first batch for verification
            if batch_idx == 0 and epoch == 0:
                logger.info(f"EEGPT features shape: {features.shape} -> flattened: {features.reshape(features.size(0), -1).shape[1]} features")

        # Forward through probe
        logits = probe(features)

        # Compute loss
        if config["training"].get("weighted_loss", False):
            # Compute class weights robustly even if a batch has a single class
            n_classes = logits.size(1)
            class_counts = torch.bincount(labels, minlength=n_classes)
            class_weights = 1.0 / (class_counts.float() + 1e-5)
            class_weights = class_weights / class_weights.sum()
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, labels)

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

        # Track metrics
        losses.append(loss.item())
        preds = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Compute epoch metrics
    auroc = roc_auc_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, np.array(all_preds) > 0.5)

    return {"loss": np.mean(losses), "auroc": auroc, "bacc": bacc}


def validate(model, probe, val_loader, device):
    """Validate the model."""
    model.eval()
    probe.eval()

    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            data = data.to(device)
            labels = labels.to(device)

            # Forward with temporal features
            features = model.extract_features(data, return_all_temporal=True)
            # Verify patch count
            n_patches = features.shape[1]
            expected_patches = data.shape[-1] // 64
            assert n_patches == expected_patches, f"Val patch mismatch: {n_patches} != {expected_patches}"
            # Log shape on first validation batch
            if batch_idx == 0:
                logger.debug(f"Val features shape: {features.shape}")
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

    return {"loss": np.mean(losses), "auroc": auroc, "bacc": bacc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tuab.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory (default: auto-generated)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/{config['experiment']['name']}_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Output directory: {output_dir}")

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
        total_steps=total_steps,  # Use total_steps instead of epochs/steps_per_epoch
        pct_start=config["training"]["scheduler"]["pct_start"],
        anneal_strategy=config["training"]["scheduler"]["anneal_strategy"],
        div_factor=config["training"]["scheduler"]["div_factor"],
        final_div_factor=config["training"]["scheduler"]["final_div_factor"],
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("OneCycleLR Scheduler Configuration:")
    logger.info(
        f"  Total steps: {total_steps} ({steps_per_epoch} batches/epoch * {config['training']['max_epochs']} epochs)"
    )
    logger.info(f"  Max LR: {config['training']['scheduler']['max_lr']:.6f}")
    logger.info(
        f"  Initial LR: {config['training']['scheduler']['max_lr'] / config['training']['scheduler']['div_factor']:.6f}"
    )
    logger.info(
        f"  Final LR: {config['training']['scheduler']['max_lr'] / config['training']['scheduler']['final_div_factor']:.6f}"
    )
    logger.info(
        f"  Warmup: {config['training']['scheduler']['pct_start'] * 100:.1f}% ({int(total_steps * config['training']['scheduler']['pct_start'])} steps)"
    )
    logger.info(f"{'=' * 60}\n")

    # Prepare graceful shutdown handling and history logging
    history_path = output_dir / "history.jsonl"
    state = {
        "probe": None,
        "optimizer": None,
        "scheduler": None,
        "epoch": -1,
        "best_val_auroc": 0.0,
    }

    def save_checkpoint(tag: str = "manual") -> None:
        try:
            if state["probe"] is None:
                return
            checkpoint = {
                "epoch": state["epoch"],
                "probe_state_dict": state["probe"].state_dict(),
                "optimizer_state_dict": state["optimizer"].state_dict() if state["optimizer"] else None,
                "scheduler_state_dict": state["scheduler"].state_dict() if state["scheduler"] else None,
                "best_val_auroc": state["best_val_auroc"],
                "config": config,
                "tag": tag,
            }
            torch.save(checkpoint, output_dir / "last_model.pt")
            logger.info(f"Saved checkpoint (tag={tag}) at epoch {state['epoch']}")
        except Exception:
            logger.exception("Failed to save checkpoint on signal/exception")

    def _handle_signal(signum, _frame):
        signame = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}.get(signum, str(signum))
        logger.error(f"Received {signame}; saving checkpoint and exiting...")
        save_checkpoint(tag=f"signal_{signame}")
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Training loop
    best_val_auroc = 0
    patience_counter = 0

    for epoch in range(config["training"]["max_epochs"]):
        state["probe"] = probe
        state["optimizer"] = optimizer
        state["scheduler"] = scheduler
        state["epoch"] = epoch
        state["best_val_auroc"] = best_val_auroc
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")

        # Train
        train_metrics = train_epoch(
            backbone, probe, train_loader, optimizer, scheduler, device, config, epoch
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"AUROC: {train_metrics['auroc']:.4f}, "
            f"BACC: {train_metrics['bacc']:.4f}"
        )

        # Persist train metrics incrementally
        try:
            with open(history_path, "a", encoding="utf-8") as hf:
                json.dump({"epoch": epoch + 1, "split": "train", **train_metrics}, hf)
                hf.write("\n")
        except Exception:
            logger.exception("Failed writing train metrics to history.jsonl")

        # Validate
        if (epoch + 1) % 2 == 0:  # Validate every 2 epochs
            val_metrics = validate(backbone, probe, val_loader, device)
            logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"AUROC: {val_metrics['auroc']:.4f}, "
                f"BACC: {val_metrics['bacc']:.4f}"
            )

            # Persist val metrics incrementally
            try:
                with open(history_path, "a", encoding="utf-8") as hf:
                    json.dump({"epoch": epoch + 1, "split": "val", **val_metrics}, hf)
                    hf.write("\n")
            except Exception:
                logger.exception("Failed writing val metrics to history.jsonl")

            # Save checkpoint if best
            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                patience_counter = 0

                checkpoint = {
                    "epoch": epoch,
                    "probe_state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_auroc": val_metrics["auroc"],
                    "val_bacc": val_metrics["bacc"],
                    "config": config,
                }
                torch.save(checkpoint, output_dir / "best_model.pt")
                logger.info(f"Saved best model with AUROC: {val_metrics['auroc']:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config["training"]["early_stopping"]["patience"]:
                logger.info("Early stopping triggered")
                break

    logger.info(f"\nTraining complete! Best AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logging.exception("Fatal exception during training")
        # Best-effort: exit with non-zero to signal failure in logs
        sys.exit(1)
