#!/usr/bin/env python
"""GPU-optimized training script for PC with RTX 4090."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:brain_go_brrr.data"
# GPU optimizations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async GPU operations

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: E402

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from train_tuab_probe import LinearProbeTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Set seed for reproducibility
    pl.seed_everything(42)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load base config
    cfg = OmegaConf.load("configs/tuab_config.yaml")

    # GPU-optimized settings
    cfg.training.epochs = 10
    cfg.training.batch_size = 128  # Increased for GPU
    cfg.training.learning_rate = 5e-4
    cfg.training.patience = 5
    cfg.training.monitor = "val_loss"
    cfg.training.mode = "min"
    cfg.training.save_top_k = 5
    cfg.training.num_workers = 8  # Increased for faster data loading

    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    model_checkpoint = data_root / cfg.model.checkpoint_path
    dataset_root = data_root / cfg.data.root_dir
    log_dir = Path(f"logs/gpu_run_{timestamp}")

    logger.info("=" * 60)
    logger.info("GPU-OPTIMIZED TRAINING ON RTX 4090")
    logger.info("=" * 60)
    logger.info(f"Starting GPU training run: {timestamp}")
    logger.info(f"Model checkpoint: {model_checkpoint}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info(f"Workers: {cfg.training.num_workers}")
    
    # Verify GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.error("No GPU detected! This script is optimized for GPU training.")
        sys.exit(1)

    # Initialize model
    model = AbnormalityDetectionProbe(
        checkpoint_path=model_checkpoint,
        n_input_channels=20,
    )

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = TUABDataset(
        root_dir=dataset_root,
        split="train",
        sampling_rate=cfg.data.sampling_rate,
        window_duration=cfg.data.window_duration,
        window_stride=cfg.data.window_stride,
        normalize=cfg.data.normalize,
    )

    val_dataset = TUABDataset(
        root_dir=dataset_root,
        split="eval",
        sampling_rate=cfg.data.sampling_rate,
        window_duration=cfg.data.window_duration,
        window_stride=cfg.data.window_stride,
        normalize=cfg.data.normalize,
    )

    logger.info(f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")
    logger.info(f"Train distribution: {train_dataset.class_counts}")
    logger.info(f"Val distribution: {val_dataset.class_counts}")

    # Create weighted sampler for training
    class_weights = train_dataset.get_class_weights()
    sample_weights = torch.zeros(len(train_dataset))
    for idx in range(len(train_dataset)):
        label = train_dataset.samples[idx]["label"]
        sample_weights[idx] = class_weights[label]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create dataloaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=True,  # GPU optimization
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Calculate steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.training.epochs

    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Expected time per epoch: ~{steps_per_epoch / 60:.1f} minutes at 1 it/s")

    # Initialize Lightning module
    lightning_model = LinearProbeTrainer(
        model=model,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        pct_start=cfg.training.pct_start,
        div_factor=cfg.training.div_factor,
        final_div_factor=cfg.training.final_div_factor,
        class_weights=class_weights,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir / "checkpoints",
        filename="tuab_probe-{epoch:02d}-{val_loss:.4f}-{val_auroc:.4f}",
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=cfg.training.save_top_k,
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode=cfg.training.mode,
        verbose=True,
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="tuab_linear_probe",
        version=timestamp,
    )

    # Trainer with GPU optimizations
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",  # Explicitly use GPU
        devices=1,
        precision=16,  # Mixed precision for faster training
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=0.25,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=1,  # Can increase if memory allows
        benchmark=True,  # Enable cudNN autotuner
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting GPU-accelerated training:")
    logger.info(f"  Epochs: {cfg.training.epochs}")
    logger.info(f"  Batch size: {cfg.training.batch_size}")
    logger.info(f"  Learning rate: {cfg.training.learning_rate}")
    logger.info(f"  Mixed precision: 16-bit")
    logger.info(f"  Workers: {cfg.training.num_workers}")
    logger.info("  Expected speed: 1-3 it/s on RTX 4090")
    logger.info("=" * 60)

    trainer.fit(lightning_model, train_loader, val_loader)

    # Save final model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model: {best_model_path}")

        # Load best model and save probe weights
        checkpoint = torch.load(best_model_path, map_location="cuda")
        lightning_model.load_state_dict(checkpoint["state_dict"])

        probe_path = log_dir / "tuab_probe_best.pth"
        lightning_model.model.save_probe_weights(probe_path)
        logger.info(f"Saved probe weights to {probe_path}")

    logger.info("Training complete!")
    logger.info(f"Logs saved to: {log_dir}")

    # Print final metrics
    if hasattr(trainer, "logged_metrics"):
        logger.info("Final metrics:")
        for key, value in trainer.logged_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()