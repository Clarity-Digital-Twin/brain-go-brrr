#!/usr/bin/env python
"""Final proper training script with all fixes from checklist."""

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
# Reduce channel warning spam
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:brain_go_brrr.data"

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


def create_stratified_dataloader(dataset, batch_size, num_workers=0, shuffle=True):
    """Create a dataloader that ensures each batch has both classes."""
    # Get all labels
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset.samples[i]["label"])
    labels = np.array(labels)

    if shuffle:
        # Create stratified batches by alternating between classes
        normal_indices = np.where(labels == 0)[0]
        abnormal_indices = np.where(labels == 1)[0]

        # Shuffle within each class
        np.random.shuffle(normal_indices)
        np.random.shuffle(abnormal_indices)

        # Interleave indices to ensure mixed batches
        min_len = min(len(normal_indices), len(abnormal_indices))
        indices = []
        for i in range(min_len):
            indices.append(normal_indices[i])
            indices.append(abnormal_indices[i])

        # Add remaining samples
        if len(normal_indices) > min_len:
            indices.extend(normal_indices[min_len:])
        if len(abnormal_indices) > min_len:
            indices.extend(abnormal_indices[min_len:])

        # Create sampler
        sampler = torch.utils.data.SubsetRandomSampler(indices)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,  # MPS doesn't support pinned memory
        )
    else:
        # For validation, just use sequential sampling
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )


def main():
    # Set seed for reproducibility
    pl.seed_everything(42)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load base config
    cfg = OmegaConf.load("configs/tuab_config.yaml")

    # Override with proper settings
    cfg.training.epochs = 10
    cfg.training.batch_size = 64  # Paper setting
    cfg.training.learning_rate = 5e-4
    cfg.training.patience = 3
    cfg.training.monitor = "val_loss"  # Avoid AUROC=nan
    cfg.training.mode = "min"
    cfg.training.save_top_k = 5

    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    model_checkpoint = data_root / cfg.model.checkpoint_path
    dataset_root = data_root / cfg.data.root_dir
    log_dir = Path(f"logs/run_{timestamp}")

    logger.info(f"Starting proper training run: {timestamp}")
    logger.info(f"Model checkpoint: {model_checkpoint}")
    logger.info(f"Dataset root: {dataset_root}")

    # Initialize model
    model = AbnormalityDetectionProbe(
        checkpoint_path=model_checkpoint,
        n_input_channels=20,  # Fixed channel count
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
    )

    # Use stratified validation loader to ensure mixed batches
    val_loader = create_stratified_dataloader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,  # Shuffle to mix classes
    )

    # Calculate steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.training.epochs

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

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1,
        precision=32,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Check 4 times per epoch
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting training with proper configuration:")
    logger.info(f"  Epochs: {cfg.training.epochs}")
    logger.info(f"  Batch size: {cfg.training.batch_size}")
    logger.info(f"  Learning rate: {cfg.training.learning_rate}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Monitor: {cfg.training.monitor}")
    logger.info("  Stratified validation: Yes")
    logger.info("=" * 60)

    trainer.fit(lightning_model, train_loader, val_loader)

    # Save final model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model: {best_model_path}")

        # Load best model and save probe weights
        checkpoint = torch.load(best_model_path, map_location="cpu")
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
            logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
