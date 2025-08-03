#!/usr/bin/env python
"""Fast-starting robust EEGPT training with optimized data loading."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from brain_go_brrr.data.tuab_dataset import TUABDataset
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe

sys.path.insert(0, str(Path(__file__).parent))
from train_nan_robust import RobustLinearProbeTrainer, NaNDebugCallback, create_robust_probe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    pl.seed_everything(42)
    
    # Load config
    cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_config.yaml")
    
    # FAST START SETTINGS
    cfg.training.epochs = 20  # Quick initial run
    cfg.training.batch_size = 64  # Balanced batch size
    cfg.training.learning_rate = 5e-4
    cfg.training.weight_decay = 0.01
    cfg.training.patience = 10
    
    # Use float32 for stability
    cfg.experiment.precision = 32
    
    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    logger.info("=" * 60)
    logger.info("FAST-START ROBUST TRAINING")
    logger.info("=" * 60)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TUABDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=8.0,
        window_stride=8.0,  # No overlap
        sampling_rate=256,
        preload=False,
        normalize=True,
        cache_dir=data_root / "cache/tuab_preprocessed",
    )
    
    val_dataset = TUABDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="eval",
        window_duration=8.0,
        window_stride=8.0,
        sampling_rate=256,
        preload=False,
        normalize=True,
        cache_dir=data_root / "cache/tuab_preprocessed",
    )
    
    logger.info(f"Train: {len(train_dataset)} windows")
    logger.info(f"Val: {len(val_dataset)} windows")
    
    # FAST CLASS WEIGHTS - Don't iterate through all samples
    # Use dataset statistics directly
    train_stats = train_dataset.class_counts
    n_normal = train_stats.get('normal', 1)
    n_abnormal = train_stats.get('abnormal', 1)
    
    # Calculate class weights
    total = n_normal + n_abnormal
    class_weights = torch.tensor([
        total / (2 * n_normal),
        total / (2 * n_abnormal)
    ], dtype=torch.float32)
    
    logger.info(f"Class distribution: normal={n_normal}, abnormal={n_abnormal}")
    logger.info(f"Class weights: {class_weights}")
    
    # Simple data loaders - NO WEIGHTED SAMPLING initially
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,  # Simple shuffle instead of weighted sampler
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Model
    logger.info("Creating model...")
    probe = create_robust_probe(checkpoint_path, n_channels=20)
    
    # Lightning module with class weights in loss
    lightning_model = RobustLinearProbeTrainer(
        model=probe,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        class_weights=class_weights.cuda() if torch.cuda.is_available() else class_weights,
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"fast_robust_{timestamp}"
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=log_dir / "checkpoints",
        filename="tuab-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        patience=cfg.training.patience,
        mode="max",
        verbose=True,
    )
    
    nan_debug = NaNDebugCallback()
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="tuab_fast",
        version=timestamp,
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.experiment.precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping, nan_debug],
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,
        detect_anomaly=False,  # Faster without anomaly detection
        deterministic=True,
    )
    
    logger.info("Starting fast robust training...")
    logger.info(f"Settings: BS={cfg.training.batch_size}, LR={cfg.training.learning_rate}")
    logger.info(f"Using class weights in loss: {class_weights}")
    
    trainer.fit(lightning_model, train_loader, val_loader)
    
    logger.info(f"Best AUROC: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"Total NaN batches: {lightning_model.nan_count}")
    logger.info(f"Model saved to: {log_dir}")


if __name__ == "__main__":
    main()