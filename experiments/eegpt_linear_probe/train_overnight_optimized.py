#!/usr/bin/env python
"""Overnight EEGPT training with paper-optimized settings for better AUROC."""

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

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from brain_go_brrr.data.tuab_dataset import TUABDataset
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe

sys.path.insert(0, str(Path(__file__).parent))
from train_tuab_probe import LinearProbeTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    pl.seed_everything(42)
    
    # Load config
    cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_config.yaml")
    
    # OPTIMIZED SETTINGS based on paper analysis
    cfg.training.epochs = 100  # More epochs for convergence
    cfg.training.batch_size = 64  # Smaller batch for better gradients
    cfg.training.learning_rate = 1e-3  # Higher LR based on paper
    cfg.training.weight_decay = 0.01  # Less regularization
    cfg.training.patience = 20  # More patience
    cfg.training.gradient_clip_val = 0.5  # Gradient clipping
    
    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    logger.info("=" * 60)
    logger.info("OVERNIGHT OPTIMIZED TRAINING")
    logger.info(f"Target AUROC: ≥0.87 (paper baseline)")
    logger.info("=" * 60)
    
    # Create datasets
    train_dataset = TUABDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=8.0,
        window_stride=4.0,  # 50% overlap for more samples
        sampling_rate=256,
        preload=False,
        normalize=True,
        cache_dir=data_root / "cache/tuab_preprocessed",
    )
    
    val_dataset = TUABDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="eval",
        window_duration=8.0,
        window_stride=8.0,  # No overlap for validation
        sampling_rate=256,
        preload=False,
        normalize=True,
        cache_dir=data_root / "cache/tuab_preprocessed",
    )
    
    logger.info(f"Train: {len(train_dataset)} windows")
    logger.info(f"Val: {len(val_dataset)} windows")
    
    # Create balanced sampler
    all_labels = []
    for i in range(len(train_dataset)):
        try:
            _, label = train_dataset[i]
            all_labels.append(label)
        except:
            all_labels.append(0)
    
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = [class_weights[label].item() for label in all_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
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
    probe = AbnormalityDetectionProbe(checkpoint_path, n_input_channels=20)
    
    # Use class weights in loss
    criterion_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")
    
    lightning_model = LinearProbeTrainer(
        model=probe,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        class_weights=criterion_weights,
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"overnight_{timestamp}"
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=log_dir / "checkpoints",
        filename="tuab-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=5,
        mode="max",
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        patience=cfg.training.patience,
        mode="max",
        verbose=True,
    )
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="tuab_overnight",
        version=timestamp,
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=1,
        precision=16,  # Mixed precision for speed
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Check 4x per epoch
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=True,
    )
    
    logger.info("Starting overnight training...")
    logger.info(f"Settings: BS={cfg.training.batch_size}, LR={cfg.training.learning_rate}")
    logger.info(f"Expected duration: 8-10 hours")
    
    trainer.fit(lightning_model, train_loader, val_loader)
    
    logger.info(f"Best AUROC: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"Model saved to: {log_dir}")


if __name__ == "__main__":
    main()