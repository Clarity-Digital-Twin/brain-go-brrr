#!/usr/bin/env python
"""NaN-robust EEGPT training with comprehensive fixes and debugging."""

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
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
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


class NaNDebugCallback(Callback):
    """Debug callback to catch NaN issues early."""
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, y = batch
        if torch.isnan(x).any():
            logger.error(f"NaN in input data at batch {batch_idx}")
            # Find which samples have NaN
            nan_samples = torch.isnan(x).any(dim=(1,2))
            logger.error(f"Samples with NaN: {nan_samples.nonzero().squeeze().tolist()}")
    
    def on_after_backward(self, trainer, pl_module):
        # Check gradients
        for name, param in pl_module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}")
                

class RobustLinearProbeTrainer(LinearProbeTrainer):
    """Enhanced trainer with NaN protection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_count = 0
        self.batch_nan_counts = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NaN debugging."""
        # Check input
        if torch.isnan(x).any():
            logger.warning(f"NaN in forward input! Shape: {x.shape}")
            # Replace NaN with zeros as emergency fix
            x = torch.nan_to_num(x, nan=0.0)
        
        # Clamp values to prevent overflow
        x = torch.clamp(x, min=-100, max=100)
        
        # Forward through model
        logits = self.model(x)
        
        # Check output
        if torch.isnan(logits).any():
            logger.warning("NaN in logits!")
            logits = torch.nan_to_num(logits, nan=0.0)
            
        # Clamp logits to prevent overflow in loss
        logits = torch.clamp(logits, min=-20, max=20)
        
        return logits
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with NaN tracking."""
        x, y = batch
        
        # Data validation
        if torch.isnan(x).any():
            self.nan_count += 1
            logger.warning(f"Batch {batch_idx}: NaN in input data")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Check for extreme values
        x_std = x.std()
        if x_std > 1000 or x_std < 1e-6:
            logger.warning(f"Batch {batch_idx}: Extreme std={x_std.item():.2e}")
        
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Batch {batch_idx}: NaN/Inf loss={loss.item()}")
            # Skip this batch
            return torch.tensor(0.0, requires_grad=True)
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("nan_count", float(self.nan_count), on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Optimizer with gradient clipping."""
        config = super().configure_optimizers()
        
        # Add gradient clipping
        for opt_dict in (config if isinstance(config, list) else [config]):
            opt_dict["gradient_clip_val"] = 1.0
            opt_dict["gradient_clip_algorithm"] = "norm"
        
        return config


def create_robust_probe(checkpoint_path: Path, n_channels: int = 20) -> nn.Module:
    """Create probe with additional safeguards."""
    
    class RobustEEGPTProbe(AbnormalityDetectionProbe):
        def forward(self, x):
            # Extra normalization
            if x.std() < 1e-6:
                x = x + torch.randn_like(x) * 1e-4  # Add tiny noise
            
            # Ensure finite values
            x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
            
            return super().forward(x)
    
    return RobustEEGPTProbe(checkpoint_path, n_input_channels=n_channels)


def main():
    pl.seed_everything(42)
    
    # Load config
    cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_config.yaml")
    
    # ROBUST SETTINGS
    cfg.training.epochs = 50  # Shorter first to test
    cfg.training.batch_size = 32  # Smaller batches for stability
    cfg.training.learning_rate = 5e-4  # Conservative LR
    cfg.training.weight_decay = 0.01
    cfg.training.patience = 15
    cfg.training.accumulate_grad_batches = 2  # Effective batch = 64
    
    # Use float32 for stability
    cfg.experiment.precision = 32  # Full precision
    
    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    logger.info("=" * 60)
    logger.info("NaN-ROBUST TRAINING")
    logger.info("=" * 60)
    
    # Create datasets with extra validation
    logger.info("Creating datasets...")
    train_dataset = TUABDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=8.0,
        window_stride=8.0,  # No overlap initially for debugging
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
    
    # Validate a few samples
    logger.info("Validating dataset samples...")
    for i in range(min(10, len(train_dataset))):
        try:
            x, y = train_dataset[i]
            assert not np.isnan(x).any(), f"Sample {i} has NaN"
            assert np.isfinite(x).all(), f"Sample {i} has infinite values"
            assert x.std() > 1e-6, f"Sample {i} has zero std"
        except Exception as e:
            logger.error(f"Sample {i} validation failed: {e}")
    
    # Create balanced sampler
    logger.info("Creating balanced sampler...")
    all_labels = []
    for i in range(len(train_dataset)):
        try:
            _, label = train_dataset[i]
            all_labels.append(label)
        except:
            all_labels.append(0)  # Default to normal
    
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = 1.0 / (class_counts.float() + 1.0)
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * 2
    
    sample_weights = [class_weights[label].item() for label in all_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders with worker init
    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=2,  # Fewer workers for debugging
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    
    # Model
    logger.info("Creating model...")
    probe = create_robust_probe(checkpoint_path, n_channels=20)
    
    # Test model on a batch
    logger.info("Testing model on sample batch...")
    for batch in train_loader:
        x, y = batch
        try:
            with torch.no_grad():
                out = probe(x.cuda() if torch.cuda.is_available() else x)
            logger.info(f"Test batch OK: output shape {out.shape}")
            break
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            raise
    
    # Lightning module
    lightning_model = RobustLinearProbeTrainer(
        model=probe,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        class_weights=class_weights.cuda() if torch.cuda.is_available() else class_weights,
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"robust_{timestamp}"
    
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
        name="tuab_robust",
        version=timestamp,
    )
    
    # Trainer with safety features
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.experiment.precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping, nan_debug],
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=0.25,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        detect_anomaly=True,  # Enable anomaly detection
        deterministic=True,
    )
    
    logger.info("Starting robust training...")
    logger.info(f"Settings: BS={cfg.training.batch_size}, LR={cfg.training.learning_rate}")
    logger.info(f"Precision: {cfg.experiment.precision}, Accumulate: {cfg.training.accumulate_grad_batches}")
    
    try:
        trainer.fit(lightning_model, train_loader, val_loader)
        
        logger.info(f"Best AUROC: {checkpoint_callback.best_model_score:.4f}")
        logger.info(f"Total NaN batches: {lightning_model.nan_count}")
        logger.info(f"Model saved to: {log_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()