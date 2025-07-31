#!/usr/bin/env python
"""Train from preprocessed cache files directly - GPU optimized."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402

from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: E402

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from train_tuab_probe import LinearProbeTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachedDataset(Dataset):
    """Dataset that loads from preprocessed cache files."""
    
    def __init__(self, cache_files, label_map):
        self.cache_files = cache_files
        self.label_map = label_map
        
    def __len__(self):
        return len(self.cache_files)
    
    def __getitem__(self, idx):
        # Load preprocessed data from cache
        with open(self.cache_files[idx], 'rb') as f:
            data = pickle.load(f)
        
        # Extract label from filename
        filename = Path(self.cache_files[idx]).name
        if 'abnormal' in str(self.cache_files[idx]):
            label = self.label_map['abnormal']
        else:
            label = self.label_map['normal']
        
        # Return tensor and label
        return torch.from_numpy(data['data']).float(), label


def main():
    # Set seed
    pl.seed_everything(42)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # GPU check
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.error("No GPU detected!")
        sys.exit(1)
    
    # Load cache files
    cache_dir = Path(os.environ["BGB_DATA_ROOT"]) / "cache" / "tuab_preprocessed"
    cache_files = list(cache_dir.glob("*.pkl"))
    logger.info(f"Found {len(cache_files)} cached windows")
    
    # Separate by class
    normal_files = [f for f in cache_files if 'normal' in str(f.parent) or not ('abnormal' in str(f))]
    abnormal_files = [f for f in cache_files if 'abnormal' in str(f)]
    
    # If we can't distinguish, use random split
    if len(normal_files) == 0 and len(abnormal_files) == 0:
        logger.warning("Cannot distinguish classes from filenames, using 75/25 split")
        all_files = list(cache_files)
        np.random.shuffle(all_files)
        split_idx = int(0.75 * len(all_files))
        normal_files = all_files[:split_idx]
        abnormal_files = all_files[split_idx:]
    
    logger.info(f"Normal: {len(normal_files)}, Abnormal: {len(abnormal_files)}")
    
    # Create label mapping
    label_map = {'normal': 0, 'abnormal': 1}
    
    # Split into train/val
    all_files = normal_files + abnormal_files
    labels = [0] * len(normal_files) + [1] * len(abnormal_files)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")
    logger.info(f"Train distribution: Normal={train_labels.count(0)}, Abnormal={train_labels.count(1)}")
    logger.info(f"Val distribution: Normal={val_labels.count(0)}, Abnormal={val_labels.count(1)}")
    
    # Create datasets
    train_dataset = CachedDataset(train_files, label_map)
    val_dataset = CachedDataset(val_files, label_map)
    
    # Calculate class weights
    total_train = len(train_labels)
    n_normal = train_labels.count(0)
    n_abnormal = train_labels.count(1)
    
    weight_normal = total_train / (2 * n_normal) if n_normal > 0 else 1.0
    weight_abnormal = total_train / (2 * n_abnormal) if n_abnormal > 0 else 1.0
    class_weights = torch.tensor([weight_normal, weight_abnormal])
    
    # Create sampler
    sample_weights = torch.zeros(len(train_dataset))
    for idx, label in enumerate(train_labels):
        sample_weights[idx] = class_weights[label]
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    # Model
    model_checkpoint = Path(os.environ["BGB_DATA_ROOT"]) / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    model = AbnormalityDetectionProbe(
        checkpoint_path=model_checkpoint,
        n_input_channels=20,
    )
    
    # Lightning module
    lightning_model = LinearProbeTrainer(
        model=model,
        learning_rate=5e-4,
        weight_decay=1e-4,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
        class_weights=class_weights,
    )
    
    # Log directory
    log_dir = Path(f"logs/cache_gpu_run_{timestamp}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir / "checkpoints",
        filename="tuab_probe-{epoch:02d}-{val_loss:.4f}-{val_auroc:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        verbose=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
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
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        precision=16,  # Mixed precision
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=log_dir,
        log_every_n_steps=50,
        val_check_interval=0.25,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        benchmark=True,  # CuDNN autotuner
    )
    
    # Train info
    steps_per_epoch = len(train_loader)
    logger.info("=" * 60)
    logger.info("GPU TRAINING FROM CACHE")
    logger.info("=" * 60)
    logger.info(f"Epochs: 10")
    logger.info(f"Batch size: 128")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Workers: 8")
    logger.info(f"Mixed precision: 16-bit")
    logger.info(f"Expected speed: 1-3 it/s on RTX 4090")
    logger.info("=" * 60)
    
    # Train
    trainer.fit(lightning_model, train_loader, val_loader)
    
    # Save best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model: {best_model_path}")
        
        checkpoint = torch.load(best_model_path, map_location="cuda")
        lightning_model.load_state_dict(checkpoint["state_dict"])
        
        probe_path = log_dir / "tuab_probe_best.pth"
        lightning_model.model.save_probe_weights(probe_path)
        logger.info(f"Saved probe weights to {probe_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()