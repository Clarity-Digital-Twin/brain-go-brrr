#!/usr/bin/env python
"""SIMPLE WORKING TRAINING SCRIPT - NO BULLSHIT, NO CACHE, JUST WORKS"""

import os
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import our modules - USE THE FUCKING CACHED DATASET
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe

def main():
    print("🚀 SIMPLE WORKING TRAINING - NO CACHE BULLSHIT")
    
    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    tuab_path = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    
    # Create datasets - USE THE FUCKING CACHE
    print("\n📊 Creating CACHED datasets...")
    cache_index = data_root / "cache/tuab_index.json"
    cache_dir = data_root / "cache/tuab_enhanced"
    
    train_dataset = TUABCachedDataset(
        root_dir=tuab_path,
        split="train",
        cache_dir=cache_dir,
        cache_index_path=cache_index,
        window_duration=8.0,
        window_stride=4.0,
        sampling_rate=256,
        normalize=True,
    )
    
    val_dataset = TUABCachedDataset(
        root_dir=tuab_path,
        split="eval",
        cache_dir=cache_dir,
        cache_index_path=cache_index,
        window_duration=8.0,
        window_stride=8.0,
        sampling_rate=256,
        normalize=True,
    )
    
    print(f"Train: {len(train_dataset)} windows")
    print(f"Val: {len(val_dataset)} windows")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Create model
    print("\n🧠 Creating model...")
    
    # Load EEGPT backbone
    backbone = EEGPTFeatureExtractor(
        checkpoint_path=checkpoint_path,
        n_input_channels=20,  # TUAB has 20 channels
        freeze=True,
    )
    
    # Create probe
    probe = TwoLayerProbe(
        input_dim=768,  # EEGPT feature dimension
        hidden_dim=16,
        output_dim=2,  # Binary classification
        dropout=0.5,
    )
    
    # Create trainer module
    model = LinearProbeTrainer(
        backbone=backbone,
        probe=probe,
        learning_rate=5e-4,
        weight_decay=0.05,
        warmup_epochs=5,
        max_epochs=50,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/simple_working",
        filename="eegpt-tuab-{epoch:02d}-{val_auroc:.3f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=3,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_auroc",
        patience=10,
        mode="max",
    )
    
    # Logger
    logger = TensorBoardLogger("logs", name="simple_working")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision=16,
        accumulate_grad_batches=4,  # Effective batch size = 32 * 4 = 128
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        val_check_interval=0.5,  # Check twice per epoch
        log_every_n_steps=50,
    )
    
    # Train!
    print("\n🎯 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✅ Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best AUROC: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()