#!/usr/bin/env python3
"""Test EEGPT training with FAST loading - NO WAITING!"""

import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FastMockDataset(Dataset):
    """Ultra-fast mock dataset for testing training pipeline."""
    
    def __init__(self, n_samples=1000, n_channels=19, window_samples=2048):
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.window_samples = window_samples
        logger.info(f"Created mock dataset with {n_samples} samples")
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        # Return random data
        data = torch.randn(self.n_channels, self.window_samples)
        label = torch.tensor(idx % 2)  # Alternate labels
        return data, label


def test_fast_training():
    """Test training with mock data - NO FILE I/O!"""
    
    logger.info("=" * 80)
    logger.info("TESTING FAST TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Import after path setup
    from src.brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
    from src.brain_go_brrr.tasks.enhanced_abnormality_detection import EnhancedAbnormalityDetectionProbe
    import pytorch_lightning as pl
    
    # Create mock datasets
    train_dataset = FastMockDataset(n_samples=1000)
    val_dataset = FastMockDataset(n_samples=100)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Test loading speed
    start = time.time()
    for i, (data, labels) in enumerate(train_loader):
        if i == 0:
            logger.info(f"First batch loaded in {time.time() - start:.3f}s")
            logger.info(f"Batch shape: {data.shape}, labels: {labels.shape}")
        if i >= 5:
            break
    
    # Create model
    logger.info("\nCreating model...")
    checkpoint_path = Path(os.environ.get("BGB_DATA_ROOT", "data")) / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
        
    probe = EEGPTTwoLayerProbe(
        backbone_dim=768,
        n_input_channels=19,
        n_adapted_channels=19,
        hidden_dim=16,
        n_classes=2,
        dropout=0.5,
        use_channel_adapter=True,
    )
    
    model = EnhancedAbnormalityDetectionProbe(
        checkpoint_path=str(checkpoint_path),
        probe=probe,
        n_channels=19,
        learning_rate=0.001,
        weight_decay=0.01,
    )
    
    # Create trainer
    logger.info("\nCreating trainer...")
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu",
        devices=1,
        precision=16,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_checkpointing=False,
        enable_model_summary=True,
        gradient_clip_val=1.0,
    )
    
    # Train!
    logger.info("\nStarting training...")
    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"\n✅ Training completed in {time.time() - start:.1f}s!")
    
    # Check results
    if hasattr(trainer, 'logged_metrics'):
        logger.info(f"Final metrics: {trainer.logged_metrics}")


if __name__ == "__main__":
    test_fast_training()