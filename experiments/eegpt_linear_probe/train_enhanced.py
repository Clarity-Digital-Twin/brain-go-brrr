#!/usr/bin/env python
"""Enhanced EEGPT training with all paper-matching improvements."""

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
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from brain_go_brrr.tasks.enhanced_abnormality_detection import EnhancedAbnormalityDetectionProbe
from experiments.eegpt_linear_probe.custom_collate import collate_eeg_batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LayerDecayOptimizer:
    """Optimizer with layer-wise learning rate decay."""
    
    def __init__(self, model, base_lr=5e-4, weight_decay=0.05, layer_decay=0.65):
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        
        # Build parameter groups with decayed learning rates
        self.param_groups = self._build_param_groups(model)
    
    def _build_param_groups(self, model):
        """Build parameter groups with layer-wise decay."""
        param_groups = []
        
        # Backbone parameters (if unfrozen) - apply layer decay
        if hasattr(model, 'backbone') and not model.backbone_frozen:
            # Get depth of each parameter
            for i, (name, param) in enumerate(model.backbone.named_parameters()):
                if not param.requires_grad:
                    continue
                    
                # Estimate layer depth from parameter name
                layer_id = self._get_layer_id(name)
                lr_scale = self.layer_decay ** (12 - layer_id)  # Assuming 12 layers
                
                param_groups.append({
                    'params': [param],
                    'lr': self.base_lr * lr_scale,
                    'weight_decay': self.weight_decay,
                    'name': f'backbone.{name}'
                })
        
        # Probe parameters - use base learning rate
        probe_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'backbone' not in name:
                probe_params.append(param)
        
        if probe_params:
            param_groups.append({
                'params': probe_params,
                'lr': self.base_lr,
                'weight_decay': self.weight_decay,
                'name': 'probe'
            })
        
        return param_groups
    
    def _get_layer_id(self, name):
        """Extract layer ID from parameter name."""
        if 'layers' in name:
            return int(name.split('layers.')[1].split('.')[0])
        return 0
    
    def get_optimizer(self):
        """Get AdamW optimizer with parameter groups."""
        return torch.optim.AdamW(self.param_groups)


def create_weighted_sampler(dataset, num_samples=None):
    """Create weighted sampler for balanced training."""
    # Handle different dataset types
    if hasattr(dataset, 'get_sample_weights'):
        weights = dataset.get_sample_weights()
    else:
        # Fallback for cached dataset
        weights = []
        for sample in dataset.samples:
            label = sample['label']
            # Inverse frequency weighting
            weight = 1.0 / dataset.class_counts[sample['class_name']]
            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float32)
    
    if num_samples is None:
        num_samples = len(dataset)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler


def main():
    # Seed everything
    pl.seed_everything(42, workers=True)
    
    # Load configuration
    # Check for environment variable or use memsafe config
    config_file = os.environ.get("EEGPT_CONFIG", "configs/tuab_memsafe.yaml")
    cfg = OmegaConf.load(Path(__file__).parent / config_file)
    
    # Paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    logger.info("=" * 80)
    logger.info("ENHANCED EEGPT TRAINING")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config_file}")
    logger.info(f"Key settings from config:")
    logger.info(f"  ✓ {cfg.data.window_duration}s windows @ {cfg.data.sampling_rate}Hz")
    logger.info(f"  ✓ Two-layer probe with {cfg.model.probe.dropout} dropout")
    logger.info(f"  ✓ Batch size {cfg.data.batch_size}")
    logger.info(f"  ✓ {cfg.training.epochs} epochs with {cfg.training.warmup_epochs}-epoch warmup")
    logger.info(f"  ✓ Layer decay {cfg.training.layer_decay}")
    logger.info(f"  ✓ Weight decay {cfg.training.weight_decay}")
    logger.info(f"  ✓ {cfg.training.scheduler} learning rate schedule")
    logger.info(f"  ✓ {cfg.data.bandpass_low}-{cfg.data.bandpass_high}Hz bandpass")
    logger.info("=" * 80)
    
    # Create datasets
    logger.info("Creating enhanced TUAB datasets...")
    
    # Use cached dataset if specified
    if hasattr(cfg.data, 'use_cached_dataset') and cfg.data.use_cached_dataset:
        from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        DatasetClass = TUABCachedDataset
        logger.info("Using CACHED dataset for fast loading!")
    else:
        DatasetClass = TUABEnhancedDataset
    
    # Additional kwargs for cached dataset
    extra_kwargs = {}
    if hasattr(cfg.data, 'max_files') and cfg.data.max_files:
        extra_kwargs['max_files'] = cfg.data.max_files
    if hasattr(cfg.data, 'cache_index_path'):
        extra_kwargs['cache_index_path'] = Path(cfg.data.cache_index_path)
    
    # Create train dataset with appropriate args
    if hasattr(cfg.data, 'use_cached_dataset') and cfg.data.use_cached_dataset:
        # Cached dataset has simpler args
        train_dataset = DatasetClass(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            split="train",
            window_duration=cfg.data.window_duration,
            window_stride=cfg.data.window_stride,
            sampling_rate=cfg.data.sampling_rate,
            preload=False,
            normalize=True,
            cache_dir=data_root / "cache/tuab_enhanced",
            **extra_kwargs
        )
    else:
        # Enhanced dataset with all args
        train_dataset = DatasetClass(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            split="train",
            window_duration=cfg.data.window_duration,
            window_stride=cfg.data.window_stride,
            sampling_rate=cfg.data.sampling_rate,
            channels=cfg.data.channel_names,
            preload=False,
            normalize=True,
            bandpass_low=cfg.data.bandpass_low,
            bandpass_high=cfg.data.bandpass_high,
            notch_freq=cfg.data.notch_filter,
            cache_dir=data_root / "cache/tuab_enhanced",
            use_old_naming=True,
            n_jobs=4,
            use_autoreject=getattr(cfg.data, 'use_autoreject', False),
            ar_cache_dir=str(data_root / "cache/autoreject"),
            **extra_kwargs
        )
    
    # Create val dataset with appropriate args
    if hasattr(cfg.data, 'use_cached_dataset') and cfg.data.use_cached_dataset:
        # Cached dataset has simpler args
        val_dataset = DatasetClass(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            split="eval",
            window_duration=cfg.data.window_duration,
            window_stride=cfg.data.window_duration,  # No overlap for validation
            sampling_rate=cfg.data.sampling_rate,
            preload=False,
            normalize=True,
            cache_dir=data_root / "cache/tuab_enhanced",
            **extra_kwargs
        )
    else:
        # Enhanced dataset with all args
        val_dataset = DatasetClass(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            split="eval",
            window_duration=cfg.data.window_duration,
            window_stride=cfg.data.window_duration,  # No overlap for validation
            sampling_rate=cfg.data.sampling_rate,
            channels=cfg.data.channel_names,
            preload=False,
            normalize=True,
            bandpass_low=cfg.data.bandpass_low,
            bandpass_high=cfg.data.bandpass_high,
            notch_freq=cfg.data.notch_filter,
            cache_dir=data_root / "cache/tuab_enhanced",
            use_old_naming=True,
            n_jobs=4,
            use_autoreject=getattr(cfg.data, 'use_autoreject', False),
            ar_cache_dir=str(data_root / "cache/autoreject"),
            **extra_kwargs
        )
    
    logger.info(f"Train dataset: {len(train_dataset)} windows")
    logger.info(f"Val dataset: {len(val_dataset)} windows")
    logger.info(f"Class distribution: {train_dataset.class_counts}")
    
    # Create data loaders
    train_sampler = create_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        drop_last=True,
        collate_fn=collate_eeg_batch,  # Custom collate for consistent dimensions
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        collate_fn=collate_eeg_batch,  # Custom collate for consistent dimensions
    )
    
    # Create model
    logger.info("Creating enhanced model with two-layer probe...")
    
    # Initialize probe
    probe = EEGPTTwoLayerProbe(
        backbone_dim=768,  # EEGPT hidden dimension
        n_input_channels=cfg.model.probe.channel_adapter_in,
        n_adapted_channels=cfg.model.probe.channel_adapter_out,
        hidden_dim=cfg.model.probe.hidden_dim,
        n_classes=cfg.model.probe.n_classes,
        dropout=cfg.model.probe.dropout,
        use_channel_adapter=cfg.model.probe.use_channel_adapter,
    )
    
    # Create Lightning module
    lightning_model = EnhancedAbnormalityDetectionProbe(
        checkpoint_path=str(checkpoint_path),
        probe=probe,
        n_channels=cfg.model.probe.channel_adapter_in,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        total_epochs=cfg.training.epochs,
        layer_decay=cfg.training.layer_decay,
        scheduler_type=cfg.training.scheduler,
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"enhanced_{timestamp}"
    
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.monitor,
        dirpath=log_dir / "checkpoints",
        filename="tuab-enhanced-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=cfg.logging.save_top_k,
        mode=cfg.training.mode,
        save_last=True,
        save_weights_only=False,
    )
    
    early_stopping = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode=cfg.training.mode,
        verbose=True,
        min_delta=0.001,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = RichProgressBar()
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="tuab_enhanced",
        version=timestamp,
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.experiment.precision,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            progress_bar,
        ],
        default_root_dir=log_dir,
        gradient_clip_val=cfg.training.gradient_clip_val,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
        enable_model_summary=True,
    )
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create log directory and save configuration
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")
    
    # Train
    logger.info("Starting enhanced training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    
    # Log results
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best AUROC: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Logs saved to: {log_dir}")
    logger.info("=" * 80)
    
    # Save final results
    results = {
        "best_auroc": float(checkpoint_callback.best_model_score),
        "best_epoch": checkpoint_callback.best_k_models,
        "config": OmegaConf.to_container(cfg),
    }
    
    import json
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()