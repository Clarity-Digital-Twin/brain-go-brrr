#!/usr/bin/env python
"""Robust GPU training with error handling and reduced memory usage."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:brain_go_brrr.data"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning import Trainer  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split  # noqa: E402

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: E402

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from train_tuab_probe import LinearProbeTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_robust.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Set seed for reproducibility
        pl.seed_everything(42)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load base config
        cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_config.yaml")

        # REDUCED settings for stability
        cfg.training.epochs = 10
        cfg.training.batch_size = 64  # Reduced from 128
        cfg.training.learning_rate = 5e-4
        cfg.training.patience = 5
        cfg.training.monitor = "val_loss"
        cfg.training.mode = "min"
        cfg.training.save_top_k = 3
        cfg.training.num_workers = 2  # REDUCED from 8 to avoid memory issues

        # Paths
        data_root = Path(os.environ["BGB_DATA_ROOT"])
        model_checkpoint = data_root / cfg.model.checkpoint_path
        dataset_root = data_root / cfg.data.root_dir
        log_dir = Path(f"logs/robust_gpu_run_{timestamp}")

        logger.info("=" * 60)
        logger.info("ROBUST GPU TRAINING (REDUCED MEMORY)")
        logger.info("=" * 60)
        logger.info(f"Starting training run: {timestamp}")
        logger.info(f"Batch size: {cfg.training.batch_size} (reduced)")
        logger.info(f"Workers: {cfg.training.num_workers} (reduced)")
        
        # Verify GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.error("No GPU detected!")
            sys.exit(1)

        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Initialize model
        logger.info("Loading model...")
        model = AbnormalityDetectionProbe(
            checkpoint_path=model_checkpoint,
            n_input_channels=20,
        )
        
        # Move model to GPU
        model = model.cuda()
        
        # Load dataset with error handling
        logger.info("Loading dataset...")
        try:
            full_dataset = TUABDataset(
                root_dir=dataset_root,
                split="eval",
                sampling_rate=cfg.data.sampling_rate,
                window_duration=cfg.data.window_duration,
                window_stride=cfg.data.window_stride,
                normalize=cfg.data.normalize,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        logger.info(f"Total windows: {len(full_dataset)}")
        logger.info(f"Distribution: {full_dataset.class_counts}")

        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Calculate class weights
        total_samples = len(full_dataset)
        class_counts = full_dataset.class_counts
        n_normal = class_counts.get('normal', 0)
        n_abnormal = class_counts.get('abnormal', 0)
        
        weight_normal = total_samples / (2 * n_normal) if n_normal > 0 else 1.0
        weight_abnormal = total_samples / (2 * n_abnormal) if n_abnormal > 0 else 1.0
        class_weights = torch.tensor([weight_normal, weight_abnormal])

        # Create weighted sampler
        train_labels = []
        for idx in train_dataset.indices:
            label = full_dataset.samples[idx]["label"]
            train_labels.append(label)
        
        sample_weights = torch.zeros(len(train_dataset))
        for i, label in enumerate(train_labels):
            sample_weights[i] = class_weights[label]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        # Create dataloaders with reduced workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Changed to False
            prefetch_factor=2,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Changed to False
            prefetch_factor=2,
        )

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
            filename="tuab_probe-{epoch:02d}-{val_loss:.4f}",
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

        # Trainer with stability settings
        trainer = Trainer(
            max_epochs=cfg.training.epochs,
            accelerator="gpu",
            devices=1,
            precision=16,
            logger=tb_logger,
            callbacks=[checkpoint_callback, early_stopping],
            default_root_dir=log_dir,
            log_every_n_steps=50,
            val_check_interval=0.5,  # Check twice per epoch
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,  # Effective batch size 128
            benchmark=False,  # More stable
        )

        logger.info("=" * 60)
        logger.info("Starting training:")
        logger.info(f"  Epochs: {cfg.training.epochs}")
        logger.info(f"  Batch size: {cfg.training.batch_size} x 2 accumulation = 128 effective")
        logger.info(f"  Workers: {cfg.training.num_workers}")
        logger.info("=" * 60)

        # Train with error handling
        try:
            trainer.fit(lightning_model, train_loader, val_loader)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            raise

        # Save final model
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Best model: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location="cuda")
            lightning_model.load_state_dict(checkpoint["state_dict"])
            probe_path = log_dir / "tuab_probe_best.pth"
            lightning_model.model.save_probe_weights(probe_path)
            logger.info(f"Saved probe weights to {probe_path}")

        logger.info("Training complete!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()