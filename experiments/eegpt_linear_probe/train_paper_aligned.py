#!/usr/bin/env python
"""EEGPT Linear Probe Training - Paper-Aligned Implementation.

Based on EEGPT paper hyperparameters and best practices to avoid NaN issues.
"""

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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error messages

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
        logging.FileHandler('training_paper_aligned.log'),
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
        
        # Load base config and override with paper parameters
        cfg = OmegaConf.load(Path(__file__).parent / "configs/tuab_config.yaml")
        
        # PAPER-ALIGNED HYPERPARAMETERS
        cfg.training.epochs = 30  # Paper uses 30 for downstream
        cfg.training.batch_size = 100  # Paper uses 100 for TUAB specifically
        cfg.training.learning_rate = 5e-4  # Paper uses 5e-4 for TUAB
        cfg.training.weight_decay = 0.05  # From paper
        cfg.training.warmup_epochs = 5  # From reference implementation
        cfg.training.num_workers = 4  # Stable value
        
        # SAFETY SETTINGS TO PREVENT NaN
        cfg.training.gradient_clip_val = 1.0
        cfg.training.precision = 32  # Start with fp32 for stability
        cfg.training.accumulate_grad_batches = 1  # No accumulation initially
        
        # Paths
        data_root = Path(os.environ["BGB_DATA_ROOT"])
        checkpoint_path = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EEGPT checkpoint not found: {checkpoint_path}")
        
        logger.info("=" * 60)
        logger.info("EEGPT Linear Probe Training - Paper-Aligned")
        logger.info("=" * 60)
        logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")
        
        # Create datasets
        # Convert window_duration to window_size in samples
        window_size = int(cfg.data.window_duration * cfg.data.sampling_rate)  # 8s * 256Hz = 2048 samples
        
        train_dataset = TUABDataset(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1",
            split="train",
            window_size=window_size,
            stride=window_size,  # Non-overlapping
            sampling_rate=cfg.data.sampling_rate,
            apply_augmentation=False,  # No augmentation for linear probe
        )
        
        val_dataset = TUABDataset(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1",
            split="eval",
            window_size=window_size,
            stride=window_size,
            sampling_rate=cfg.data.sampling_rate,
            apply_augmentation=False,
        )
        
        # Log dataset info
        logger.info(f"Train dataset: {len(train_dataset)} windows")
        logger.info(f"Val dataset: {len(val_dataset)} windows")
        
        # Check class balance
        train_labels = [train_dataset[i][1] for i in range(min(1000, len(train_dataset)))]
        unique, counts = np.unique(train_labels, return_counts=True)
        logger.info(f"Train label distribution (first 1000): {dict(zip(unique, counts))}")
        
        # Create balanced sampler
        all_labels = []
        for i in range(len(train_dataset)):
            try:
                _, label = train_dataset[i]
                all_labels.append(label)
            except Exception as e:
                logger.warning(f"Error getting label for sample {i}: {e}")
                all_labels.append(0)  # Default to normal
        
        class_counts = np.bincount(all_labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[label] for label in all_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.training.num_workers > 0 else False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size * 2,  # Can use larger batch for validation
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.training.num_workers > 0 else False,
        )
        
        # Create model with NaN protection
        class RobustLinearProbeTrainer(LinearProbeTrainer):
            def validation_step(self, batch, batch_idx):
                x, labels = batch
                logits = self(x)
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    logger.warning(f"NaN in logits at val batch {batch_idx}")
                    # Replace NaN with zeros
                    logits = torch.nan_to_num(logits, nan=0.0)
                
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)
                
                # Check for NaN in probs
                if torch.isnan(probs).any():
                    logger.warning(f"NaN in probs at val batch {batch_idx}")
                    probs = torch.nan_to_num(probs, nan=0.5)
                
                preds = (probs > 0.5).long()
                
                self.validation_step_outputs.append({
                    "loss": loss,
                    "probs": probs,
                    "preds": preds,
                    "labels": labels,
                })
                
                return loss
            
            def on_validation_epoch_end(self):
                import numpy as np
                from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
                
                # Gather all outputs
                all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])
                all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
                all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
                
                # Calculate metrics
                avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
                acc = (all_preds == all_labels).float().mean()
                
                # Move to CPU for sklearn metrics
                probs_cpu = all_probs.cpu().numpy()
                labels_cpu = all_labels.cpu().numpy()
                
                # Comprehensive NaN handling
                if np.isnan(probs_cpu).any():
                    self.log("val/has_nan", 1.0)
                    nan_count = np.isnan(probs_cpu).sum()
                    logger.warning(f"Found {nan_count} NaN values in predictions")
                    # Replace NaN with class-balanced random predictions
                    nan_mask = np.isnan(probs_cpu)
                    probs_cpu[nan_mask] = 0.5  # Neutral prediction
                else:
                    self.log("val/has_nan", 0.0)
                
                # Check for single-class batches
                unique_labels = np.unique(labels_cpu)
                if len(unique_labels) < 2:
                    logger.warning(f"Single class in validation: {unique_labels}")
                    self.log("val/loss", avg_loss)
                    self.log("val/acc", acc)
                    self.log("val/auroc", 0.5)  # Undefined, use neutral
                    self.log("val/pr_auc", 0.5)
                else:
                    # Calculate AUROC and PR-AUC
                    try:
                        auroc = roc_auc_score(labels_cpu, probs_cpu)
                        precision, recall, _ = precision_recall_curve(labels_cpu, probs_cpu)
                        pr_auc = auc(recall, precision)
                    except Exception as e:
                        logger.error(f"Error calculating metrics: {e}")
                        auroc = 0.5
                        pr_auc = 0.5
                    
                    # Log metrics
                    self.log("val/loss", avg_loss)
                    self.log("val/acc", acc)
                    self.log("val/auroc", auroc)
                    self.log("val/pr_auc", pr_auc)
                    
                    logger.info(f"Val - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, AUROC: {auroc:.4f}")
                
                # Clear outputs
                self.validation_step_outputs.clear()
        
        # Initialize model
        probe = AbnormalityDetectionProbe(checkpoint_path, n_input_channels=20)
        lightning_model = RobustLinearProbeTrainer(
            model=probe,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_epochs=cfg.training.warmup_epochs,
            max_epochs=cfg.training.epochs,
        )
        
        # Callbacks
        log_dir = Path("logs") / f"paper_aligned_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val/auroc",
            dirpath=log_dir / "checkpoints",
            filename="epoch={epoch}-auroc={val/auroc:.4f}",
            save_top_k=3,
            mode="max",
            save_last=True,
        )
        
        early_stopping = EarlyStopping(
            monitor="val/auroc",
            patience=10,
            mode="max",
            verbose=True,
        )
        
        # Logger
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="tuab_linear_probe",
            version=timestamp,
        )
        
        # Trainer with conservative settings
        trainer = Trainer(
            max_epochs=cfg.training.epochs,
            accelerator="gpu",
            devices=1,
            precision=cfg.training.precision,
            logger=tb_logger,
            callbacks=[checkpoint_callback, early_stopping],
            default_root_dir=log_dir,
            log_every_n_steps=10,
            val_check_interval=0.5,  # Check twice per epoch
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=cfg.training.gradient_clip_val,
            accumulate_grad_batches=cfg.training.accumulate_grad_batches,
            deterministic=True,  # For reproducibility
            benchmark=False,  # More stable
        )
        
        logger.info("=" * 60)
        logger.info("Starting training with paper-aligned hyperparameters:")
        logger.info(f"  Epochs: {cfg.training.epochs}")
        logger.info(f"  Batch size: {cfg.training.batch_size}")
        logger.info(f"  Learning rate: {cfg.training.learning_rate}")
        logger.info(f"  Weight decay: {cfg.training.weight_decay}")
        logger.info(f"  Precision: {cfg.training.precision}")
        logger.info(f"  Gradient clip: {cfg.training.gradient_clip_val}")
        logger.info("=" * 60)
        
        # Train
        trainer.fit(lightning_model, train_loader, val_loader)
        
        # Save final model
        torch.save(lightning_model.state_dict(), log_dir / "final_model.pth")
        logger.info(f"Training completed! Model saved to {log_dir}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()