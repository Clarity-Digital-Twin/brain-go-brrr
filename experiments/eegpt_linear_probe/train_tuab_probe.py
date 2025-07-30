"""Train EEGPT Linear Probe on TUAB dataset.

Target: AUROC ≥ 0.93 for abnormality detection
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402
from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearProbeTrainer(pl.LightningModule):
    """PyTorch Lightning module for linear probe training."""

    def __init__(
        self,
        model: AbnormalityDetectionProbe,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        pct_start: float = 0.2,
        div_factor: float = 25,
        final_div_factor: float = 10000,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: The linear probe model
            learning_rate: Max learning rate for OneCycleLR
            weight_decay: Weight decay for AdamW
            pct_start: Percentage of cycle for increasing LR
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Min LR = initial_lr / final_div_factor
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Loss function with optional class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # For validation metrics
        self.validation_step_outputs = []

        # Log hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Store for epoch-end metrics
        output = {
            "loss": loss,
            "probs": probs[:, 1],  # Abnormal probability
            "labels": y,
            "preds": logits.argmax(dim=-1),
        }

        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self) -> None:
        """Calculate validation metrics at epoch end."""
        # Gather all outputs
        all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])

        # Calculate metrics
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        acc = (all_preds == all_labels).float().mean()

        # Move to CPU for sklearn metrics
        probs_cpu = all_probs.cpu().numpy()
        labels_cpu = all_labels.cpu().numpy()

        # AUROC
        auroc = roc_auc_score(labels_cpu, probs_cpu)

        # PR AUC
        precision, recall, _ = precision_recall_curve(labels_cpu, probs_cpu)
        pr_auc = auc(recall, precision)

        # Log metrics
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_pr_auc", pr_auc, prog_bar=False)

        # Log to console
        logger.info(
            f"Validation - Loss: {avg_loss:.4f}, Acc: {acc:.4f}, "
            f"AUROC: {auroc:.4f}, PR-AUC: {pr_auc:.4f}"
        )

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> dict[str, torch.Tensor]:
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Calculate test metrics at epoch end."""
        self.on_validation_epoch_end()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Only optimize non-frozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        logger.info(
            f"Optimizing {len(trainable_params)} parameter groups "
            f"({self.model.get_num_trainable_params():,} parameters)"
        )

        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.pct_start,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train_tuab_probe(cfg: DictConfig) -> None:
    """Main training function."""
    # Set seed
    pl.seed_everything(cfg.experiment.seed)

    # Create output directories
    checkpoint_dir = Path(cfg.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg.output.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    logger.info("Initializing model...")
    model = AbnormalityDetectionProbe(
        checkpoint_path=Path(cfg.model.checkpoint_path), n_input_channels=cfg.model.n_input_channels
    )

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = TUABDataset(
        root_dir=Path(cfg.data.root_dir),
        split="train",
        sampling_rate=cfg.data.sampling_rate,
        window_duration=cfg.data.window_duration,
        window_stride=cfg.data.window_stride,
        normalize=cfg.data.normalize,
    )

    val_dataset = TUABDataset(
        root_dir=Path(cfg.data.root_dir),
        split="val",
        sampling_rate=cfg.data.sampling_rate,
        window_duration=cfg.data.window_duration,
        window_stride=cfg.data.window_stride,
        normalize=cfg.data.normalize,
    )

    # Get class weights for balanced training
    class_weights = train_dataset.get_class_weights()
    logger.info(f"Class weights: {class_weights}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Initialize Lightning module
    lightning_model = LinearProbeTrainer(
        model=model,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        pct_start=cfg.training.pct_start,
        div_factor=cfg.training.div_factor,
        final_div_factor=cfg.training.final_div_factor,
        class_weights=class_weights.to("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tuab_probe-{epoch:02d}-{val_auroc:.4f}",
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        save_top_k=cfg.training.save_top_k,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor=cfg.training.monitor,
        patience=cfg.training.patience,
        mode=cfg.training.mode,
        verbose=True,
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir, name=cfg.experiment.name, version=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.experiment.accelerator,
        precision=cfg.experiment.precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping, RichProgressBar()],
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

    # Load best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Loading best model from {best_model_path}")

    # Save probe weights only
    probe_save_path = Path(cfg.output.probe_save_path)
    probe_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load best checkpoint and save probe
    checkpoint = torch.load(best_model_path)
    lightning_model.load_state_dict(checkpoint["state_dict"])
    lightning_model.model.save_probe(probe_save_path)

    # Test on test set if available
    try:
        test_dataset = TUABDataset(
            root_dir=Path(cfg.data.root_dir),
            split="test",
            sampling_rate=cfg.data.sampling_rate,
            window_duration=cfg.data.window_duration,
            window_stride=cfg.data.window_stride,
            normalize=cfg.data.normalize,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

        logger.info("Running test evaluation...")
        test_results = trainer.test(lightning_model, test_loader)

        # Log final results
        logger.info(f"Test results: {test_results}")

    except Exception as e:
        logger.warning(f"Could not run test evaluation: {e}")

    logger.info("Training complete!")


if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent / "configs" / "tuab_config.yaml"

    with open(config_path) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # Run training
    train_tuab_probe(cfg)
