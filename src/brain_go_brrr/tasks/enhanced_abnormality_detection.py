"""Enhanced abnormality detection probe with paper-matching features.

WARNING: This module uses PyTorch Lightning which has a CRITICAL BUG in v2.5.2
that causes training to hang with large cached datasets (>100k samples).
DO NOT USE for training! Use experiments/eegpt_linear_probe/train_pytorch_stable.py instead.
See experiments/eegpt_linear_probe/LIGHTNING_BUG_REPORT.md for details.
"""

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from ..models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
from ..models.eegpt_wrapper import create_normalized_eegpt

logger = logging.getLogger(__name__)


class EnhancedAbnormalityDetectionProbe(pl.LightningModule):
    """Enhanced Lightning module for EEGPT abnormality detection.

    Improvements:
    - Two-layer probe with dropout
    - Layer-wise learning rate decay
    - OneCycle learning rate schedule
    - Proper warmup handling
    - Channel adaptation
    """

    def __init__(
        self,
        checkpoint_path: str,
        probe: nn.Module | None = None,
        n_channels: int = 20,
        n_classes: int = 2,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        layer_decay: float = 0.65,
        scheduler_type: str = "onecycle",
        freeze_backbone: bool = True,
    ):
        """Initialize enhanced abnormality detection module."""
        super().__init__()
        self.save_hyperparameters(ignore=["probe"])

        # Store freeze setting first
        self.backbone_frozen = freeze_backbone

        # Initialize EEGPT backbone
        self.backbone = self._load_backbone(checkpoint_path, n_channels)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Initialize probe
        if probe is None:
            probe = EEGPTTwoLayerProbe(
                backbone_dim=768,
                n_input_channels=n_channels,
                n_classes=n_classes,
            )
        self.probe = probe

        # Loss function - NO label smoothing for binary classification to avoid NaN
        self.criterion = nn.CrossEntropyLoss()

        # Metrics storage
        self.train_outputs: list[dict[str, Any]] = []
        self.val_outputs: list[dict[str, Any]] = []
        self.test_outputs: list[dict[str, Any]] = []

        logger.info("Initialized EnhancedAbnormalityDetectionProbe:")
        logger.info(f"  Backbone frozen: {freeze_backbone}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Layer decay: {layer_decay}")
        logger.info(f"  Scheduler: {scheduler_type}")
        logger.info(f"  Warmup epochs: {warmup_epochs}/{total_epochs}")

    def _load_backbone(self, checkpoint_path: str, n_channels: int) -> nn.Module:  # noqa: ARG002
        """Load EEGPT backbone from checkpoint."""
        try:
            # Use the wrapper to create EEGPT model
            backbone = create_normalized_eegpt(checkpoint_path=checkpoint_path)

            logger.info(f"Loaded EEGPT backbone from {checkpoint_path}")
            return backbone

        except Exception as e:
            logger.error(f"Failed to load backbone: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and probe.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Logits [B, n_classes]
        """
        # Apply channel adaptation if needed
        if hasattr(self.probe, "adapt_channels"):
            x = self.probe.adapt_channels(x)

        # Extract features with backbone
        if self.backbone_frozen:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Apply probe
        logits = self.probe(features)

        return logits

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Training step."""
        x, y = batch

        # Forward pass
        logits = self(x)

        # Safety check for NaN in logits
        if torch.isnan(logits).any():
            raise RuntimeError(f"NaN detected in logits at step {self.global_step}")

        loss = self.criterion(logits, y)

        # Safety check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Loss became NaN/Inf at step {self.global_step}: {loss.item()}")

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-level metrics
        self.train_outputs.append(
            {
                "loss": loss.detach(),
                "logits": logits.detach(),
                "labels": y.detach(),
            }
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:  # noqa: ARG002
        """Validation step."""
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)

        # Store outputs
        self.val_outputs.append(
            {
                "loss": loss.detach(),
                "logits": logits.detach(),
                "labels": y.detach(),
            }
        )

    def on_train_epoch_end(self) -> None:
        """Calculate training metrics at epoch end."""
        if not self.train_outputs:
            return

        # Gather all outputs
        torch.cat([x["logits"] for x in self.train_outputs])
        torch.cat([x["labels"] for x in self.train_outputs])

        # Clear stored outputs
        self.train_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Calculate validation metrics at epoch end."""
        if not self.val_outputs:
            return

        # Gather all outputs
        all_logits = torch.cat([x["logits"] for x in self.val_outputs])
        all_labels = torch.cat([x["labels"] for x in self.val_outputs])
        avg_loss = torch.stack([x["loss"] for x in self.val_outputs]).mean()

        # Convert to numpy
        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        preds = all_logits.argmax(dim=1).cpu().numpy()
        labels = all_labels.cpu().numpy()

        # Calculate metrics
        metrics = self._calculate_metrics(labels, preds, probs[:, 1])

        # Log metrics
        self.log("val_loss", avg_loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"val_{name}", value, prog_bar=name in ["auroc", "acc"])

        # Clear stored outputs
        self.val_outputs.clear()

    def _calculate_metrics(
        self, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}

        # Basic metrics
        metrics["acc"] = accuracy_score(labels, preds)
        metrics["balanced_acc"] = balanced_accuracy_score(labels, preds)
        metrics["kappa"] = cohen_kappa_score(labels, preds)

        # F1 scores
        metrics["f1_weighted"] = f1_score(labels, preds, average="weighted")
        metrics["f1_macro"] = f1_score(labels, preds, average="macro")

        # AUROC (for binary classification)
        if len(np.unique(labels)) == 2:
            metrics["auroc"] = roc_auc_score(labels, probs)

        return metrics

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer with layer decay and scheduler."""
        # Build parameter groups with layer decay
        param_groups = self._get_param_groups()

        # Create optimizer
        optimizer = AdamW(
            param_groups,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Create scheduler based on type
        if self.hparams.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.hparams.warmup_epochs / self.hparams.total_epochs,
                anneal_strategy="cos",
                div_factor=25,  # Initial lr = max_lr / 25
                final_div_factor=1000,  # Final lr = max_lr / 1000
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            # Simple warmup + cosine annealing
            from torch.optim.lr_scheduler import CosineAnnealingLR

            def warmup_lambda(epoch: int) -> float:
                if epoch < self.hparams.warmup_epochs:
                    return float(epoch) / float(self.hparams.warmup_epochs)
                return 1.0

            # Use only cosine scheduler (warmup is handled by OneCycleLR-like behavior)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=1e-6,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

    def _get_param_groups(self) -> list:
        """Get parameter groups with layer decay."""
        param_groups = []

        # Probe parameters - base learning rate
        probe_params = []
        for _name, param in self.probe.named_parameters():
            if param.requires_grad:
                probe_params.append(param)

        param_groups.append(
            {
                "params": probe_params,
                "lr": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
                "name": "probe",
            }
        )

        # Backbone parameters (if unfrozen) - apply layer decay
        if not self.backbone_frozen:
            for layer_id in range(12):  # EEGPT has 12 layers
                layer_params = []
                for name, param in self.backbone.named_parameters():
                    if param.requires_grad and f"layers.{layer_id}." in name:
                        layer_params.append(param)

                if layer_params:
                    lr_scale = self.hparams.layer_decay ** (11 - layer_id)
                    param_groups.append(
                        {
                            "params": layer_params,
                            "lr": self.hparams.learning_rate * lr_scale,
                            "weight_decay": self.hparams.weight_decay,
                            "name": f"backbone_layer_{layer_id}",
                        }
                    )

        return param_groups
