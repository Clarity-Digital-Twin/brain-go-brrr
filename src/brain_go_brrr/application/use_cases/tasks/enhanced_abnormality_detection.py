"""Enhanced abnormality detection probe with paper-matching features.

⚠️ CRITICAL: DO NOT USE THIS MODULE FOR TRAINING!
PyTorch Lightning 2.5.2 has a CRITICAL BUG that causes training to hang/crash
with large cached datasets (>100k samples). This module is kept ONLY for:
- Unit test compatibility
- Reference implementation

FOR ACTUAL TRAINING: Use experiments/eegpt_linear_probe/train_tuab.py (pure PyTorch)
See AGENTS.md and CLAUDE.md for details on the Lightning bug.
"""

import logging
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
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

# P1 FIX: Removed direct EEGPTProbe import - using ProbeFactory instead
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.utils import mask_path_for_log

logger = logging.getLogger(__name__)


class HParams(TypedDict, total=False):
    """Typed hyperparameters for Lightning module."""

    learning_rate: float
    weight_decay: float
    scheduler_type: str  # "onecycle" | "cosine" | "none"
    warmup_epochs: int
    total_epochs: int
    layer_decay: float
    batch_size: int
    max_epochs: int


class EnhancedAbnormalityDetectionProbe(nn.Module):  # Changed from pl.LightningModule
    """Enhanced Lightning module for EEGPT abnormality detection.

    Improvements:
    - Two-layer probe with dropout
    - Layer-wise learning rate decay
    - OneCycle learning rate schedule
    - Proper warmup handling
    - Channel adaptation
    """

    # Lightning's hparams is a MutableMapping, not our HParams type
    # We access it dynamically via self.hparams

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
        # self.save_hyperparameters(ignore=["probe"])  # Lightning-specific, removed

        # Store hyperparameters manually (replacing Lightning's save_hyperparameters)
        self.hparams = {
            "checkpoint_path": checkpoint_path,
            "layer_decay": layer_decay,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "scheduler_type": scheduler_type,
            "freeze_backbone": freeze_backbone,
        }

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
            # P1 FIX: Use ProbeFactory instead of direct EEGPTProbe
            from brain_go_brrr.infra.ml_models.probe_factory import ProbeFactory

            probe = ProbeFactory.create_for_task(
                task="abnormality",
                backbone=self.backbone,  # Use the already loaded backbone
                architecture='two_layer',
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

            logger.info(f"Loaded EEGPT backbone from {mask_path_for_log(checkpoint_path)}")
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
        # Extract features with backbone
        if self.backbone_frozen:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Apply probe
        logits = self.probe(features)

        return cast("torch.Tensor", logits)

    # REMOVED: Lightning-specific training methods
    # Use plain PyTorch training loop instead
    def training_step_deprecated(
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

        # Log metrics (only if trainer is attached to avoid warnings in tests)
        if getattr(self, "trainer", None) is not None:
            # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            # Use standard logging instead
            logger.debug(f"train_loss: {loss:.4f}, train_acc: {acc:.4f}")

        # Store outputs for epoch-level metrics
        self.train_outputs.append(
            {"loss": loss.detach(), "logits": logits.detach(), "labels": y.detach()}
        )

        return cast("torch.Tensor", loss)

    def validation_step_deprecated(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step."""
        _ = batch_idx  # Required by PyTorch Lightning interface
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)

        # Store outputs
        self.val_outputs.append(
            {"loss": loss.detach(), "logits": logits.detach(), "labels": y.detach()}
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

        # Log metrics (only if trainer is attached to avoid warnings in tests)
        if getattr(self, "trainer", None) is not None:
            # self.log("val_loss", avg_loss, prog_bar=True, logger=True)
            # for name, value in metrics.items():
            #     self.log(f"val_{name}", value, prog_bar=name in ["auroc", "acc"], logger=True)
            logger.info(f"val_loss: {avg_loss:.4f}, metrics: {metrics}")

        # Clear stored outputs
        self.val_outputs.clear()

    def _calculate_metrics(
        self,
        labels: npt.NDArray[np.float64],
        preds: npt.NDArray[np.float64],
        probs: npt.NDArray[np.float64],
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

    # NOTE: Former Lightning-only optimizer config removed.
    # This module is kept for reference and testing; use pure PyTorch training scripts instead.

    def _get_param_groups(self) -> list[dict[str, Any]]:
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
                "lr": self.hparams.get("learning_rate", 1e-3),
                "weight_decay": self.hparams.get("weight_decay", 0.01),
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
                    lr_scale = float(self.hparams.get("layer_decay", 0.75)) ** (11 - layer_id)  # type: ignore[arg-type]
                    param_groups.append(
                        {
                            "params": layer_params,
                            "lr": float(self.hparams.get("learning_rate", 1e-3)) * lr_scale,  # type: ignore[arg-type]
                            "weight_decay": self.hparams.get("weight_decay", 0.01),
                            "name": f"backbone_layer_{layer_id}",
                        }
                    )

        return param_groups
