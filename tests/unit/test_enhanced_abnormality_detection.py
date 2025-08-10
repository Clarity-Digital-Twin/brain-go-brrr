"""CLEAN tests for Enhanced Abnormality Detection - no Lightning, pure logic."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from brain_go_brrr.tasks.enhanced_abnormality_detection import (
    EnhancedAbnormalityDetectionProbe,
    HParams,
)


class TestEnhancedAbnormalityDetectionClean:
    """Test enhanced abnormality detection WITHOUT PyTorch Lightning."""

    @pytest.fixture
    def hparams(self):
        """Create hyperparameters for testing."""
        return HParams(
            learning_rate=1e-4,
            weight_decay=0.01,
            scheduler_type="onecycle",
            warmup_epochs=2,
            total_epochs=10,
            layer_decay=0.8,
            batch_size=32,
            max_epochs=10
        )

    @pytest.fixture
    def synthetic_batch(self):
        """Create synthetic EEG batch."""
        np.random.seed(1337)
        torch.manual_seed(1337)

        batch_size = 8
        n_channels = 20
        n_samples = 2048  # 8 seconds at 256 Hz

        # Create batch
        x = torch.randn(batch_size, n_channels, n_samples) * 20e-6
        y = torch.randint(0, 2, (batch_size,))  # Binary labels

        return x.float(), y.long()

    @pytest.fixture
    def fake_checkpoint_path(self, tmp_path):
        """Create fake EEGPT checkpoint."""
        checkpoint_path = tmp_path / "eegpt.ckpt"
        checkpoint = {
            "state_dict": {},
            "config": {
                "embed_dim": 512,
                "n_summary_tokens": 4,
                "patch_size": 64
            }
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_init_creates_probe(self, fake_checkpoint_path, hparams):
        """Test initialization of enhanced probe."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            window_size=8.0,
            sampling_rate=256,
            hparams=hparams
        )

        assert probe is not None
        assert probe.n_channels == 20
        assert probe.n_classes == 2
        assert probe.window_size == 8.0
        assert probe.sampling_rate == 256
        assert probe.hparams["learning_rate"] == 1e-4

    def test_configure_optimizers(self, fake_checkpoint_path, hparams):
        """Test optimizer configuration."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Mock the probe parameters
        probe.probe = MagicMock()
        probe.probe.parameters = MagicMock(return_value=[
            torch.nn.Parameter(torch.randn(10, 10)),
            torch.nn.Parameter(torch.randn(5, 5))
        ])

        # Configure optimizers
        opt_config = probe.configure_optimizers()

        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

        optimizer = opt_config["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_forward_pass(self, fake_checkpoint_path, synthetic_batch, hparams):
        """Test forward pass through the probe."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Mock the underlying probe
        probe.probe = MagicMock()
        probe.probe.return_value = torch.randn(8, 2)  # batch_size=8, n_classes=2

        x, _ = synthetic_batch
        output = probe(x)

        assert output.shape == (8, 2)
        probe.probe.assert_called_once()

    def test_training_step(self, fake_checkpoint_path, synthetic_batch, hparams):
        """Test training step logic."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Mock the probe forward
        probe.probe = MagicMock()
        probe.probe.return_value = torch.randn(8, 2)

        batch = synthetic_batch
        batch_idx = 0

        # Execute training step
        loss = probe.training_step(batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.ndim == 0  # Scalar

    def test_validation_step(self, fake_checkpoint_path, synthetic_batch, hparams):
        """Test validation step with metrics."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Mock probe
        logits = torch.tensor([
            [2.0, -1.0],
            [-1.0, 2.0],
            [3.0, -2.0],
            [-2.0, 3.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
        ])
        probe.probe = MagicMock(return_value=logits)

        x, y = synthetic_batch
        batch = (x, y)
        batch_idx = 0

        # Mock the logging
        probe.log = MagicMock()

        # Execute validation step
        result = probe.validation_step(batch, batch_idx)

        assert "val_loss" in result
        assert "val_acc" in result
        assert "val_auroc" in result
        assert result["val_acc"] >= 0 and result["val_acc"] <= 1

    def test_compute_metrics(self, fake_checkpoint_path, hparams):
        """Test metric computation."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Create predictions and targets
        preds = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
        targets = torch.tensor([0, 1, 0, 0, 1, 1, 1, 0])
        probs = torch.tensor([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.9, 0.1],
        ])

        metrics = probe.compute_metrics(preds, targets, probs)

        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert "f1_score" in metrics
        assert "auroc" in metrics
        assert "cohen_kappa" in metrics

        # Check metric ranges
        for metric_name, value in metrics.items():
            assert value >= 0 and value <= 1, f"{metric_name} out of range"

    def test_layer_wise_decay(self, fake_checkpoint_path, hparams):
        """Test layer-wise learning rate decay."""
        hparams["layer_decay"] = 0.8

        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Create mock layers
        probe.probe = MagicMock()
        probe.probe.eegpt_backbone = MagicMock()
        probe.probe.probe_head = MagicMock()

        # Mock parameters for different layers
        backbone_params = [
            torch.nn.Parameter(torch.randn(10, 10)),
            torch.nn.Parameter(torch.randn(5, 5))
        ]
        head_params = [
            torch.nn.Parameter(torch.randn(2, 512)),
            torch.nn.Parameter(torch.randn(2))
        ]

        probe.probe.eegpt_backbone.parameters = MagicMock(return_value=backbone_params)
        probe.probe.probe_head.parameters = MagicMock(return_value=head_params)

        # Apply layer-wise decay
        param_groups = probe.get_parameter_groups_with_decay()

        assert len(param_groups) >= 2
        # Backbone should have lower LR due to decay
        assert param_groups[0]["lr"] < param_groups[-1]["lr"]

    def test_class_weight_handling(self, fake_checkpoint_path, hparams):
        """Test handling of class weights for imbalanced data."""
        class_weights = torch.tensor([1.0, 3.0])  # Weight abnormal class more

        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            class_weights=class_weights,
            hparams=hparams
        )

        assert torch.allclose(probe.class_weights, class_weights)

        # Test loss computation with weights
        logits = torch.randn(8, 2)
        targets = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])

        loss = probe.compute_loss(logits, targets)
        assert loss > 0

    def test_early_stopping_logic(self, fake_checkpoint_path, hparams):
        """Test early stopping logic."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            early_stopping_patience=3,
            hparams=hparams
        )

        # Simulate validation losses
        val_losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38]  # Starts improving then degrades

        for epoch, loss in enumerate(val_losses):
            should_stop = probe.check_early_stopping(loss, epoch)

            if epoch < 3:
                assert not should_stop  # Should not stop while improving
            elif epoch >= 5:
                assert should_stop  # Should stop after patience exhausted

    def test_confidence_thresholding(self, fake_checkpoint_path, hparams):
        """Test confidence-based prediction thresholding."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            confidence_threshold=0.7,
            hparams=hparams
        )

        # Create predictions with varying confidence
        probs = torch.tensor([
            [0.9, 0.1],   # High confidence normal
            [0.6, 0.4],   # Low confidence normal
            [0.2, 0.8],   # High confidence abnormal
            [0.45, 0.55], # Low confidence abnormal
        ])

        preds, confidences = probe.predict_with_confidence(probs)

        # Low confidence predictions should be marked uncertain
        assert preds[1] == -1  # Uncertain due to low confidence
        assert preds[3] == -1  # Uncertain due to low confidence
        assert confidences[0] > 0.7
        assert confidences[1] < 0.7

    def test_mixup_augmentation(self, fake_checkpoint_path, synthetic_batch, hparams):
        """Test mixup data augmentation."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            use_mixup=True,
            mixup_alpha=0.2,
            hparams=hparams
        )

        x, y = synthetic_batch

        # Apply mixup
        mixed_x, mixed_y = probe.apply_mixup(x, y)

        assert mixed_x.shape == x.shape
        assert mixed_y.shape[0] == y.shape[0]
        assert mixed_y.shape[1] == 2  # One-hot encoded

        # Check that mixing occurred
        assert not torch.allclose(mixed_x, x)

    def test_gradient_accumulation(self, fake_checkpoint_path, hparams):
        """Test gradient accumulation for large effective batch size."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            gradient_accumulation_steps=4,
            hparams=hparams
        )

        assert probe.gradient_accumulation_steps == 4

        # Check that effective batch size is computed correctly
        effective_batch = probe.get_effective_batch_size(batch_size=8)
        assert effective_batch == 32  # 8 * 4

    def test_learning_rate_scheduling(self, fake_checkpoint_path, hparams):
        """Test different LR scheduling strategies."""
        # Test OneCycle
        probe_onecycle = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams={**hparams, "scheduler_type": "onecycle"}
        )

        scheduler = probe_onecycle.get_scheduler(
            optimizer=torch.optim.AdamW(probe_onecycle.parameters(), lr=1e-4),
            total_steps=1000
        )
        assert scheduler is not None

        # Test Cosine
        probe_cosine = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams={**hparams, "scheduler_type": "cosine"}
        )

        scheduler = probe_cosine.get_scheduler(
            optimizer=torch.optim.AdamW(probe_cosine.parameters(), lr=1e-4),
            total_steps=1000
        )
        assert scheduler is not None

    def test_model_checkpointing(self, fake_checkpoint_path, tmp_path, hparams):
        """Test model checkpoint saving and loading."""
        probe = EnhancedAbnormalityDetectionProbe(
            checkpoint_path=fake_checkpoint_path,
            n_channels=20,
            n_classes=2,
            hparams=hparams
        )

        # Save checkpoint
        save_path = tmp_path / "checkpoint.pt"
        probe.save_checkpoint(save_path, epoch=5, val_loss=0.3)

        assert save_path.exists()

        # Load checkpoint
        checkpoint = torch.load(save_path)
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint
        assert "state_dict" in checkpoint
        assert checkpoint["epoch"] == 5
        assert checkpoint["val_loss"] == 0.3
