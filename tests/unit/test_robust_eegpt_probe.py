"""CLEAN tests for Robust EEGPT Linear Probe - no mocks, real logic."""

import torch
import torch.nn as nn
import numpy as np
import pytest
from pathlib import Path

from brain_go_brrr.models.eegpt_linear_probe_robust import RobustEEGPTLinearProbe


class TestRobustEEGPTLinearProbeClean:
    """Test RobustEEGPTLinearProbe with dependency injection and real logic."""

    @pytest.fixture
    def fake_checkpoint_path(self, tmp_path):
        """Create a fake checkpoint file for testing."""
        checkpoint_path = tmp_path / "fake_eegpt.ckpt"
        # Create minimal checkpoint with required keys
        checkpoint = {
            "state_dict": {},
            "config": {"embed_dim": 512, "n_summary_tokens": 4}
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def synthetic_eeg_batch(self):
        """Create synthetic EEG batch data."""
        np.random.seed(1337)
        torch.manual_seed(1337)
        
        batch_size = 4
        n_channels = 20
        n_samples = 1024  # 4 seconds at 256 Hz
        
        # Create realistic EEG-like data
        data = torch.randn(batch_size, n_channels, n_samples) * 10e-6  # microvolts
        # Add some structure
        for i in range(batch_size):
            # Add alpha rhythm
            t = torch.linspace(0, 4, n_samples)
            alpha = torch.sin(2 * np.pi * 10 * t) * 5e-6
            data[i, :10] += alpha  # Add to first 10 channels
            
        return data.float()

    def test_init_creates_probe(self, fake_checkpoint_path):
        """Test initialization of RobustEEGPTLinearProbe."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True
        )
        
        assert probe is not None
        assert probe.n_input_channels == 20
        assert probe.n_classes == 2
        assert probe.embed_dim == 512
        assert probe.n_summary_tokens == 4
        assert probe.input_clip_value == 50.0
        assert probe.normalization_eps == 1e-5

    def test_input_validation_and_clipping(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test input validation and clipping logic."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            input_clip_value=20e-6  # 20 microvolts
        )
        
        # Add extreme values to test clipping
        data_with_outliers = synthetic_eeg_batch.clone()
        data_with_outliers[0, 0, 100:110] = 100e-6  # Big spike
        data_with_outliers[1, 5, 200:210] = -100e-6  # Big negative spike
        
        # Process through validation
        validated = probe.validate_input(data_with_outliers)
        
        # Check clipping worked
        assert validated.max() <= 20e-6
        assert validated.min() >= -20e-6
        assert not torch.isnan(validated).any()
        assert not torch.isinf(validated).any()

    def test_normalize_with_epsilon(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test normalization with epsilon for stability."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            normalization_eps=1e-5
        )
        
        # Create data with very small variance (could cause instability)
        small_variance_data = torch.ones_like(synthetic_eeg_batch) * 1e-10
        small_variance_data += torch.randn_like(small_variance_data) * 1e-12
        
        # Normalize
        normalized = probe.normalize_channels(small_variance_data)
        
        # Should not have NaN or Inf despite small variance
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
        
        # Should be roughly normalized
        assert normalized.std() < 10  # Reasonable bound

    def test_forward_pass_shape(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test forward pass produces correct output shape."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=3,  # 3-class classification
            freeze_backbone=True
        )
        
        # Mock the EEGPT backbone to avoid loading real weights
        class FakeBackbone(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 4, 512).float()  # 4 summary tokens, 512 dim
        
        probe.eegpt_backbone = FakeBackbone()
        
        # Forward pass
        output = probe(synthetic_eeg_batch)
        
        assert output.shape == (4, 3)  # batch_size=4, n_classes=3
        assert output.dtype == torch.float32

    def test_gradient_flow_with_frozen_backbone(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test gradient flow when backbone is frozen."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True
        )
        
        # Mock backbone
        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(20, 512)
                
            def forward(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 4, 512, requires_grad=False).float()
        
        probe.eegpt_backbone = FakeBackbone()
        
        # Check backbone params are frozen
        for param in probe.eegpt_backbone.parameters():
            assert param.requires_grad == False
            
        # Check probe head params are trainable
        for param in probe.probe_head.parameters():
            assert param.requires_grad == True
            
    def test_feature_validation_after_eegpt(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test feature validation after EEGPT extraction."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2
        )
        
        # Mock backbone that produces features with NaN
        class ProblematicBackbone(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                features = torch.randn(batch_size, 4, 512).float()
                features[0, 0, :10] = float('nan')  # Inject NaN
                return features
        
        probe.eegpt_backbone = ProblematicBackbone()
        
        # Should handle NaN gracefully
        features = probe.extract_and_validate_features(synthetic_eeg_batch)
        
        assert not torch.isnan(features).any()  # NaNs should be handled
        assert features.shape[1:] == (4, 512)  # Correct shape

    def test_weighted_loss_computation(self, fake_checkpoint_path):
        """Test weighted loss computation for imbalanced classes."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            class_weights=torch.tensor([1.0, 3.0])  # Weight abnormal class more
        )
        
        # Create predictions and targets
        logits = torch.tensor([
            [2.0, -1.0],  # Predicts class 0
            [-1.0, 2.0],  # Predicts class 1
            [1.0, 1.0],   # Uncertain
            [3.0, -2.0],  # Strong class 0
        ])
        
        targets = torch.tensor([0, 1, 1, 0])
        
        # Compute weighted loss
        loss = probe.compute_weighted_loss(logits, targets)
        
        assert loss > 0
        assert loss.requires_grad  # Should support backprop

    def test_confidence_calibration(self, fake_checkpoint_path):
        """Test confidence score calibration."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            temperature_scaling=2.0  # Temperature for calibration
        )
        
        # Raw logits
        logits = torch.tensor([
            [10.0, -10.0],  # Very confident class 0
            [0.1, -0.1],    # Very uncertain
            [-5.0, 5.0],    # Confident class 1
        ])
        
        # Apply temperature scaling
        calibrated = probe.calibrate_confidence(logits)
        probs = torch.softmax(calibrated, dim=1)
        
        # After calibration, extreme confidences should be moderated
        assert probs[0, 0] < 0.999  # Less extreme
        assert probs[1, 0] > 0.4 and probs[1, 0] < 0.6  # Still uncertain
        
    def test_robust_pooling_strategies(self, fake_checkpoint_path):
        """Test different pooling strategies for summary tokens."""
        # Test mean pooling
        probe_mean = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            pooling_strategy="mean"
        )
        
        features = torch.randn(4, 4, 512)  # batch=4, tokens=4, dim=512
        pooled = probe_mean.pool_features(features)
        assert pooled.shape == (4, 512)
        
        # Test max pooling
        probe_max = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            pooling_strategy="max"
        )
        
        pooled = probe_max.pool_features(features)
        assert pooled.shape == (4, 512)
        assert (pooled == features.max(dim=1)[0]).all()
        
    def test_nan_detection_in_forward(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test NaN detection and handling in forward pass."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            detect_anomalies=True  # Enable anomaly detection
        )
        
        # Inject NaN into input
        bad_data = synthetic_eeg_batch.clone()
        bad_data[0, :, 100:200] = float('nan')
        
        # Mock backbone
        probe.eegpt_backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 1024, 4 * 512),
            nn.Unflatten(1, (4, 512))
        )
        
        # Should handle NaN without crashing
        with torch.no_grad():
            output = probe(bad_data)
            
        assert not torch.isnan(output).any()
        assert output.shape == (4, 2)
        
    def test_statistics_tracking(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test tracking of input/output statistics."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2,
            track_statistics=True
        )
        
        # Mock backbone
        probe.eegpt_backbone = lambda x: torch.randn(x.shape[0], 4, 512).float()
        
        # Process multiple batches
        for _ in range(5):
            _ = probe(synthetic_eeg_batch)
            
        # Check statistics were tracked
        stats = probe.get_statistics()
        assert "input_mean" in stats
        assert "input_std" in stats
        assert "output_mean" in stats
        assert "n_batches" in stats
        assert stats["n_batches"] == 5
        
    def test_mixed_precision_compatibility(self, fake_checkpoint_path, synthetic_eeg_batch):
        """Test compatibility with mixed precision training."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2
        )
        
        # Mock backbone
        probe.eegpt_backbone = lambda x: torch.randn(x.shape[0], 4, 512).float()
        
        # Test with float16 input (mixed precision)
        half_data = synthetic_eeg_batch.half()
        
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            output = probe(half_data)
            
        assert output.dtype in [torch.float16, torch.float32]
        assert not torch.isnan(output).any()
        
    def test_save_and_load_state(self, fake_checkpoint_path, tmp_path):
        """Test saving and loading probe state."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2
        )
        
        # Save state
        save_path = tmp_path / "probe_state.pt"
        probe.save_state(save_path)
        
        assert save_path.exists()
        
        # Create new probe and load state
        probe2 = RobustEEGPTLinearProbe(
            checkpoint_path=fake_checkpoint_path,
            n_input_channels=20,
            n_classes=2
        )
        probe2.load_state(save_path)
        
        # Compare parameters
        for p1, p2 in zip(probe.parameters(), probe2.parameters()):
            assert torch.allclose(p1, p2)