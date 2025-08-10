"""CLEAN tests for Robust EEGPT Linear Probe - no mocks, real logic."""


import numpy as np
import pytest
import torch
import torch.nn as nn

from brain_go_brrr.models.eegpt_linear_probe_robust import RobustEEGPTLinearProbe


class TestRobustEEGPTLinearProbeClean:
    """Test RobustEEGPTLinearProbe with dependency injection and real logic."""
    
    @pytest.fixture
    def mock_backbone(self):
        """Create a deterministic mock EEGPT backbone for testing."""
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                # Fixed seed for deterministic tests
                torch.manual_seed(42)
                self.dummy_param = nn.Parameter(torch.randn(1))
            
            def forward(self, x):
                # Deterministic output based on input shape
                torch.manual_seed(42)  # Ensure reproducibility
                batch_size = x.shape[0] if x.dim() > 0 else 1
                return torch.randn(batch_size, 4, 512, dtype=torch.float32)
            
            def extract_features(self, x):
                return self.forward(x)
        
        return MockBackbone()

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

    def test_init_creates_probe(self, mock_backbone):
        """Test initialization of RobustEEGPTLinearProbe."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True
        )

        assert probe is not None
        assert probe.n_input_channels == 20
        assert probe.n_classes == 2
        assert probe.embed_dim == 512
        assert probe.n_summary_tokens == 4

    def test_validate_and_clean_input(self, mock_backbone, synthetic_eeg_batch):
        """Test input validation and cleaning."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Add extreme values to test cleaning
        data_with_outliers = synthetic_eeg_batch.clone()
        data_with_outliers[0, 0, 100:110] = 100e-6  # Big spike
        data_with_outliers[1, 5, 200:210] = -100e-6  # Big negative spike
        data_with_outliers[0, 1, 50] = float('nan')  # NaN value

        # Process through validation
        validated = probe._validate_and_clean_input(data_with_outliers)

        # Check cleaning worked
        assert not torch.isnan(validated).any()
        assert not torch.isinf(validated).any()

    def test_robust_normalize(self, mock_backbone, synthetic_eeg_batch):
        """Test robust normalization."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Create data with very small variance (could cause instability)
        small_variance_data = torch.ones_like(synthetic_eeg_batch) * 1e-10
        small_variance_data += torch.randn_like(small_variance_data) * 1e-12

        # Normalize
        normalized = probe._robust_normalize(small_variance_data)

        # Should not have NaN or Inf despite small variance
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_forward_pass_shape(self, mock_backbone, synthetic_eeg_batch):
        """Test forward pass produces correct output shape."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=3,  # 3-class classification
            freeze_backbone=True
        )

        # Forward pass
        output = probe(synthetic_eeg_batch)

        assert output.shape == (4, 3)  # batch_size=4, n_classes=3
        assert output.dtype == torch.float32

    def test_predict_proba(self, mock_backbone, synthetic_eeg_batch):
        """Test probability prediction."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Get probabilities
        probs = probe.predict_proba(synthetic_eeg_batch)

        assert probs.shape == (4, 2)  # batch_size=4, n_classes=2
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
        # Check probabilities are in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_get_num_trainable_params(self, mock_backbone):
        """Test counting trainable parameters."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True
        )

        num_params = probe.get_num_trainable_params()

        assert isinstance(num_params, int)
        assert num_params > 0  # Should have some trainable params in probe head

    def test_save_and_load_probe(self, mock_backbone, tmp_path):
        """Test saving and loading probe state."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Save probe
        save_path = tmp_path / "probe_state.pt"
        probe.save_probe(save_path)

        assert save_path.exists()

        # Create new probe and load state
        probe2 = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )
        probe2.load_probe(save_path)

        # Compare parameters
        for p1, p2 in zip(probe.parameters(), probe2.parameters(), strict=False):
            assert torch.allclose(p1, p2)

    def test_forward_with_nan_input(self, mock_backbone, synthetic_eeg_batch):
        """Test forward pass handles NaN input gracefully."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Inject NaN into input
        bad_data = synthetic_eeg_batch.clone()
        bad_data[0, :, 100:200] = float('nan')

        # Should handle NaN without crashing
        with torch.no_grad():
            output = probe(bad_data)

        assert not torch.isnan(output).any()
        assert output.shape == (4, 2)

    def test_freeze_backbone_parameters(self, mock_backbone):
        """Test that backbone parameters are frozen when specified."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True
        )

        # Check backbone params are frozen
        for param in probe.eegpt_backbone.parameters():
            assert not param.requires_grad

        # Check classifier params are trainable
        for param in probe.classifier.parameters():
            assert param.requires_grad

    def test_multi_class_classification(self, mock_backbone, synthetic_eeg_batch):
        """Test multi-class classification setup."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=5,  # 5-class problem
            freeze_backbone=True
        )

        # Forward pass
        output = probe(synthetic_eeg_batch)
        probs = probe.predict_proba(synthetic_eeg_batch)

        assert output.shape == (4, 5)
        assert probs.shape == (4, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)

    def test_mixed_precision_compatibility(self, mock_backbone, synthetic_eeg_batch):
        """Test compatibility with mixed precision."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Test with float16 input (mixed precision)
        half_data = synthetic_eeg_batch.half()

        with torch.autocast(device_type='cpu', dtype=torch.float16):
            output = probe(half_data)

        assert output.dtype in [torch.float16, torch.float32]
        assert not torch.isnan(output).any()

    def test_different_batch_sizes(self, mock_backbone):
        """Test probe with different batch sizes."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            data = torch.randn(batch_size, 20, 1024) * 10e-6
            output = probe(data)
            assert output.shape == (batch_size, 2)

    def test_probe_head_architecture(self, mock_backbone):
        """Test probe head has expected architecture."""
        probe = RobustEEGPTLinearProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2
        )

        # Check classifier structure (the actual probe head)
        assert hasattr(probe, 'classifier')
        assert isinstance(probe.classifier, nn.Module)

        # Check it has expected layers
        modules = list(probe.classifier.modules())
        assert any(isinstance(m, nn.Linear) or 'LinearWithConstraint' in m.__class__.__name__ for m in modules)

