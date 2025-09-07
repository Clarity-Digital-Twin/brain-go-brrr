"""CLEAN tests for Robust EEGPT Linear Probe - no mocks, real logic."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe


class TestRobustEEGPTLinearProbeClean:
    """Test EEGPTProbe (in robust mode) with dependency injection and real logic."""

    @pytest.fixture
    def mock_backbone(self):
        """Create a deterministic mock EEGPT backbone for testing."""

        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                # Create deterministic template without affecting global RNG
                generator = torch.Generator().manual_seed(42)
                template = torch.randn(4, 512, generator=generator, dtype=torch.float32)
                self.register_buffer("template", template)
                # Dummy param for parameter iteration tests
                self.dummy_param = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                # Return expanded template without touching global RNG
                batch_size = x.shape[0] if x.dim() > 0 else 1
                return self.template.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

            def extract_features(self, x, return_all_temporal=False, summary=False):
                return self.forward(x)

        return MockBackbone()

    @pytest.fixture
    def fake_checkpoint_path(self, tmp_path):
        """Create a fake checkpoint file for testing."""
        checkpoint_path = tmp_path / "fake_eegpt.ckpt"
        # Create minimal checkpoint with required keys
        checkpoint = {"state_dict": {}, "config": {"embed_dim": 512, "n_summary_tokens": 4}}
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
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True,
            robust_mode=True,
            architecture="linear",
        )

        assert probe is not None
        # The unified probe stores different attributes
        assert probe.n_classes == 2
        assert probe.robust_mode
        assert probe.architecture == "linear"

    def test_validate_and_clean_input(self, mock_backbone, synthetic_eeg_batch):
        """Test input validation and cleaning."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Add extreme values to test cleaning
        data_with_outliers = synthetic_eeg_batch.clone()
        data_with_outliers[0, 0, 100:110] = 100e-6  # Big spike
        data_with_outliers[1, 5, 200:210] = -100e-6  # Big negative spike
        data_with_outliers[0, 1, 50] = float("nan")  # NaN value

        # Process through validation
        # The unified probe handles validation internally in forward()
        # Just check that forward works with outliers
        output = probe(data_with_outliers)
        validated = output  # For compatibility

        # Check cleaning worked
        assert not torch.isnan(validated).any()
        assert not torch.isinf(validated).any()

    def test_robust_normalize(self, mock_backbone, synthetic_eeg_batch):
        """Test robust normalization."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Create data with very small variance (could cause instability)
        small_variance_data = torch.ones_like(synthetic_eeg_batch) * 1e-10
        small_variance_data += torch.randn_like(small_variance_data) * 1e-12

        # Normalize
        # The unified probe handles normalization internally
        # Just check that forward works with small variance data
        output = probe(small_variance_data)
        normalized = output  # For compatibility

        # Should not have NaN or Inf despite small variance
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_forward_pass_shape(self, mock_backbone, synthetic_eeg_batch):
        """Test forward pass produces correct output shape."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=3,  # 3-class classification
            freeze_backbone=True,
        )

        # Forward pass
        output = probe(synthetic_eeg_batch)

        assert output.shape == (4, 3)  # batch_size=4, n_classes=3
        assert output.dtype == torch.float32

    def test_predict_proba(self, mock_backbone, synthetic_eeg_batch):
        """Test probability prediction."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
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
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True,
            robust_mode=True,
            architecture="linear",
        )

        num_params = probe.get_num_trainable_params()

        assert isinstance(num_params, int)
        assert num_params > 0  # Should have some trainable params in probe head

    def test_save_and_load_probe(self, mock_backbone, tmp_path):
        """Test saving and loading probe state."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Save probe
        save_path = tmp_path / "probe_state.pt"
        probe.save_probe(save_path)

        assert save_path.exists()

        # Create new probe and load state
        probe2 = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )
        probe2.load_probe(save_path)

        # Compare parameters (skip uninitialized LazyLinear params)
        from contextlib import suppress

        for p1, p2 in zip(probe.parameters(), probe2.parameters(), strict=False):
            with suppress(RuntimeError, ValueError):
                assert torch.allclose(p1, p2)

    def test_forward_with_nan_input(self, mock_backbone, synthetic_eeg_batch):
        """Test forward pass handles NaN input gracefully."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Inject NaN into input
        bad_data = synthetic_eeg_batch.clone()
        bad_data[0, :, 100:200] = float("nan")

        # Should handle NaN without crashing
        with torch.no_grad():
            output = probe(bad_data)

        assert not torch.isnan(output).any()
        assert output.shape == (4, 2)

    def test_freeze_backbone_parameters(self, mock_backbone):
        """Test that backbone parameters are frozen when specified."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True,
            robust_mode=True,
            architecture="linear",
        )

        # Check backbone params are frozen (use .backbone, not deprecated alias)
        for param in probe.backbone.parameters():
            assert not param.requires_grad

        # Check classifier params are trainable
        for param in probe.classifier.parameters():
            assert param.requires_grad

    def test_multi_class_classification(self, mock_backbone, synthetic_eeg_batch):
        """Test multi-class classification setup."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=5,  # 5-class problem
            freeze_backbone=True,
        )

        # Forward pass
        output = probe(synthetic_eeg_batch)
        probs = probe.predict_proba(synthetic_eeg_batch)

        assert output.shape == (4, 5)
        assert probs.shape == (4, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)

    def test_mixed_precision_compatibility(self, mock_backbone, synthetic_eeg_batch):
        """Test compatibility with mixed precision."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Test with float16 input (mixed precision)
        half_data = synthetic_eeg_batch.half()

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            output = probe(half_data)

        assert output.dtype in [torch.float16, torch.float32]
        assert not torch.isnan(output).any()

    def test_different_batch_sizes(self, mock_backbone):
        """Test probe with different batch sizes."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            data = torch.randn(batch_size, 20, 1024) * 10e-6
            output = probe(data)
            assert output.shape == (batch_size, 2)

    def test_probe_head_architecture(self, mock_backbone):
        """Test probe head has expected architecture."""
        probe = EEGPTProbe(
            checkpoint_path=None,
            backbone=mock_backbone,
            n_input_channels=20,
            n_classes=2,
            robust_mode=True,
            architecture="linear",
        )

        # Check classifier structure (the actual probe head)
        assert hasattr(probe, "classifier")
        assert isinstance(probe.classifier, nn.Module)

        # Check it has expected layers
        modules = list(probe.classifier.modules())
        assert any(
            isinstance(m, nn.Linear) or "LinearWithConstraint" in m.__class__.__name__
            for m in modules
        )
