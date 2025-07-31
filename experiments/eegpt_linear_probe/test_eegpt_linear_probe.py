"""Comprehensive tests for EEGPT Linear Probe implementation."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402
from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe  # noqa: E402
from brain_go_brrr.modules.constraints import LinearWithConstraint  # noqa: E402


class TestLinearWithConstraint:
    """Test the constrained linear layer."""

    def test_weight_norm_constraint(self):
        """Test that weight norms are properly constrained."""
        layer = LinearWithConstraint(10, 5, max_norm=2.0)

        # Set weights with large norms
        with torch.no_grad():
            layer.weight.data = torch.randn(5, 10) * 10

        # Forward pass applies constraint
        x = torch.randn(1, 10)
        _ = layer(x)

        # Check norms are constrained
        norms = torch.norm(layer.weight.data, p=2, dim=0)
        assert torch.all(norms <= 2.0 + 1e-6)

    def test_forward_pass(self):
        """Test forward pass through constrained layer."""
        layer = LinearWithConstraint(10, 5, max_norm=2.0)
        x = torch.randn(32, 10)

        output = layer(x)

        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()

    def test_bias_usage(self):
        """Test bias parameter configuration."""
        # With bias
        layer_with_bias = LinearWithConstraint(10, 5, bias=True)
        assert layer_with_bias.bias is not None

        # Without bias
        layer_no_bias = LinearWithConstraint(10, 5, bias=False)
        assert layer_no_bias.bias is None


@pytest.fixture
def mock_eegpt_checkpoint(tmp_path):
    """Create a mock EEGPT checkpoint."""
    checkpoint = {
        "state_dict": {
            "eegpt.patch_embed.proj.weight": torch.randn(768, 58, 64),
            "eegpt.patch_embed.proj.bias": torch.randn(768),
            "eegpt.pos_embed": torch.randn(1, 4097, 768),
            "eegpt.cls_token": torch.randn(1, 1, 768),
            # Add some transformer blocks
            "eegpt.blocks.0.norm1.weight": torch.randn(768),
            "eegpt.blocks.0.norm1.bias": torch.randn(768),
        }
    }
    checkpoint_path = tmp_path / "eegpt_mock.ckpt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


class TestEEGPTLinearProbe:
    """Test the full EEGPT Linear Probe model."""

    def test_model_initialization(self, mock_eegpt_checkpoint):
        """Test model initialization with checkpoint."""
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True,
        )

        assert model.n_classes == 2
        assert model.channel_adapter is not None
        assert model.eegpt is not None
        assert model.classifier is not None

    def test_channel_adaptation(self, mock_eegpt_checkpoint):
        """Test channel adapter converts 20 to 58 channels."""
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=2
        )

        # Test with 20-channel input
        x = torch.randn(4, 20, 2048)  # [batch, channels, time]
        adapted = model.channel_adapter(x)

        assert adapted.shape == (4, 58, 2048)

    def test_forward_pass(self, mock_eegpt_checkpoint):
        """Test complete forward pass."""
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=2
        )

        # Input: [batch, channels, time]
        x = torch.randn(4, 20, 2048)

        # Forward pass
        output = model(x)

        # Check output
        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_freeze_backbone(self, mock_eegpt_checkpoint):
        """Test that backbone is properly frozen."""
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint,
            n_input_channels=20,
            n_classes=2,
            freeze_backbone=True,
        )

        # Check EEGPT parameters are frozen
        for name, param in model.named_parameters():
            if "eegpt" in name:
                assert not param.requires_grad
            elif "channel_adapter" in name or "classifier" in name:
                assert param.requires_grad

    def test_feature_extraction(self, mock_eegpt_checkpoint):
        """Test feature extraction from EEGPT."""
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=2
        )

        x = torch.randn(2, 20, 2048)

        # Get features through forward pass
        with torch.no_grad():
            output = model(x)

        # Output should be [batch, n_classes]
        assert output.ndim == 2
        assert output.shape[0] == 2
        assert output.shape[1] == 2

    def test_different_num_classes(self, mock_eegpt_checkpoint):
        """Test model with different numbers of output classes."""
        for num_classes in [2, 3, 5, 10]:
            model = EEGPTLinearProbe(
                checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=num_classes
            )

            x = torch.randn(4, 20, 2048)
            output = model(x)

            assert output.shape == (4, num_classes)


class TestTUABIntegration:
    """Test integration with TUAB dataset."""

    def test_channel_mapping(self):
        """Test that channel mapping is consistent."""
        # Check that STANDARD_CHANNELS uses modern naming
        assert "T7" in TUABDataset.STANDARD_CHANNELS
        assert "T8" in TUABDataset.STANDARD_CHANNELS
        assert "P7" in TUABDataset.STANDARD_CHANNELS
        assert "P8" in TUABDataset.STANDARD_CHANNELS

        # Old names should not be in standard channels
        assert "T3" not in TUABDataset.STANDARD_CHANNELS
        assert "T4" not in TUABDataset.STANDARD_CHANNELS
        assert "T5" not in TUABDataset.STANDARD_CHANNELS
        assert "T6" not in TUABDataset.STANDARD_CHANNELS

    def test_channel_count(self):
        """Test that we have exactly 20 channels."""
        assert len(TUABDataset.STANDARD_CHANNELS) == 20

    @pytest.mark.parametrize(
        "old_name,new_name",
        [
            ("T3", "T7"),
            ("T4", "T8"),
            ("T5", "P7"),
            ("T6", "P8"),
        ],
    )
    def test_channel_mapping_correctness(self, old_name, new_name):
        """Test that channel mappings are correct."""
        assert TUABDataset.CHANNEL_MAPPING.get(old_name) == new_name


class TestModelCheckpoint:
    """Test model checkpoint saving and loading."""

    def test_save_load_checkpoint(self, mock_eegpt_checkpoint, tmp_path):
        """Test saving and loading model checkpoints."""
        # Create and train model briefly
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=2
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": 2,
                "config": {"freeze_backbone": True},
            },
            checkpoint_path,
        )

        # Load checkpoint
        new_model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=2
        )

        checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify weights are the same
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters(), strict=False
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


class TestEndToEnd:
    """End-to-end tests with realistic scenarios."""

    @patch("mne.io.read_raw_edf")
    def test_realistic_prediction(self, mock_read_edf, mock_eegpt_checkpoint):
        """Test realistic prediction scenario."""
        # Mock EDF file reading
        mock_raw = Mock()
        mock_raw.get_data.return_value = np.random.randn(20, 2048).astype(np.float32)
        mock_raw.ch_names = TUABDataset.STANDARD_CHANNELS
        mock_raw.info = {"sfreq": 256}
        mock_read_edf.return_value = mock_raw

        # Create model
        model = EEGPTLinearProbe(
            checkpoint_path=mock_eegpt_checkpoint, n_input_channels=20, n_classes=2
        )
        model.eval()

        # Prepare input
        eeg_data = torch.from_numpy(mock_raw.get_data()).unsqueeze(0)  # [1, 20, 2048]

        # Predict
        with torch.no_grad():
            output = model(eeg_data)
            probs = torch.softmax(output, dim=1)

        # Check predictions
        assert probs.shape == (1, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1))
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
