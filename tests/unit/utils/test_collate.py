"""Tests for TUAB and TUEV collate functions."""

import pytest
import torch

from brain_go_brrr.utils.collate_tuab import collate_tuab_batch
from brain_go_brrr.utils.collate_tuev import collate_tuev_batch


class TestTUABCollate:
    """Test TUAB collate function."""

    def test_collate_19_channels(self):
        """Test collating batch with correct 19 channels."""
        # Create batch with 19 channels, 1024 samples
        batch = [
            (torch.randn(19, 1024), 0),
            (torch.randn(19, 1024), 1),
            (torch.randn(19, 1024), 0),
        ]

        data, labels = collate_tuab_batch(batch)

        assert data.shape == (3, 19, 1024)
        assert labels.shape == (3,)
        assert labels.dtype == torch.float32
        assert torch.all(labels == torch.tensor([0.0, 1.0, 0.0]))

    def test_collate_20_channels_rejected(self):
        """Test that 20-channel windows are strictly rejected (workaround removed)."""
        # 20 channel sample (would have been accepted with old workaround)
        batch = [
            (torch.randn(20, 1024), 1),  # Should be rejected
        ]

        # After workaround removal, 20 channels should raise error
        with pytest.raises(RuntimeError, match="Expected exactly 19 channels, got 20"):
            collate_tuab_batch(batch)

    def test_collate_tensor_labels(self):
        """Test collating when labels are already tensors."""
        batch = [
            (torch.randn(19, 1024), torch.tensor(0.0)),
            (torch.randn(19, 1024), torch.tensor(1.0)),
        ]

        data, labels = collate_tuab_batch(batch)

        assert data.shape == (2, 19, 1024)
        assert labels.shape == (2,)
        assert labels.dtype == torch.float32

    def test_collate_invalid_channels_raises(self):
        """Test that invalid channel count raises error."""
        batch = [
            (torch.randn(18, 1024), 0),  # Wrong channel count
        ]

        with pytest.raises(RuntimeError, match="Unexpected channel count 18"):
            collate_tuab_batch(batch)

    def test_19_channel_invariant_strict(self):
        """Test TUAB must ONLY yield 19 channels - no 20-channel contamination.
        
        This test enforces the invariant discovered during cache audit:
        - 100/100 sampled cache files have exactly 19 channels
        - 0% have 20 channels (Fz contamination)
        - The workaround dropping Fz is obsolete
        
        This test will PASS with current workaround but documents expected behavior.
        After removing workaround, it ensures we reject 20-channel inputs.
        """
        # Test 1: Pure 19-channel batch (expected case)
        batch_19 = [
            (torch.randn(19, 1024), 0),
            (torch.randn(19, 1024), 1),
            (torch.randn(19, 1024), 0),
        ]
        
        data, labels = collate_tuab_batch(batch_19)
        assert data.shape == (3, 19, 1024), "Must yield exactly 19 channels"
        assert labels.shape == (3,)
        
        # Test 2: Strict rejection of 20 channels (workaround removed)
        batch_20 = [
            (torch.randn(20, 1024), 1),  # Must be rejected
        ]
        
        # After workaround removal: 20 channels must raise error
        with pytest.raises(RuntimeError, match="Expected exactly 19 channels, got 20"):
            collate_tuab_batch(batch_20)
        
        # Test 3: Must reject other channel counts
        for wrong_channels in [17, 18, 21, 25]:
            batch_wrong = [(torch.randn(wrong_channels, 1024), 0)]
            with pytest.raises(RuntimeError, match=f"channel count {wrong_channels}"):
                collate_tuab_batch(batch_wrong)


class TestTUEVCollate:
    """Test TUEV collate function."""

    def test_collate_20_channels(self):
        """Test collating batch with correct 20 channels."""
        # Create batch with 20 channels, 1024 samples
        batch = [
            (torch.randn(20, 1024), 0),
            (torch.randn(20, 1024), 2),
            (torch.randn(20, 1024), 5),
        ]

        data, labels = collate_tuev_batch(batch)

        assert data.shape == (3, 20, 1024)
        assert labels.shape == (3,)
        assert labels.dtype == torch.long  # TUEV uses long for multi-class
        assert torch.all(labels == torch.tensor([0, 2, 5]))

    def test_collate_tensor_labels(self):
        """Test collating when labels are already tensors."""
        batch = [
            (torch.randn(20, 1024), torch.tensor(1)),
            (torch.randn(20, 1024), torch.tensor(3)),
        ]

        data, labels = collate_tuev_batch(batch)

        assert data.shape == (2, 20, 1024)
        assert labels.shape == (2,)
        assert labels.dtype == torch.long

    def test_collate_invalid_channels_raises(self):
        """Test that invalid channel count raises error."""
        batch = [
            (torch.randn(19, 1024), 0),  # Wrong channel count (should be 20)
        ]

        with pytest.raises(RuntimeError, match="CHANNEL COUNT ERROR.*Got 19 channels"):
            collate_tuev_batch(batch)

    def test_collate_preserves_label_range(self):
        """Test that all 6 TUEV classes are preserved."""
        # Test all 6 classes (0-5)
        batch = [(torch.randn(20, 1024), i) for i in range(6)]

        data, labels = collate_tuev_batch(batch)

        assert data.shape == (6, 20, 1024)
        assert labels.shape == (6,)
        assert torch.all(labels == torch.arange(6))
