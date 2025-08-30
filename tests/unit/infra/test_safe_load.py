"""Tests for safe torch load wrapper."""

import pickle
from pathlib import Path

import pytest
import torch

from brain_go_brrr.infra.safe_load import safe_load


class TestSafeLoad:
    """Test safe torch load functionality."""

    def test_safe_load_roundtrip(self, tmp_path):
        """Test saving and loading with safe_load."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Create test data
        test_data = {
            "epoch": 42,
            "model_state": {"weight": [1.0, 2.0, 3.0]},
            "optimizer": "adam",
            "loss": 0.123,
        }

        # Save with regular torch
        torch.save(test_data, checkpoint_path)

        # Load with safe wrapper
        loaded = safe_load(checkpoint_path)

        # Verify all data preserved
        assert loaded["epoch"] == 42
        assert loaded["model_state"]["weight"] == [1.0, 2.0, 3.0]
        assert loaded["optimizer"] == "adam"
        assert abs(loaded["loss"] - 0.123) < 1e-6

    def test_safe_load_with_string_path(self, tmp_path):
        """Test safe_load accepts string paths."""
        checkpoint_path = tmp_path / "model.ckpt"
        torch.save({"version": "1.0"}, checkpoint_path)

        # Pass as string
        loaded = safe_load(str(checkpoint_path))
        assert loaded["version"] == "1.0"

    def test_safe_load_nonexistent_file(self):
        """Test safe_load with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            safe_load(Path("/nonexistent/file.pt"))

    def test_safe_load_corrupt_file(self, tmp_path):
        """Test safe_load with corrupt file."""
        bad_file = tmp_path / "corrupt.pt"
        bad_file.write_bytes(b"not a valid torch file")

        with pytest.raises((RuntimeError, pickle.UnpicklingError)):
            safe_load(bad_file)

    def test_safe_load_empty_checkpoint(self, tmp_path):
        """Test loading empty dict checkpoint."""
        checkpoint_path = tmp_path / "empty.pt"
        torch.save({}, checkpoint_path)

        loaded = safe_load(checkpoint_path)
        assert loaded == {}

    def test_safe_load_with_tensors(self, tmp_path):
        """Test loading checkpoint with tensors."""
        checkpoint_path = tmp_path / "tensors.pt"

        test_data = {
            "weights": torch.randn(10, 5),
            "bias": torch.zeros(5),
            "mask": torch.ones(10, dtype=torch.bool),
        }

        torch.save(test_data, checkpoint_path)
        loaded = safe_load(checkpoint_path)

        assert torch.allclose(loaded["weights"], test_data["weights"])
        assert torch.allclose(loaded["bias"], test_data["bias"])
        assert torch.equal(loaded["mask"], test_data["mask"])
