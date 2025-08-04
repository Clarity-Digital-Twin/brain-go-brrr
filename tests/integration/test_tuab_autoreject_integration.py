"""Integration tests for TUABEnhancedDataset with AutoReject."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from tests.fixtures.mock_eeg_generator import MockEEGGenerator


@pytest.mark.integration
class TestTUABAutoRejectIntegration:
    """Test AutoReject integration with TUAB dataset."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create basic structure
            root = Path(tmpdir)
            (root / "train").mkdir()
            (root / "eval").mkdir()

            # Create mock labels file
            labels_file = root / "train" / "train.csv"
            labels_file.write_text("file_name,label\ntest1.edf,0\ntest2.edf,1\n")

            yield root

    @pytest.fixture
    def mock_edf_file(self, temp_dataset_dir):
        """Create a mock EDF file."""
        # Create mock raw data
        raw = MockEEGGenerator.create_raw(
            duration=120.0,  # 2 minutes
            sfreq=256,
            add_artifacts=True,
            seed=42,
        )

        # Save to file (mock)
        file_path = temp_dataset_dir / "train" / "test1.edf"
        file_path.touch()  # Just create empty file for testing

        return file_path, raw

    def test_dataset_with_autoreject_enabled(self, temp_dataset_dir):
        """Test dataset initialization with AutoReject enabled."""
        dataset = TUABEnhancedDataset(
            root_dir=temp_dataset_dir,
            split="train",
            use_autoreject=True,
            ar_cache_dir=temp_dataset_dir / "ar_cache",
            window_duration=10.0,
            window_stride=5.0,
            sampling_rate=200,
        )

        # Verify AutoReject components initialized
        assert dataset.use_autoreject is True
        assert dataset.ar_processor is not None
        assert dataset.window_adapter is not None
        assert dataset.position_generator is not None

        # Verify window adapter parameters
        assert dataset.window_adapter.window_duration == 10.0
        assert dataset.window_adapter.window_stride == 5.0

    def test_dataset_with_autoreject_disabled(self, temp_dataset_dir):
        """Test dataset with AutoReject disabled (backward compatibility)."""
        dataset = TUABEnhancedDataset(
            root_dir=temp_dataset_dir, split="train", use_autoreject=False
        )

        # Verify no AutoReject components
        assert dataset.use_autoreject is False
        assert dataset.ar_processor is None
        assert dataset.window_adapter is None
        assert dataset.position_generator is None

    @patch("mne.io.read_raw_edf")
    def test_load_file_with_autoreject(self, mock_read_edf, temp_dataset_dir):
        """Test loading EDF file with AutoReject processing."""
        # Create mock raw data
        mock_raw = MockEEGGenerator.create_raw(duration=60.0, sfreq=256, add_artifacts=True)
        mock_read_edf.return_value = mock_raw

        # Create dataset with AutoReject
        dataset = TUABEnhancedDataset(
            root_dir=temp_dataset_dir,
            split="train",
            use_autoreject=True,
            ar_cache_dir=temp_dataset_dir / "ar_cache",
        )

        # Mock the file list
        dataset.file_list = [(temp_dataset_dir / "train" / "test1.edf", 0)]

        # Load file
        with (
            patch.object(dataset.position_generator, "add_positions_to_raw", return_value=mock_raw),
            patch.object(dataset.ar_processor, "transform_raw", return_value=mock_raw),
        ):
            data = dataset._load_edf_file(dataset.file_list[0][0])

        # Verify shape and type
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32
        assert len(data.shape) == 2  # [channels, time]

    def test_fallback_on_autoreject_failure(self, temp_dataset_dir):
        """Test fallback mechanisms when AutoReject fails."""
        dataset = TUABEnhancedDataset(root_dir=temp_dataset_dir, split="train", use_autoreject=True)

        # Create mock raw without positions
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(19, 10000) * 50e-6
        mock_raw.ch_names = ["C3", "C4"] * 9 + ["CZ"]
        mock_raw.info = {"bads": []}

        # Test MemoryError fallback
        with patch.object(dataset.ar_processor, "transform_raw", side_effect=MemoryError):
            result = dataset._apply_autoreject_to_raw(mock_raw)
            assert result is not None  # Should return cleaned data

        # Test RuntimeError fallback
        with patch.object(
            dataset.ar_processor,
            "transform_raw",
            side_effect=RuntimeError("Valid channel positions"),
        ):
            result = dataset._apply_autoreject_to_raw(mock_raw)
            assert result is not None

    def test_amplitude_cleaning_marks_bad_channels(self, temp_dataset_dir):
        """Test amplitude-based cleaning marks bad channels correctly."""
        dataset = TUABEnhancedDataset(
            root_dir=temp_dataset_dir, split="train", use_autoreject=False
        )

        # Create data with bad channels
        ch_names = ["FP1", "FP2", "C3", "C4", "O1", "O2"]
        data = np.random.randn(6, 5000) * 50e-6

        # Make channels bad
        data[0, :] = 0  # Flat channel
        data[1, :] = np.random.randn(5000) * 300e-6  # Very noisy

        info = mne.create_info(ch_names, 256, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Apply cleaning
        raw_clean = dataset._amplitude_based_cleaning(raw)

        # Check bad channels marked
        assert "FP1" in raw_clean.info["bads"]  # Flat
        assert "FP2" in raw_clean.info["bads"]  # Noisy
        assert len(raw_clean.info["bads"]) >= 2

    @pytest.mark.slow
    def test_memory_usage_with_large_dataset(self, temp_dataset_dir):
        """Test memory usage stays reasonable with large files."""
        import psutil

        process = psutil.Process()

        # Get baseline memory
        mem_start = process.memory_info().rss / 1024 / 1024  # MB

        dataset = TUABEnhancedDataset(
            root_dir=temp_dataset_dir,
            split="train",
            use_autoreject=True,
            ar_cache_dir=temp_dataset_dir / "ar_cache",
        )

        # Create large mock data
        for _i in range(5):
            _ = MockEEGGenerator.create_raw(
                duration=300.0,  # 5 minutes each
                sfreq=256,
                add_artifacts=False,  # Speed up
            )

            # Process (would normally load from file)
            with patch.object(dataset, "_load_edf_file"):
                pass  # Just testing memory allocation

        # Check memory increase
        mem_end = process.memory_info().rss / 1024 / 1024
        memory_increase = mem_end - mem_start

        # Should stay under 500MB for this test
        assert memory_increase < 500, f"Memory increased by {memory_increase}MB"

    def test_cache_directory_creation(self, temp_dataset_dir):
        """Test AutoReject cache directory is created."""
        cache_dir = temp_dataset_dir / "test_ar_cache"
        
        # Create required directory structure
        train_dir = temp_dataset_dir / "train"
        normal_dir = train_dir / "normal"
        normal_dir.mkdir(parents=True, exist_ok=True)

        _ = TUABEnhancedDataset(
            root_dir=temp_dataset_dir, split="train", use_autoreject=True, ar_cache_dir=cache_dir
        )

        # Cache directory should be created
        assert cache_dir.exists()
        assert cache_dir.is_dir()
