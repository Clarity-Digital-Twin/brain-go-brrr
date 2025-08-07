"""Tests for data.tuab_dataset - CLEAN, NO MOCKING OF CORE LOGIC."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from brain_go_brrr.data.tuab_dataset import TUABDataset


class TestTUABDataset:
    """Test TUAB dataset class."""

    @patch('brain_go_brrr.data.tuab_dataset.Path.glob')
    def test_dataset_initialization(self, mock_glob):
        """Test dataset initialization."""
        # Mock file discovery
        mock_glob.return_value = [
            Path('/data/normal/file1.npy'),
            Path('/data/abnormal/file2.npy'),
        ]

        dataset = TUABDataset(
            root_dir='/data',
            window_size=4.0,
            window_overlap=0.5,
            sampling_rate=256
        )

        assert dataset.window_size == 4.0
        assert dataset.window_overlap == 0.5
        assert dataset.sampling_rate == 256

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data files
            normal_dir = Path(tmpdir) / 'normal'
            normal_dir.mkdir()

            # Save dummy EEG data
            dummy_data = np.random.randn(20, 2560).astype(np.float32)  # 10 seconds
            np.save(normal_dir / 'test.npy', dummy_data)

            # Create dataset
            dataset = TUABDataset(
                root_dir=tmpdir,
                window_size=4.0,
                window_overlap=0.0,
                sampling_rate=256
            )

            # Mock the file list
            dataset.files = [normal_dir / 'test.npy']
            dataset.labels = [0]  # Normal

            # Get item
            with patch.object(dataset, '_load_and_window') as mock_load:
                mock_load.return_value = [dummy_data[:, :1024]]  # Return one window

                data, label = dataset[0]

                assert data.shape == (20, 1024)  # 4 seconds at 256 Hz
                assert label == 0

    def test_dataset_length(self):
        """Test dataset length calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TUABDataset(root_dir=tmpdir)

            # Mock file list
            dataset.files = [Path(f'file{i}.npy') for i in range(10)]
            dataset.window_counts = [5] * 10  # 5 windows per file

            # Total windows = 10 files * 5 windows = 50
            assert len(dataset) == 50

    def test_dataset_train_test_split(self):
        """Test splitting dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TUABDataset(root_dir=tmpdir)

            # Mock some files
            dataset.files = [Path(f'file{i}.npy') for i in range(100)]
            dataset.labels = [i % 2 for i in range(100)]  # Alternating labels

            train_dataset, val_dataset = dataset.split(train_ratio=0.8)

            # Check split ratio
            assert len(train_dataset.files) == 80
            assert len(val_dataset.files) == 20


