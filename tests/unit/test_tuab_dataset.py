"""Tests for TUAB dataset implementation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest
import torch

from brain_go_brrr.infra.data.tuab_dataset import TUABDataset


class TestTUABDataset:
    """Test TUAB dataset functionality."""

    @pytest.fixture
    def mock_raw(self):
        """Create mock MNE raw object with TUAB channels."""
        raw = MagicMock()
        # TUAB uses old naming (T3, T4, T5, T6) instead of modern (T7, T8, P7, P8)
        raw.info = {
            'sfreq': 256,
            'ch_names': [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'T3', 'C3', 'Cz', 'C4', 'T4',  # Old naming
                'T5', 'P3', 'Pz', 'P4', 'T6',  # Old naming
                'O1', 'O2'
            ]
        }
        raw.get_data.return_value = np.random.randn(19, 256 * 20)  # 19 channels, 20 seconds
        raw.pick_channels = MagicMock(return_value=raw)
        raw.filter = MagicMock(return_value=raw)
        raw.resample = MagicMock(return_value=raw)
        raw.notch_filter = MagicMock(return_value=raw)
        raw.n_times = 256 * 20
        return raw

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary TUAB dataset directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "edf"
            
            # Create TUAB structure
            for split in ['train', 'eval']:
                for label in ['normal', 'abnormal']:
                    split_dir = root / split / label
                    split_dir.mkdir(parents=True)
                    
                    # Create dummy EDF files with TUAB naming
                    for i in range(2):
                        edf_file = split_dir / f"aaaa{i:04d}_s001_t000.edf"
                        edf_file.touch()
            
            yield root

    def test_dataset_initialization(self, temp_dataset_dir):
        """Test dataset can be initialized."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir.parent / 'cache',
            sampling_rate=256,
            window_duration=4.0,
            window_stride=4.0,
            normalize=False
        )
        
        assert dataset is not None
        assert dataset.split == 'train'
        assert dataset.sampling_rate == 256
        assert dataset.window_duration == 4.0
        assert dataset.window_samples == 1024  # 4s * 256Hz

    def test_channel_mapping(self, temp_dataset_dir):
        """Test TUAB old to modern channel name mapping."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train'
        )
        
        # Test channel mapping
        assert dataset.CHANNEL_MAPPING['T3'] == 'T7'
        assert dataset.CHANNEL_MAPPING['T4'] == 'T8'
        assert dataset.CHANNEL_MAPPING['T5'] == 'P7'
        assert dataset.CHANNEL_MAPPING['T6'] == 'P8'

    @patch('brain_go_brrr.infra.data.tuab_dataset.mne.io.read_raw_edf')
    def test_file_collection(self, mock_read_edf, mock_raw, temp_dataset_dir):
        """Test collecting EDF files from directory."""
        mock_read_edf.return_value = mock_raw
        
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train'
        )
        
        # Should find 4 files (2 normal, 2 abnormal)
        assert len(dataset.file_paths) == 4
        assert len(dataset.labels) == 4
        
        # Check labels are correct
        normal_count = sum(1 for l in dataset.labels if l == 0)
        abnormal_count = sum(1 for l in dataset.labels if l == 1)
        assert normal_count == 2
        assert abnormal_count == 2

    @patch('brain_go_brrr.infra.data.tuab_dataset.mne.io.read_raw_edf')
    def test_preprocessing_pipeline(self, mock_read_edf, mock_raw, temp_dataset_dir):
        """Test the preprocessing pipeline."""
        mock_read_edf.return_value = mock_raw
        
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train',
            sampling_rate=256,
            bandpass_low=0.5,
            bandpass_high=50,
            notch_freq=60
        )
        
        # Create a test file
        test_file = temp_dataset_dir / 'train' / 'normal' / 'test.edf'
        test_file.touch()
        
        # Process file
        windows, labels = dataset._process_file(test_file, label=0)
        
        # Check preprocessing was applied
        mock_raw.filter.assert_called()
        mock_raw.notch_filter.assert_called()
        
        # Check output shape
        assert isinstance(windows, list)
        assert isinstance(labels, list)

    @patch('brain_go_brrr.infra.data.tuab_dataset.torch.save')
    @patch('brain_go_brrr.infra.data.tuab_dataset.mne.io.read_raw_edf')
    def test_cache_saving(self, mock_read_edf, mock_save, mock_raw, temp_dataset_dir):
        """Test saving processed data to cache."""
        mock_read_edf.return_value = mock_raw
        
        cache_dir = temp_dataset_dir.parent / 'cache'
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=cache_dir
        )
        
        # Process and cache a file
        test_file = temp_dataset_dir / 'train' / 'normal' / 'test.edf'
        test_file.touch()
        
        windows, labels = dataset._process_file(test_file, label=0)
        
        # Should attempt to save to cache
        if windows:  # If windows were extracted
            mock_save.assert_called()

    @patch('brain_go_brrr.infra.data.tuab_dataset.torch.load')
    def test_cache_loading(self, mock_load, temp_dataset_dir):
        """Test loading from cache when available."""
        cache_dir = temp_dataset_dir.parent / 'cache'
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=cache_dir
        )
        
        # Mock cached data
        mock_load.return_value = {
            'windows': [torch.randn(19, 1024) for _ in range(3)],
            'labels': [0, 0, 0]
        }
        
        # Create cache file
        cache_file = cache_dir / 'train' / 'test_cache.pt'
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch()
        
        # Load from cache
        windows, labels = dataset._load_from_cache(cache_file)
        
        assert len(windows) == 3
        assert len(labels) == 3
        mock_load.assert_called_once()

    def test_window_extraction(self, temp_dataset_dir):
        """Test extracting windows from continuous data."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train',
            window_duration=4.0,
            window_stride=2.0  # 50% overlap
        )
        
        # Create test data (20 seconds)
        data = np.random.randn(19, 256 * 20)
        
        # Extract windows
        windows = dataset._extract_windows(data, dataset.window_samples, dataset.stride_samples)
        
        # With 20s data, 4s windows, 2s stride: should get 9 windows
        # (20 - 4) / 2 + 1 = 9
        assert len(windows) == 9
        assert windows[0].shape == (19, 1024)

    @patch('brain_go_brrr.infra.data.tuab_dataset.mne.io.read_raw_edf')
    def test_getitem(self, mock_read_edf, mock_raw, temp_dataset_dir):
        """Test dataset indexing."""
        mock_read_edf.return_value = mock_raw
        
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train'
        )
        
        # Mock that we have windows
        dataset.all_windows = [torch.randn(19, 1024) for _ in range(10)]
        dataset.all_labels = [0] * 5 + [1] * 5
        
        # Test indexing
        x, y = dataset[0]
        assert x.shape == (19, 1024)
        assert y in [0, 1]
        
        # Test last item
        x, y = dataset[9]
        assert x.shape == (19, 1024)

    def test_len(self, temp_dataset_dir):
        """Test dataset length."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train'
        )
        
        # Mock windows
        dataset.all_windows = [torch.randn(19, 1024) for _ in range(15)]
        dataset.all_labels = [0] * 15
        
        assert len(dataset) == 15

    def test_normalization(self, temp_dataset_dir):
        """Test data normalization when enabled."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train',
            normalize=True  # Enable normalization
        )
        
        # Test normalization function
        data = np.random.randn(19, 1024) * 100 + 50  # Non-normalized data
        normalized = dataset._normalize_data(data)
        
        # Check mean ~0 and std ~1 per channel
        assert np.abs(normalized.mean(axis=1)).max() < 0.1
        assert np.abs(normalized.std(axis=1) - 1.0).max() < 0.1

    def test_missing_channels_handling(self, temp_dataset_dir):
        """Test handling of files with missing channels."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='train'
        )
        
        # Create mock raw with missing channels
        raw = MagicMock()
        raw.info = {
            'sfreq': 256,
            'ch_names': ['Fp1', 'Fp2', 'C3', 'C4']  # Only 4 channels
        }
        raw.get_data.return_value = np.random.randn(4, 256 * 10)
        
        # Should handle gracefully (log warning and continue)
        with patch('brain_go_brrr.infra.data.tuab_dataset.logger') as mock_logger:
            result = dataset._validate_channels(raw)
            # Should log warning about missing channels
            assert mock_logger.warning.called

    def test_eval_split(self, temp_dataset_dir):
        """Test eval split loading."""
        dataset = TUABDataset(
            root_dir=temp_dataset_dir,
            split='eval'
        )
        
        # Should load eval files
        assert dataset.split == 'eval'
        assert len(dataset.file_paths) == 4  # 2 normal + 2 abnormal in eval

    def test_invalid_split_error(self, temp_dataset_dir):
        """Test error on invalid split."""
        with pytest.raises(ValueError, match="split must be"):
            TUABDataset(
                root_dir=temp_dataset_dir,
                split='test'  # Invalid split
            )

    def test_empty_directory_error(self):
        """Test error when no EDF files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()
            
            with pytest.raises(FileNotFoundError, match="No .edf files found"):
                TUABDataset(
                    root_dir=empty_dir,
                    split='train'
                )