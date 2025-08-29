"""Tests for TUEV dataset implementation."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from brain_go_brrr.infra.data.tuev_dataset import TUEVDataset


class TestTUEVDataset:
    """Test TUEV dataset functionality."""

    @pytest.fixture
    def mock_raw(self):
        """Create mock MNE raw object."""
        raw = MagicMock()
        raw.get_data.return_value = np.random.randn(20, 256 * 10)  # 20 channels, 10 seconds
        raw.info = {
            'sfreq': 256,
            'ch_names': [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'T3', 'C3', 'Cz', 'C4', 'T4',
                'T5', 'P3', 'Pz', 'P4', 'T6',
                'O1', 'O2', 'EKG'  # TUEV has Fz, no Fpz
            ]
        }
        return raw

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create TUEV structure
            for split in ['train', 'eval']:
                for label in ['normal', 'abnormal']:
                    split_dir = root / split / label
                    split_dir.mkdir(parents=True)
                    
                    # Create dummy EDF files
                    for i in range(2):
                        edf_file = split_dir / f"test_{i:03d}.edf"
                        edf_file.touch()
            
            yield root

    def test_dataset_initialization(self, temp_dataset_dir):
        """Test dataset can be initialized."""
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir / 'cache',
            sampling_rate=256,
            window_duration=4.0,
            window_stride=4.0,
            normalize=False
        )
        
        assert dataset is not None
        assert dataset.split == 'train'
        assert dataset.sampling_rate == 256
        assert dataset.window_duration == 4.0

    def test_dataset_channel_validation(self, temp_dataset_dir):
        """Test TUEV channel requirements."""
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir / 'cache'
        )
        
        # TUEV should have Fz, not Fpz
        assert 'Fz' in dataset.EXPECTED_CHANNELS
        assert 'Fpz' not in dataset.EXPECTED_CHANNELS
        assert len(dataset.EXPECTED_CHANNELS) == 19  # Standard 10-20 without Fpz

    @patch('brain_go_brrr.infra.data.tuev_dataset.mne.io.read_raw_edf')
    def test_load_and_preprocess(self, mock_read_edf, mock_raw, temp_dataset_dir):
        """Test loading and preprocessing single file."""
        mock_read_edf.return_value = mock_raw
        
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir / 'cache'
        )
        
        # Create a dummy file path
        dummy_file = temp_dataset_dir / 'train' / 'normal' / 'test.edf'
        dummy_file.touch()
        
        # Test loading
        x, y = dataset._load_and_preprocess_file(dummy_file, label=0)
        
        # Should return preprocessed data
        assert x is not None
        assert isinstance(x, np.ndarray)
        assert y == 0

    @patch('brain_go_brrr.infra.data.tuev_dataset.mne.io.read_raw_edf')
    def test_window_extraction(self, mock_read_edf, mock_raw, temp_dataset_dir):
        """Test window extraction from continuous data."""
        mock_read_edf.return_value = mock_raw
        
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir / 'cache',
            window_duration=4.0,
            window_stride=4.0
        )
        
        # Test window extraction
        data = np.random.randn(19, 256 * 12)  # 12 seconds of data
        windows = dataset._extract_windows(data)
        
        # Should get 3 windows (12s / 4s = 3)
        assert windows.shape[0] == 3
        assert windows.shape[1] == 19  # channels
        assert windows.shape[2] == 1024  # 4s * 256Hz

    def test_label_extraction(self, temp_dataset_dir):
        """Test label extraction from file path."""
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir / 'cache'
        )
        
        # Test normal label
        normal_path = temp_dataset_dir / 'train' / 'normal' / 'test.edf'
        assert dataset._get_label_from_path(normal_path) == 0
        
        # Test abnormal label
        abnormal_path = temp_dataset_dir / 'train' / 'abnormal' / 'test.edf'
        assert dataset._get_label_from_path(abnormal_path) == 1

    def test_cache_functionality(self, temp_dataset_dir):
        """Test caching mechanism."""
        cache_dir = temp_dataset_dir / 'cache'
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=cache_dir
        )
        
        # Cache directory should be created
        assert cache_dir.exists()
        
        # Test cache file naming
        test_file = temp_dataset_dir / 'train' / 'normal' / 'test.edf'
        cache_file = dataset._get_cache_path(test_file)
        assert cache_file.parent == cache_dir / 'train'
        assert cache_file.suffix == '.pt'

    @patch('brain_go_brrr.infra.data.tuev_dataset.torch.load')
    @patch('brain_go_brrr.infra.data.tuev_dataset.torch.save')
    def test_cache_loading(self, mock_save, mock_load, temp_dataset_dir):
        """Test loading from cache."""
        cache_dir = temp_dataset_dir / 'cache'
        dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=cache_dir
        )
        
        # Mock cached data
        mock_load.return_value = {
            'x': np.random.randn(3, 19, 1024),
            'y': 0
        }
        
        # Create cache file
        test_file = temp_dataset_dir / 'train' / 'normal' / 'test.edf'
        cache_file = dataset._get_cache_path(test_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch()
        
        # Should load from cache
        data = dataset._load_cached_or_process(test_file, 0)
        assert data is not None
        mock_load.assert_called_once()

    def test_dataset_splits(self, temp_dataset_dir):
        """Test train/eval split handling."""
        # Test train split
        train_dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='train',
            cache_dir=temp_dataset_dir / 'cache'
        )
        assert train_dataset.split == 'train'
        
        # Test eval split
        eval_dataset = TUEVDataset(
            root_dir=temp_dataset_dir,
            split='eval',
            cache_dir=temp_dataset_dir / 'cache'
        )
        assert eval_dataset.split == 'eval'
        
        # Test invalid split
        with pytest.raises(ValueError):
            TUEVDataset(
                root_dir=temp_dataset_dir,
                split='test',  # Invalid
                cache_dir=temp_dataset_dir / 'cache'
            )

    def test_empty_directory_handling(self, temp_dataset_dir):
        """Test handling of empty directories."""
        # Create empty directory
        empty_dir = temp_dataset_dir / 'empty'
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No EDF files found"):
            TUEVDataset(
                root_dir=empty_dir,
                split='train',
                cache_dir=temp_dataset_dir / 'cache'
            )