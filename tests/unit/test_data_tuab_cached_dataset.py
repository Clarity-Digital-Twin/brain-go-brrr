"""REAL tests for TUAB cached dataset - Critical for training."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestTUABCachedDataset:
    """Test TUAB cached dataset functionality."""

    def test_cached_dataset_initialization(self):
        """Test cached dataset initialization."""
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create mock index file
            index_data = {
                "n_files": 2,
                "total_windows": 10,
                "files": {
                    "file1.edf": {
                        "cache_file": "cache_0.pt",
                        "n_windows": 5,
                        "label": 0
                    },
                    "file2.edf": {
                        "cache_file": "cache_1.pt",
                        "n_windows": 5,
                        "label": 1
                    }
                }
            }
            
            index_file = cache_dir / "index.json"
            with open(index_file, 'w') as f:
                json.dump(index_data, f)
            
            # Create dataset
            dataset = TUABCachedDataset(cache_dir=str(cache_dir))
            
            assert len(dataset) == 10
            assert dataset.n_files == 2

    def test_cached_dataset_getitem(self):
        """Test getting items from cached dataset."""
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create index
            index_data = {
                "n_files": 1,
                "total_windows": 2,
                "files": {
                    "test.edf": {
                        "cache_file": "cache_0.pt",
                        "n_windows": 2,
                        "label": 0,
                        "window_indices": [(0, 1), (1, 2)]
                    }
                }
            }
            
            index_file = cache_dir / "index.json"
            with open(index_file, 'w') as f:
                json.dump(index_data, f)
            
            # Create cache file with dummy data
            cache_data = []
            for i in range(2):
                window_data = {
                    'x': torch.randn(20, 1024),  # 20 channels, 4s at 256Hz
                    'y': torch.tensor(0)
                }
                cache_data.append(window_data)
            
            torch.save(cache_data, cache_dir / "cache_0.pt")
            
            # Create dataset
            dataset = TUABCachedDataset(cache_dir=str(cache_dir))
            
            # Get items
            x, y = dataset[0]
            assert x.shape == (20, 1024)
            assert y == 0

    def test_cached_dataset_memory_mapping(self):
        """Test memory-mapped loading."""
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create simple index
            index_data = {
                "n_files": 1,
                "total_windows": 100,
                "memory_mapped": True,
                "files": {
                    "large.edf": {
                        "cache_file": "cache_large.pt",
                        "n_windows": 100,
                        "label": 0
                    }
                }
            }
            
            with open(cache_dir / "index.json", 'w') as f:
                json.dump(index_data, f)
            
            # Create large cache file
            large_data = []
            for i in range(100):
                large_data.append({
                    'x': torch.randn(20, 1024),
                    'y': torch.tensor(0)
                })
            torch.save(large_data, cache_dir / "cache_large.pt")
            
            # Load with memory mapping
            dataset = TUABCachedDataset(
                cache_dir=str(cache_dir),
                memory_map=True
            )
            
            # Should load without loading entire file
            assert len(dataset) == 100

    def test_cached_dataset_split(self):
        """Test train/val split of cached dataset."""
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create index with multiple files
            files = {}
            for i in range(10):
                files[f"file{i}.edf"] = {
                    "cache_file": f"cache_{i}.pt",
                    "n_windows": 10,
                    "label": i % 2  # Alternating labels
                }
            
            index_data = {
                "n_files": 10,
                "total_windows": 100,
                "files": files
            }
            
            with open(cache_dir / "index.json", 'w') as f:
                json.dump(index_data, f)
            
            # Create dataset
            dataset = TUABCachedDataset(cache_dir=str(cache_dir))
            
            # Split
            train_dataset, val_dataset = dataset.split(train_ratio=0.8)
            
            # Check split
            assert len(train_dataset) + len(val_dataset) == 100
            assert abs(len(train_dataset) - 80) <= 10  # Allow some variance


class TestCacheBuilding:
    """Test cache building functionality."""

    def test_build_cache_from_raw(self):
        """Test building cache from raw EDF files."""
        from brain_go_brrr.data.tuab_cached_dataset import build_cache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            cache_dir = Path(tmpdir) / "cache"
            raw_dir.mkdir()
            cache_dir.mkdir()
            
            # Mock EDF files
            (raw_dir / "normal").mkdir()
            (raw_dir / "abnormal").mkdir()
            
            # Create dummy files
            for i in range(2):
                (raw_dir / "normal" / f"file{i}.edf").touch()
                (raw_dir / "abnormal" / f"file{i}.edf").touch()
            
            # Mock the actual processing
            with patch('brain_go_brrr.data.tuab_cached_dataset.process_edf_file') as mock_process:
                mock_process.return_value = [
                    {'x': torch.randn(20, 1024), 'y': 0}
                    for _ in range(5)
                ]
                
                # Build cache
                build_cache(
                    raw_dir=str(raw_dir),
                    cache_dir=str(cache_dir),
                    window_size=4.0,
                    window_stride=2.0
                )
                
                # Check index created
                assert (cache_dir / "index.json").exists()

    def test_cache_validation(self):
        """Test cache validation."""
        from brain_go_brrr.data.tuab_cached_dataset import validate_cache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create valid cache
            index_data = {
                "n_files": 1,
                "total_windows": 5,
                "files": {
                    "test.edf": {
                        "cache_file": "cache_0.pt",
                        "n_windows": 5,
                        "label": 0
                    }
                }
            }
            
            with open(cache_dir / "index.json", 'w') as f:
                json.dump(index_data, f)
            
            # Create cache file
            cache_data = [
                {'x': torch.randn(20, 1024), 'y': 0}
                for _ in range(5)
            ]
            torch.save(cache_data, cache_dir / "cache_0.pt")
            
            # Validate
            is_valid, message = validate_cache(str(cache_dir))
            assert is_valid
            assert "valid" in message.lower()

    def test_cache_corruption_detection(self):
        """Test detection of corrupted cache."""
        from brain_go_brrr.data.tuab_cached_dataset import validate_cache
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create index pointing to missing file
            index_data = {
                "n_files": 1,
                "total_windows": 5,
                "files": {
                    "test.edf": {
                        "cache_file": "missing.pt",  # Doesn't exist
                        "n_windows": 5,
                        "label": 0
                    }
                }
            }
            
            with open(cache_dir / "index.json", 'w') as f:
                json.dump(index_data, f)
            
            # Validate should fail
            is_valid, message = validate_cache(str(cache_dir))
            assert not is_valid
            assert "missing" in message.lower() or "not found" in message.lower()


class TestCacheDataLoader:
    """Test DataLoader integration with cached dataset."""

    def test_dataloader_batching(self):
        """Test batching with DataLoader."""
        from torch.utils.data import DataLoader
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create small dataset
            index_data = {
                "n_files": 1,
                "total_windows": 32,
                "files": {
                    "test.edf": {
                        "cache_file": "cache_0.pt",
                        "n_windows": 32,
                        "label": 0
                    }
                }
            }
            
            with open(cache_dir / "index.json", 'w') as f:
                json.dump(index_data, f)
            
            # Create cache
            cache_data = [
                {'x': torch.randn(20, 1024), 'y': torch.tensor(i % 2)}
                for i in range(32)
            ]
            torch.save(cache_data, cache_dir / "cache_0.pt")
            
            # Create dataset and loader
            dataset = TUABCachedDataset(cache_dir=str(cache_dir))
            loader = DataLoader(dataset, batch_size=8, shuffle=False)
            
            # Check batching
            batch_count = 0
            for batch_x, batch_y in loader:
                assert batch_x.shape == (8, 20, 1024)
                assert batch_y.shape == (8,)
                batch_count += 1
            
            assert batch_count == 4  # 32 / 8

    def test_dataloader_shuffling(self):
        """Test shuffling in DataLoader."""
        from torch.utils.data import DataLoader
        from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create dataset
            index_data = {
                "n_files": 1,
                "total_windows": 10,
                "files": {
                    "test.edf": {
                        "cache_file": "cache_0.pt",
                        "n_windows": 10,
                        "label": 0
                    }
                }
            }
            
            with open(cache_dir / "index.json", 'w') as f:
                json.dump(index_data, f)
            
            # Create cache with sequential values
            cache_data = [
                {'x': torch.ones(20, 1024) * i, 'y': torch.tensor(i)}
                for i in range(10)
            ]
            torch.save(cache_data, cache_dir / "cache_0.pt")
            
            # Create loaders
            dataset = TUABCachedDataset(cache_dir=str(cache_dir))
            
            # Get two epochs with shuffling
            loader1 = DataLoader(dataset, batch_size=2, shuffle=True)
            loader2 = DataLoader(dataset, batch_size=2, shuffle=True)
            
            labels1 = []
            labels2 = []
            
            for _, y in loader1:
                labels1.extend(y.tolist())
            
            for _, y in loader2:
                labels2.extend(y.tolist())
            
            # Should be different order (with high probability)
            # This might fail rarely due to random chance
            if len(labels1) == len(labels2) == 10:
                assert labels1 != labels2  # Different order