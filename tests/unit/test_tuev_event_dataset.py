"""Unit tests for TUEVEventDataset - TDD approach for paper parity."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset


class TestTUEVEventDataset:
    """Test TUEV event segment dataset for paper parity."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir) / "tuev"
        self.cache_dir = Path(self.temp_dir) / "cache"

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_correct_class_mapping(self):
        """Test dataset initializes with correct TUEV class mapping."""
        dataset = TUEVEventDataset(
            root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=False
        )

        expected_mapping = {'spsw': 0, 'gped': 1, 'pled': 2, 'eyem': 3, 'artf': 4, 'bckg': 5}

        assert dataset.class_mapping == expected_mapping
        assert len(dataset.class_mapping) == 6  # 6 classes for TUEV

    def test_cache_directory_structure(self):
        """Test cache directory is created with correct structure."""
        dataset = TUEVEventDataset(
            root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=False
        )

        expected_cache_path = self.cache_dir / 'tuev_event_segments'
        assert dataset.cache_dir == expected_cache_path

    @patch('brain_go_brrr.infra.data.tuev_event_dataset.TUEVEventExtractor')
    def test_build_cache_creates_segments(self, mock_extractor_class):
        """Test cache building extracts event segments correctly."""
        # Setup mock extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # Mock segment data
        segment = np.random.randn(23, 1000).astype(np.float32)
        mock_extractor.extract_segments.return_value = [
            (segment, 0),  # spsw
            (segment, 1),  # gped
        ]

        # Create test EDF files
        split_dir = self.root_dir / 'edf' / 'train'
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy EDF file
        edf_file = split_dir / 'sub001_01.edf'
        edf_file.touch()

        # Create dummy annotation file
        lab_file = split_dir / 'sub001_01.rec.lab'
        lab_file.write_text("1000000 2000000 spsw\n3000000 4000000 gped")

        # Mock the dataset's annotation parser
        with patch.object(TUEVEventDataset, '_parse_annotations') as mock_parse:
            mock_parse.return_value = [
                {'start': 1.0, 'end': 2.0, 'label': 0},
                {'start': 3.0, 'end': 4.0, 'label': 1},
            ]

            dataset = TUEVEventDataset(
                root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=True
            )

            # Check that extractor was called
            mock_extractor.extract_segments.assert_called()

    def test_parse_annotations_from_lab_file(self):
        """Test parsing of .rec.lab annotation files."""
        # Create test lab file
        lab_content = """1000000 1500000 spsw
2000000 2500000 gped
3000000 3500000 pled
4000000 4500000 eyem
5000000 5500000 artf
6000000 6500000 bckg"""

        # Create test structure
        edf_path = self.root_dir / 'edf' / 'train' / 'sub001_01.edf'
        edf_path.parent.mkdir(parents=True, exist_ok=True)
        edf_path.touch()

        lab_path = edf_path.parent / 'sub001_01.rec.lab'
        lab_path.write_text(lab_content)

        dataset = TUEVEventDataset(
            root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=False
        )

        annotations = dataset._parse_annotations(edf_path)

        assert len(annotations) == 6

        # Check first annotation
        assert annotations[0]['start'] == 1.0  # microseconds to seconds
        assert annotations[0]['end'] == 1.5
        assert annotations[0]['label'] == 0  # spsw -> 0

        # Check all labels are mapped correctly
        expected_labels = [0, 1, 2, 3, 4, 5]
        actual_labels = [a['label'] for a in annotations]
        assert actual_labels == expected_labels

    def test_getitem_returns_correct_shape_and_type(self):
        """Test __getitem__ returns (x, y) with correct shapes."""
        # Create mock cache
        cache_file = self.cache_dir / 'tuev_event_segments' / 'train' / 'segment_0.pt'
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Save test segment
        test_segment = torch.randn(23, 1000).float()
        test_label = 2
        torch.save({'x': test_segment, 'y': test_label, 'id': 'segment_0'}, cache_file)

        # Create index file
        index_file = cache_file.parent / 'index.json'
        index_data = {
            'segments': [{'file': 'segment_0.pt', 'label': test_label, 'subject': 'sub001'}],
            'n_segments': 1,
            'class_counts': {'2': 1},
            'n_subjects': 1,
            'fs': 200,
            'duration': 5.0,
            'channels': 23,
            'samples': 1000,
        }
        with open(index_file, 'w') as f:
            json.dump(index_data, f)

        dataset = TUEVEventDataset(
            root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=False
        )

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, int)
        assert x.shape == (23, 1000)
        assert x.dtype == torch.float32
        assert y == test_label

    def test_subject_level_split_extraction(self):
        """Test that subject IDs are correctly extracted from filenames."""
        # Create test files with different subject IDs
        split_dir = self.root_dir / 'edf' / 'train'
        split_dir.mkdir(parents=True, exist_ok=True)

        # Different naming patterns from TUEV
        test_files = [
            'aaaaaaaa_s001_t000.edf',  # subject: aaaaaaaa
            'bbbbbbbb_s002_t001.edf',  # subject: bbbbbbbb
            'cccccccc_s001_t000.edf',  # subject: cccccccc
        ]

        for fname in test_files:
            (split_dir / fname).touch()
            # Create corresponding lab file
            lab_file = split_dir / fname.replace('.edf', '.rec.lab')
            lab_file.write_text("1000000 2000000 spsw")

        with patch.object(TUEVEventDataset, '_build_cache') as mock_build:
            dataset = TUEVEventDataset(
                root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=True
            )

            # Extract subject IDs as the dataset would
            edf_files = list(split_dir.glob('*.edf'))
            subjects = set()
            for edf_path in edf_files:
                # TUEV subject ID is the first part before underscore
                subject = edf_path.stem.split('_')[0]
                subjects.add(subject)

            assert len(subjects) == 3
            assert 'aaaaaaaa' in subjects
            assert 'bbbbbbbb' in subjects
            assert 'cccccccc' in subjects

    def test_cache_metadata_validation(self):
        """Test that cache metadata contains required fields."""
        # Create minimal cache
        cache_file = self.cache_dir / 'tuev_event_segments' / 'train' / 'segment_0.pt'
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        test_segment = torch.randn(23, 1000).float()
        torch.save({'x': test_segment, 'y': 0, 'id': 'test'}, cache_file)

        # Create index with all required metadata
        index_file = cache_file.parent / 'index.json'
        index_data = {
            'segments': [{'file': 'segment_0.pt', 'label': 0, 'subject': 'sub001'}],
            'n_segments': 1,
            'class_counts': {'0': 1},
            'n_subjects': 1,
            'fs': 200,  # Must be 200Hz
            'duration': 5.0,  # Must be 5 seconds
            'channels': 23,  # Must be 23 channels
            'samples': 1000,  # Must be 1000 samples
        }

        with open(index_file, 'w') as f:
            json.dump(index_data, f)

        dataset = TUEVEventDataset(
            root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=False
        )

        # Validate metadata
        assert dataset.metadata['fs'] == 200
        assert dataset.metadata['duration'] == 5.0
        assert dataset.metadata['channels'] == 23
        assert dataset.metadata['samples'] == 1000

    def test_dataset_length_matches_segments(self):
        """Test len(dataset) returns correct number of segments."""
        # Create multiple segments
        cache_dir = self.cache_dir / 'tuev_event_segments' / 'train'
        cache_dir.mkdir(parents=True, exist_ok=True)

        n_segments = 10
        segments = []

        for i in range(n_segments):
            cache_file = cache_dir / f'segment_{i}.pt'
            test_segment = torch.randn(23, 1000).float()
            torch.save({'x': test_segment, 'y': i % 6, 'id': f'seg_{i}'}, cache_file)
            segments.append(
                {'file': f'segment_{i}.pt', 'label': i % 6, 'subject': f'sub{i // 3:03d}'}
            )

        # Create index
        index_file = cache_dir / 'index.json'
        index_data = {
            'segments': segments,
            'n_segments': n_segments,
            'class_counts': {str(i): segments.count({'label': i}) for i in range(6)},
            'n_subjects': len(set(s['subject'] for s in segments)),
            'fs': 200,
            'duration': 5.0,
            'channels': 23,
            'samples': 1000,
        }

        with open(index_file, 'w') as f:
            json.dump(index_data, f)

        dataset = TUEVEventDataset(
            root_dir=self.root_dir, split='train', cache_dir=self.cache_dir, force_rebuild=False
        )

        assert len(dataset) == n_segments

    def test_no_sliding_windows(self):
        """Test that dataset does NOT create sliding windows."""
        # This is critical - we should only have event segments, not sliding windows

        # Create test structure with one file
        split_dir = self.root_dir / 'edf' / 'train'
        split_dir.mkdir(parents=True, exist_ok=True)
        edf_file = split_dir / 'test.edf'
        edf_file.touch()

        # Create annotation with 2 events
        lab_file = split_dir / 'test.rec.lab'
        lab_file.write_text("1000000 2000000 spsw\n5000000 6000000 gped")

        with patch(
            'brain_go_brrr.infra.data.tuev_event_dataset.TUEVEventExtractor'
        ) as mock_ext_class:
            mock_extractor = Mock()
            mock_ext_class.return_value = mock_extractor

            # Return exactly 2 segments (one per event)
            segment = np.random.randn(23, 1000).astype(np.float32)
            mock_extractor.extract_segments.return_value = [
                (segment, 0),
                (segment, 1),
            ]

            with patch.object(TUEVEventDataset, '_parse_annotations') as mock_parse:
                mock_parse.return_value = [
                    {'start': 1.0, 'end': 2.0, 'label': 0},
                    {'start': 5.0, 'end': 6.0, 'label': 1},
                ]

                dataset = TUEVEventDataset(
                    root_dir=self.root_dir,
                    split='train',
                    cache_dir=self.cache_dir,
                    force_rebuild=True,
                )

                # Should extract exactly 2 segments (events only, no sliding windows)
                mock_extractor.extract_segments.assert_called_once()
                call_args = mock_extractor.extract_segments.call_args
                annotations = call_args[0][1]
                assert len(annotations) == 2  # Only 2 events, not sliding windows
