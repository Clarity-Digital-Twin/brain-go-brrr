"""Integration tests for TUEV event extraction pipeline - paper parity validation."""

import json
import tempfile
from pathlib import Path

import torch

from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
from brain_go_brrr.infra.preprocessing.tuev_event_extractor import TUEVEventExtractor


class TestTUEVEventPipeline:
    """Integration tests for the complete TUEV event extraction pipeline."""

    def test_extractor_to_dataset_flow(self):
        """Test full flow from extractor to dataset."""
        extractor = TUEVEventExtractor()

        # Verify extractor configuration
        assert extractor.target_fs == 200
        assert extractor.segment_duration == 5.0
        assert len(extractor.TUEV_CHANNELS_REF) == 23

        # Create temporary dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TUEVEventDataset(
                root_dir=Path(temp_dir) / "tuev",
                split='train',
                cache_dir=Path(temp_dir) / "cache",
                force_rebuild=False,
            )

            # Verify dataset configuration
            assert len(dataset.class_mapping) == 6
            assert dataset.metadata['fs'] == 200
            assert dataset.metadata['duration'] == 5.0
            assert dataset.metadata['channels'] == 23
            assert dataset.metadata['samples'] == 1000

    def test_channel_mapper_integration(self):
        """Test that channel mapper works with extracted segments."""
        mapper = TUEVChannelMapper(dropout=0.0)  # No dropout for testing

        # Create mock segment with correct shape
        batch_size = 4
        segment = torch.randn(batch_size, 23, 1, 1000)  # Add spatial dim for conv

        # Apply mapping
        mapped = mapper(segment)

        # Verify output shape
        assert mapped.shape == (batch_size, 20, 1000)

        # Test gradient flow
        loss = mapped.sum()
        loss.backward()

        # Check gradients exist
        for param in mapper.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_paper_parity_requirements(self):
        """Test that all paper parity requirements are met."""
        # Test sampling rate
        extractor = TUEVEventExtractor()
        assert extractor.target_fs == 200, "Must use 200Hz (not 256Hz)"

        # Test segment duration
        assert extractor.segment_duration == 5.0, "Must use 5-second segments"

        # Test window extraction
        assert extractor.tmin == -2.0, "Must extract 2s before event"
        assert extractor.tmax == 3.0, "Must extract 3s after event"

        # Test channel count
        assert len(extractor.TUEV_CHANNELS_REF) == 23, "Must use 23 channels"

        # Test channel names (referential, not bipolar)
        for ch in extractor.TUEV_CHANNELS_REF:
            assert ch.endswith('-REF'), f"Must use referential channels, got {ch}"

    def test_data_shape_consistency(self):
        """Test that data shapes are consistent throughout pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal cache
            cache_dir = Path(temp_dir) / "cache" / "tuev_event_segments" / "train"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Save test segment
            test_segment = torch.randn(23, 1000).float()
            segment_file = cache_dir / "test_segment.pt"
            torch.save({'x': test_segment, 'y': 0, 'id': 'test'}, segment_file)

            # Create index
            index_file = cache_dir / "index.json"
            with index_file.open('w') as f:
                json.dump(
                    {
                        'segments': [{'file': 'test_segment.pt', 'label': 0, 'subject': 'test'}],
                        'n_segments': 1,
                        'class_counts': {'0': 1},
                        'n_subjects': 1,
                        'fs': 200,
                        'duration': 5.0,
                        'channels': 23,
                        'samples': 1000,
                    },
                    f,
                )

            # Load dataset
            dataset = TUEVEventDataset(
                root_dir=Path(temp_dir) / "tuev",
                split='train',
                cache_dir=Path(temp_dir) / "cache",
                force_rebuild=False,
            )

            # Get item
            x, y = dataset[0]

            # Verify shape
            assert x.shape == (23, 1000)
            assert x.dtype == torch.float32
            assert isinstance(y, int)

            # Test with channel mapper
            mapper = TUEVChannelMapper(dropout=0.0)
            x_batch = x.unsqueeze(0).unsqueeze(2)  # Add batch and spatial dims
            mapped = mapper(x_batch)

            assert mapped.shape == (1, 20, 1000)

    def test_no_sliding_windows(self):
        """Critical test: Ensure NO sliding windows are created."""
        # This is the key difference from our failed approach
        # We should ONLY extract event segments, not slide over recordings

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TUEVEventDataset(
                root_dir=Path(temp_dir) / "tuev",
                split='train',
                cache_dir=Path(temp_dir) / "cache",
                force_rebuild=False,
            )

            # With no data, should have 0 segments (not sliding windows)
            assert len(dataset) == 0

            # Metadata should still indicate event segments
            assert dataset.metadata['duration'] == 5.0
            assert dataset.metadata['fs'] == 200

    def test_class_distribution_balanced(self):
        """Test that class distribution is balanced (not 99.5% background)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache" / "tuev_event_segments" / "train"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Create balanced segments (unlike our 99.5% background problem)
            segments = []
            for class_idx in range(6):
                for i in range(10):  # 10 segments per class
                    segment_id = f"seg_{class_idx}_{i}"
                    segment_file = cache_dir / f"{segment_id}.pt"

                    test_segment = torch.randn(23, 1000).float()
                    torch.save({'x': test_segment, 'y': class_idx, 'id': segment_id}, segment_file)

                    segments.append(
                        {'file': f"{segment_id}.pt", 'label': class_idx, 'subject': f"sub{i:03d}"}
                    )

            # Create balanced index
            class_counts = {str(i): 10 for i in range(6)}

            index_file = cache_dir / "index.json"
            with index_file.open('w') as f:
                json.dump(
                    {
                        'segments': segments,
                        'n_segments': 60,
                        'class_counts': class_counts,
                        'n_subjects': 10,
                        'fs': 200,
                        'duration': 5.0,
                        'channels': 23,
                        'samples': 1000,
                    },
                    f,
                )

            dataset = TUEVEventDataset(
                root_dir=Path(temp_dir) / "tuev",
                split='train',
                cache_dir=Path(temp_dir) / "cache",
                force_rebuild=False,
            )

            # Verify balanced distribution
            counts = dataset.metadata['class_counts']
            assert len(counts) == 6

            # All classes should have equal representation
            for class_count in counts.values():
                assert int(class_count) == 10

            # No class should dominate (unlike our 99.5% background)
            total = sum(int(v) for v in counts.values())
            for v in counts.values():
                ratio = int(v) / total
                assert 0.1 <= ratio <= 0.2  # Each class ~16.7%

    def test_metadata_validation(self):
        """Test that all required metadata is present and correct."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TUEVEventDataset(
                root_dir=Path(temp_dir) / "tuev",
                split='train',
                cache_dir=Path(temp_dir) / "cache",
                force_rebuild=False,
            )

            # Required metadata fields
            required_fields = ['fs', 'duration', 'channels', 'samples']

            for field in required_fields:
                assert field in dataset.metadata, f"Missing required field: {field}"

            # Validate values match paper
            assert dataset.metadata['fs'] == 200, "Must use 200Hz sampling"
            assert dataset.metadata['duration'] == 5.0, "Must use 5-second segments"
            assert dataset.metadata['channels'] == 23, "Must use 23 channels"
            assert dataset.metadata['samples'] == 1000, "Must have 1000 samples (5s * 200Hz)"
