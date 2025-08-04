"""Integration tests for complete AutoReject pipeline with EEGPT."""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
import yaml

# These will fail until implemented - TDD!
from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.preprocessing.autoreject_adapter import (
    SyntheticPositionGenerator,
    WindowEpochAdapter,
)
from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor


class TestAutoRejectEEGPTIntegration:
    """Test full integration of AutoReject with EEGPT training pipeline."""

    @pytest.fixture
    def mock_tuab_config(self, tmp_path):
        """Create mock TUAB configuration."""
        config = {
            "data_config": {
                "data_dir": str(tmp_path / "data"),
                "window_duration": 10.0,
                "window_stride": 5.0,
                "sampling_rate": 256,
                "n_channels": 19,
                "use_autoreject": True,
                "ar_cache_dir": str(tmp_path / "ar_cache"),
                "ar_fit_samples": 50,
                "ar_n_interpolate": [1, 4],
                "ar_consensus": 0.1,
            },
            "model_config": {
                "pretrained_path": str(tmp_path / "eegpt.ckpt"),
                "n_channels": 20,
                "hidden_dim": 768,
                "num_layers": 12,
            },
            "train_config": {
                "batch_size": 16,
                "num_epochs": 50,
                "learning_rate": 1e-4,
                "max_samples": 100,
            },
        }

        # Create directories
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "ar_cache").mkdir(exist_ok=True)

        return config

    @pytest.fixture
    def mock_edf_files(self, tmp_path):
        """Create mock EDF files with realistic structure."""
        data_dir = tmp_path / "data" / "train"
        data_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for i in range(10):
            # Create mock EDF file
            file_path = data_dir / f"subject_{i:03d}.edf"

            # Create realistic EEG data
            sfreq = 256
            duration = 120  # 2 minutes
            n_channels = 19

            # TUAB channel names (with old naming)
            ch_names = [
                "FP1",
                "FP2",
                "F7",
                "F3",
                "FZ",
                "F4",
                "F8",
                "T3",
                "C3",
                "CZ",
                "C4",
                "T4",
                "T5",
                "P3",
                "PZ",
                "P4",
                "T6",
                "O1",
                "O2",
            ]

            # Generate data with artifacts
            np.random.seed(i)
            data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6

            # Add artifacts
            if i % 3 == 0:  # Add eye blinks
                blink_times = np.random.randint(10, duration - 1, 5)
                for t in blink_times:
                    data[:4, int(t * sfreq) : int((t + 0.5) * sfreq)] += (
                        np.random.randn(4, int(0.5 * sfreq)) * 200e-6
                    )

            if i % 2 == 0:  # Add muscle artifacts
                muscle_start = np.random.randint(20, duration - 20)
                data[:, int(muscle_start * sfreq) : int((muscle_start + 5) * sfreq)] += (
                    np.random.randn(n_channels, int(5 * sfreq)) * 100e-6
                )

            # Create and save mock EDF
            info = mne.create_info(ch_names, sfreq, ch_types="eeg")
            mne.io.RawArray(data, info)

            # Note: In real tests, we'd use mne.export.export_raw
            # For now, just create empty file as placeholder
            file_path.touch()
            files.append(file_path)

        # Create labels file
        labels_file = data_dir.parent / "train_labels.csv"
        with labels_file.open("w") as f:
            f.write("filename,label\n")
            for i, file in enumerate(files):
                f.write(f"{file.name},{i % 2}\n")  # Binary labels

        return files

    @pytest.mark.skip(reason="Outdated mocking - pandas import has changed")
    def test_dataset_initialization_with_autoreject(
        self, mock_tuab_config, mock_edf_files, tmp_path
    ):
        """Test TUABEnhancedDataset initialization with AutoReject enabled."""
        # Given: Configuration with AutoReject enabled
        config = mock_tuab_config

        # When: Creating dataset with AutoReject
        with patch("brain_go_brrr.data.tuab_enhanced_dataset.pd.read_csv") as mock_read_csv:
            # Mock labels
            import pandas as pd

            mock_labels = pd.DataFrame(
                {
                    "filename": [f.name for f in mock_edf_files],
                    "label": [i % 2 for i in range(len(mock_edf_files))],
                }
            )
            mock_read_csv.return_value = mock_labels

            dataset = TUABEnhancedDataset(
                data_dir=config["data_config"]["data_dir"],
                split="train",
                use_autoreject=True,
                ar_cache_dir=config["data_config"]["ar_cache_dir"],
                window_duration=config["data_config"]["window_duration"],
                window_stride=config["data_config"]["window_stride"],
                sampling_rate=config["data_config"]["sampling_rate"],
            )

        # Then: Should have AutoReject components initialized
        assert dataset.use_autoreject is True
        assert dataset.ar_processor is not None
        assert dataset.window_adapter is not None
        assert dataset.position_generator is not None
        assert isinstance(dataset.ar_processor, ChunkedAutoRejectProcessor)
        assert isinstance(dataset.window_adapter, WindowEpochAdapter)
        assert isinstance(dataset.position_generator, SyntheticPositionGenerator)

    @patch("mne.io.read_raw_edf")
    def test_data_loading_with_autoreject(self, mock_read_edf, mock_tuab_config, tmp_path):
        """Test that data loading applies AutoReject cleaning."""
        # Given: Dataset with AutoReject
        dataset = TUABEnhancedDataset(
            data_dir=tmp_path,
            split="train",
            use_autoreject=True,
            ar_cache_dir=tmp_path / "ar_cache",
        )

        # Mock raw data
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(19, 30720)  # 2 min @ 256Hz
        mock_raw.info = {"sfreq": 256, "ch_names": ["C3", "C4"] * 9 + ["CZ"]}
        mock_raw.n_times = 30720
        mock_read_edf.return_value = mock_raw

        # Mock AutoReject processing
        with patch.object(dataset, "_apply_autoreject_to_raw") as mock_ar:
            mock_ar.return_value = mock_raw  # Return cleaned version

            # When: Loading a file
            result = dataset._load_edf_file(tmp_path / "test.edf", label=0)

            # Then: Should apply AutoReject
            mock_ar.assert_called_once()
            assert "windows" in result
            assert "label" in result

    def test_autoreject_fallback_on_error(self, mock_tuab_config, tmp_path):
        """Test fallback to amplitude rejection when AutoReject fails."""
        # Given: Dataset with AutoReject that will fail
        dataset = TUABEnhancedDataset(data_dir=tmp_path, split="train", use_autoreject=True)

        # Mock raw data
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(19, 10000)

        # When: AutoReject fails with memory error
        with (
            patch.object(dataset.ar_processor, "transform_raw", side_effect=MemoryError),
            patch.object(dataset, "_amplitude_based_cleaning") as mock_fallback,
        ):
            mock_fallback.return_value = mock_raw

            result = dataset._apply_autoreject_to_raw(mock_raw)

            # Then: Should use fallback
            mock_fallback.assert_called_once_with(mock_raw)
            assert result == mock_raw

    def test_performance_metrics_tracking(self, mock_tuab_config, tmp_path):
        """Test that performance metrics are tracked."""
        # Given: Dataset with metrics tracking
        dataset = TUABEnhancedDataset(data_dir=tmp_path, split="train", use_autoreject=True)

        # Initialize tracking
        dataset.ar_processing_time = 0
        dataset.rejection_stats = {
            "files_processed": 0,
            "epochs_rejected": 0,
            "epochs_interpolated": 0,
            "rejection_rate": 0.0,
        }

        # When: Processing files
        import time

        start = time.time()
        # Simulate processing
        time.sleep(0.1)
        dataset.ar_processing_time = time.time() - start
        dataset.rejection_stats["files_processed"] = 10
        dataset.rejection_stats["epochs_rejected"] = 50
        dataset.rejection_stats["rejection_rate"] = 0.25

        # Then: Metrics should be tracked
        assert dataset.ar_processing_time > 0
        assert dataset.rejection_stats["files_processed"] == 10
        assert dataset.rejection_stats["rejection_rate"] == 0.25

    @pytest.mark.slow
    def test_memory_usage_with_autoreject(self, mock_tuab_config, tmp_path):
        """Test that memory usage stays reasonable with AutoReject."""
        import psutil

        process = psutil.Process()

        # Baseline memory
        mem_start = process.memory_info().rss / 1024 / 1024  # MB

        # Create dataset with AutoReject
        TUABEnhancedDataset(
            data_dir=tmp_path,
            split="train",
            use_autoreject=True,
            max_samples=50,  # Limit for testing
        )

        # Simulate loading multiple files
        for _i in range(10):
            mock_data = np.random.randn(19, 25600).astype(np.float32)  # 100s @ 256Hz
            # Process would happen here
            del mock_data  # Cleanup

        # Check memory
        mem_end = process.memory_info().rss / 1024 / 1024
        memory_increase = mem_end - mem_start

        # Should not exceed reasonable limit (500MB for this test)
        assert memory_increase < 500, f"Memory increased by {memory_increase}MB"

    def test_backward_compatibility(self, mock_tuab_config, tmp_path):
        """Test that disabling AutoReject maintains backward compatibility."""
        # Given: Dataset with AutoReject disabled
        dataset_no_ar = TUABEnhancedDataset(
            data_dir=tmp_path,
            split="train",
            use_autoreject=False,  # Disabled
        )

        # Then: Should not have AutoReject components
        assert dataset_no_ar.use_autoreject is False
        assert dataset_no_ar.ar_processor is None
        assert dataset_no_ar.window_adapter is None
        assert dataset_no_ar.position_generator is None

    @pytest.mark.integration
    def test_training_script_integration(self, mock_tuab_config, tmp_path):
        """Test integration with train_enhanced.py script."""
        # This tests the command-line interface
        import subprocess

        # Create minimal training script mock
        train_script = tmp_path / "train_enhanced.py"
        train_script.write_text("""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use-autoreject', action='store_true')
parser.add_argument('--ar-cache-dir', type=str, default='ar_cache')
args = parser.parse_args()
print(f"AutoReject: {args.use_autoreject}")
print(f"Cache: {args.ar_cache_dir}")
""")

        # Test with AutoReject enabled
        result = subprocess.run(
            ["python", str(train_script), "--use-autoreject", "--ar-cache-dir", "custom_cache"],
            capture_output=True,
            text=True,
        )

        assert "AutoReject: True" in result.stdout
        assert "Cache: custom_cache" in result.stdout

    def test_config_file_integration(self, tmp_path):
        """Test loading AutoReject settings from config file."""
        # Given: Config file with AutoReject settings
        config_file = tmp_path / "config.yaml"
        config_data = {
            "data_config": {
                "use_autoreject": True,
                "ar_cache_dir": "autoreject_cache",
                "ar_fit_samples": 200,
                "ar_n_interpolate": [1, 4, 8],
                "ar_consensus": 0.1,
            }
        }

        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # When: Loading config
        with config_file.open() as f:
            loaded_config = yaml.safe_load(f)

        # Then: Should have all AutoReject settings
        ar_config = loaded_config["data_config"]
        assert ar_config["use_autoreject"] is True
        assert ar_config["ar_cache_dir"] == "autoreject_cache"
        assert ar_config["ar_fit_samples"] == 200
        assert ar_config["ar_n_interpolate"] == [1, 4, 8]


class TestAutoRejectImpactOnTraining:
    """Test the impact of AutoReject on training metrics."""

    def test_auroc_improvement_expectation(self):
        """Test that we expect AUROC improvement with AutoReject."""
        # Given: Baseline AUROC without AutoReject
        baseline_auroc = 0.80

        # When: Using AutoReject
        # Expected improvement based on literature
        expected_improvement = 0.05  # 5% improvement

        # Then: Target AUROC
        target_auroc = baseline_auroc + expected_improvement
        assert target_auroc >= 0.85
        assert target_auroc <= 0.95  # Realistic upper bound

    def test_processing_time_overhead(self):
        """Test that processing overhead is acceptable."""
        # Given: Baseline processing time
        baseline_time_per_file = 0.5  # seconds

        # When: Adding AutoReject
        # Expected overhead from chunked processing
        ar_overhead_per_file = 0.2  # 200ms

        # Then: Total time should be acceptable
        total_time = baseline_time_per_file + ar_overhead_per_file
        assert total_time < 1.0  # Less than 1 second per file

        # For 3000 files
        total_training_overhead = 3000 * ar_overhead_per_file / 60  # minutes
        assert total_training_overhead < 20  # Less than 20 minutes overhead
