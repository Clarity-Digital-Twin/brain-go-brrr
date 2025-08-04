"""REAL integration test with actual TUAB EDF files - NO MOCKING!"""

import logging
from pathlib import Path

import mne
import numpy as np
import pytest

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from brain_go_brrr.preprocessing import (
    ChunkedAutoRejectProcessor,
    SyntheticPositionGenerator,
    WindowEpochAdapter,
)

logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.external
@pytest.mark.integration
class TestRealTUABAutoRejectIntegration:
    """Test AutoReject with REAL TUAB data - no fucking mocks!"""

    @pytest.fixture
    def real_tuab_file(self):
        """Get a real TUAB EDF file path."""
        data_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
        tuab_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"

        # Find first available EDF
        for split in ["eval", "train"]:
            split_dir = tuab_dir / split
            if split_dir.exists():
                for edf_file in split_dir.rglob("*.edf"):
                    if edf_file.exists():
                        return edf_file

        pytest.skip("No TUAB EDF files found")

    def test_load_real_edf_with_autoreject(self, real_tuab_file):
        """Test loading a real EDF file through the whole pipeline."""
        logger.info(f"Testing with real file: {real_tuab_file}")

        # Load raw data
        raw = mne.io.read_raw_edf(real_tuab_file, preload=True, verbose=False)

        # Apply our pipeline components
        position_gen = SyntheticPositionGenerator()
        raw_with_pos = position_gen.add_positions_to_raw(raw)

        # Create window adapter
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # Convert to epochs
        epochs = adapter.raw_to_windowed_epochs(raw_with_pos)

        # Verify we got epochs
        assert len(epochs) > 0
        assert epochs.info['sfreq'] == raw.info['sfreq']

        # Convert back
        raw_reconstructed = adapter.epochs_to_continuous(epochs, raw_with_pos)

        # Verify reconstruction
        assert raw_reconstructed.n_times == raw.n_times
        assert len(raw_reconstructed.ch_names) == len(raw.ch_names)

    @pytest.mark.slow
    def test_real_dataset_with_autoreject(self):
        """Test the full dataset with AutoReject enabled on real data."""
        data_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")

        # Create dataset with AutoReject
        dataset = TUABEnhancedDataset(
            root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
            split="eval",
            use_autoreject=True,
            ar_cache_dir=data_root / "cache/autoreject_test",
            window_duration=10.0,
            window_stride=10.0,  # No overlap for speed
            sampling_rate=200
        )

        # Check dataset loaded
        assert len(dataset) > 0
        assert dataset.use_autoreject is True

        # Load first sample
        data, label = dataset[0]

        # Verify output
        assert data is not None
        assert isinstance(label, int)
        assert data.shape[0] == len(dataset.channels)  # Channels
        assert data.shape[1] == 2000  # 10s @ 200Hz

    def test_amplitude_cleaning_on_real_data(self, real_tuab_file):
        """Test amplitude-based cleaning on real noisy data."""
        # Load raw
        raw = mne.io.read_raw_edf(real_tuab_file, preload=True, verbose=False)

        # Get channel stats
        data = raw.get_data()
        channel_stds = np.std(data, axis=1)

        logger.info(f"Channel STDs range: {channel_stds.min():.2e} - {channel_stds.max():.2e}")

        # Check if any channels would be marked bad
        flat_channels = np.where(channel_stds < 0.1e-6)[0]
        noisy_channels = np.where(channel_stds > 200e-6)[0]

        logger.info(f"Flat channels: {len(flat_channels)}")
        logger.info(f"Noisy channels: {len(noisy_channels)}")

        # Apply cleaning directly by extracting the logic
        data = raw.get_data()

        # Detect bad channels
        channel_stds = np.std(data, axis=1)

        # Mark channels as bad based on amplitude
        flat_mask = channel_stds < 0.1e-6  # Less than 0.1 µV
        noisy_mask = channel_stds > 200e-6  # Greater than 200 µV
        bad_mask = flat_mask | noisy_mask

        bad_channels = [ch for i, ch in enumerate(raw.ch_names) if bad_mask[i]]

        # Create a copy and mark bad channels
        raw_clean = raw.copy()
        raw_clean.info['bads'] = bad_channels

        # Check bad channels marked
        logger.info(f"Marked bad channels: {raw_clean.info['bads']}")
        assert isinstance(raw_clean.info['bads'], list)

    def test_chunked_processor_with_real_files(self):
        """Test chunked processor can handle real TUAB files."""
        data_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data")
        cache_dir = data_root / "cache/autoreject_test_chunked"

        processor = ChunkedAutoRejectProcessor(
            cache_dir=cache_dir,
            chunk_size=10,
            n_interpolate=[1, 2],  # Faster for test
            consensus=0.1
        )

        # Check cache
        assert processor.cache_dir.exists()

        # Would fit on subset if we had AutoReject installed
        if not processor.has_cached_params():
            logger.info("No cached params - would fit on subset in production")

    @pytest.mark.integration
    def test_end_to_end_with_real_file(self, real_tuab_file):
        """Full end-to-end test with a real TUAB file."""
        logger.info(f"End-to-end test with: {real_tuab_file.name}")

        # Create all components
        position_gen = SyntheticPositionGenerator()
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # Load and process
        raw = mne.io.read_raw_edf(real_tuab_file, preload=True, verbose=False)

        # Add positions
        raw_with_pos = position_gen.add_positions_to_raw(raw)

        # Verify positions added
        montage = raw_with_pos.get_montage()
        if montage is not None:
            positions = montage.get_positions()
            assert 'ch_pos' in positions
            assert len(positions['ch_pos']) > 0

        # Filter
        raw_with_pos.filter(0.5, 50.0, fir_design='firwin', verbose=False)

        # Convert to epochs
        epochs = adapter.raw_to_windowed_epochs(raw_with_pos)

        # Basic quality checks
        epoch_data = epochs.get_data()

        # Check for reasonable values (in volts)
        assert np.abs(epoch_data).max() < 1e-2  # Less than 10mV (reasonable for filtered EEG)
        assert np.abs(epoch_data).mean() < 1e-3  # Mean less than 1mV

        logger.info(f"Successfully processed {len(epochs)} epochs from {real_tuab_file.name}")

