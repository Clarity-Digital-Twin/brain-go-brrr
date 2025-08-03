"""Integration tests for Phase 1 data pipeline components.
Tests channel mapping, EDF validation, and window extraction working together.
"""

import numpy as np


class TestDataPipelineIntegration:
    """Test complete data pipeline from EDF to windows."""

    def test_full_pipeline_with_valid_edf(self):
        """Test processing valid EDF through complete pipeline."""
        # Given - mock EDF with old channel names
        mock_edf = MockEDFFile(
            channels=["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
                     "T3", "C3", "CZ", "C4", "T4",  # Old naming
                     "T5", "P3", "PZ", "P4", "T6",   # Old naming
                     "O1", "O2", "ECG"],
            duration_seconds=60,
            sfreq=256
        )

        # When - process through pipeline
        from brain_go_brrr.core.channels import ChannelProcessor
        from brain_go_brrr.core.edf_validator import EDFValidator
        from brain_go_brrr.core.window_extractor import WindowExtractor

        # Step 1: Validate EDF
        validator = EDFValidator()
        validation_result = validator.validate_data(mock_edf)
        assert validation_result.is_valid is True

        # Step 2: Process channels
        channel_processor = ChannelProcessor()
        channel_result = channel_processor.process_channels(mock_edf.ch_names)
        assert channel_result.is_valid is True

        # Verify old names were converted
        assert channel_result.standardized_names[7] == "T7"  # T3->T7
        assert channel_result.standardized_names[11] == "T8"  # T4->T8

        # Step 3: Extract data for selected channels
        selected_data = mock_edf.data[channel_result.selected_indices, :]
        assert selected_data.shape[0] == 19  # Standard channels only

        # Step 4: Extract windows
        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows = extractor.extract(selected_data, mock_edf.sfreq)

        # Then - verify pipeline output
        assert len(windows) > 0
        expected_windows = (60 - 8) // 4 + 1  # 14 windows
        assert len(windows) == expected_windows
        assert all(w.shape == (19, 2048) for w in windows)  # 19 channels, 8s @ 256Hz

    def test_pipeline_handles_insufficient_channels(self):
        """Test pipeline properly rejects EDF with too few channels."""
        # Given - EDF with only 10 channels
        mock_edf = MockEDFFile(
            channels=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"],
            duration_seconds=60,
            sfreq=256
        )

        # When
        from brain_go_brrr.core.channels import ChannelProcessor
        from brain_go_brrr.core.edf_validator import EDFValidator

        # Validate EDF structure
        validator = EDFValidator()
        validation_result = validator.validate_data(mock_edf)
        assert validation_result.is_valid is False  # Too few channels
        assert "channels" in validation_result.errors[0].lower()

        # Channel processing should also fail
        channel_processor = ChannelProcessor()
        channel_result = channel_processor.process_channels(mock_edf.ch_names)
        assert channel_result.is_valid is False
        assert len(channel_result.missing_channels) > 0

    def test_pipeline_handles_short_recordings(self):
        """Test pipeline properly handles recordings shorter than window size."""
        # Given - 5 second recording (shorter than 8s window)
        mock_edf = MockEDFFile(
            channels=["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                     "T7", "C3", "Cz", "C4", "T8",
                     "P7", "P3", "Pz", "P4", "P8",
                     "O1", "O2"],
            duration_seconds=5,
            sfreq=256
        )

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        from brain_go_brrr.core.window_extractor import WindowExtractor

        # Validation should fail (too short)
        validator = EDFValidator(min_duration_seconds=60)
        validation_result = validator.validate_data(mock_edf)
        assert validation_result.is_valid is False

        # But window extraction should handle gracefully
        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows = extractor.extract(mock_edf.data, mock_edf.sfreq)
        assert windows == []  # No windows possible

    def test_pipeline_with_data_quality_issues(self):
        """Test pipeline handles data with quality issues."""
        # Given - EDF with quality problems
        mock_edf = MockEDFFile(
            channels=["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                     "T7", "C3", "Cz", "C4", "T8",
                     "P7", "P3", "Pz", "P4", "P8",
                     "O1", "O2"],
            duration_seconds=60,
            sfreq=256
        )

        # Add quality issues
        mock_edf.data[3, :] = 0.0  # Flat channel
        mock_edf.data[5, 1000:1100] = np.nan  # NaN values
        # Add extreme amplitude with variation (not flat)
        mock_edf.data[7, :] = np.random.randn(mock_edf.data.shape[1]) * 0.002 + 0.003  # ~3-5mV

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        from brain_go_brrr.core.window_extractor import WindowExtractor, WindowValidator

        validator = EDFValidator()
        validation_result = validator.validate_data(mock_edf)

        # Then - should detect issues
        assert validation_result.is_valid is False  # NaN values
        assert any("nan" in err.lower() for err in validation_result.errors)
        assert any("flat" in warn.lower() for warn in validation_result.warnings)
        assert any("amplitude" in warn.lower() for warn in validation_result.warnings)

        # Window validation should also catch issues
        extractor = WindowExtractor()
        windows = extractor.extract(mock_edf.data, mock_edf.sfreq)

        window_validator = WindowValidator(expected_channels=19, expected_samples=2048)
        valid_windows = [w for w in windows if window_validator.is_valid(w)]
        assert len(valid_windows) < len(windows)  # Some windows invalid due to NaN

    def test_batch_processing_pipeline(self):
        """Test processing multiple EDF files in batch."""
        # Given - multiple mock EDFs
        mock_edfs = [
            MockEDFFile(
                channels=["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                         "T3", "C3", "Cz", "C4", "T4",  # Old naming
                         "P7", "P3", "Pz", "P4", "P8",
                         "O1", "O2"],
                duration_seconds=duration,
                sfreq=256
            )
            for duration in [60, 120, 90]  # Different durations
        ]

        # When - process batch
        from brain_go_brrr.core.channels import ChannelProcessor
        from brain_go_brrr.core.edf_validator import EDFValidator
        from brain_go_brrr.core.window_extractor import BatchWindowExtractor

        channel_processor = ChannelProcessor()
        validator = EDFValidator()
        batch_extractor = BatchWindowExtractor()

        processed_recordings = []
        for edf in mock_edfs:
            # Validate
            if not validator.validate_data(edf).is_valid:
                continue

            # Process channels
            channel_result = channel_processor.process_channels(edf.ch_names)
            if not channel_result.is_valid:
                continue

            # Select data
            selected_data = edf.data[channel_result.selected_indices, :]
            processed_recordings.append(selected_data)

        # Extract windows from all recordings
        all_windows, recording_indices = batch_extractor.extract_batch(
            processed_recordings,
            sfreq=256
        )

        # Then - verify batch results
        assert len(processed_recordings) == 3
        # Expected windows: 60s->14, 120s->29, 90s->21 = 64 total
        assert len(all_windows) == 14 + 29 + 21
        assert len(recording_indices) == len(all_windows)
        assert recording_indices.count(0) == 14
        assert recording_indices.count(1) == 29
        assert recording_indices.count(2) == 21


class MockEDFFile:
    """Mock EDF file for integration testing."""

    def __init__(self, channels: list[str], duration_seconds: float, sfreq: float):
        self.ch_names = channels
        self.duration = duration_seconds
        self.sfreq = sfreq
        self.n_channels = len(channels)
        self.n_times = int(duration_seconds * sfreq)

        # Generate realistic mock data
        self.data = np.random.randn(self.n_channels, self.n_times) * 50e-6  # 50µV typical
