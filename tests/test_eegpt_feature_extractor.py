"""Test unified EEGPT feature extraction service."""

from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest


class TestEEGPTFeatureExtractor:
    """Test EEGPT feature extraction for integration."""

    @pytest.fixture
    def sample_raw(self):
        """Create sample EEG data."""
        sfreq = 256
        duration = 12  # 12 seconds = 3 windows of 4 seconds
        n_channels = 19
        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
        ]

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        return mne.io.RawArray(data, info)

    @pytest.fixture
    def mock_eegpt_model(self):
        """Mock EEGPT model for testing."""
        mock_model = Mock()
        # EEGPT returns (batch_size, n_windows, 512) embeddings
        mock_model.extract_features.return_value = np.random.randn(1, 3, 512).astype(np.float32)
        return mock_model

    def test_feature_extractor_initialization(self):
        """Test feature extractor can be initialized."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        extractor = EEGPTFeatureExtractor()
        assert extractor is not None
        assert hasattr(extractor, "extract_embeddings")

    def test_extract_embeddings_shape(self, sample_raw, mock_eegpt_model):
        """Test that embeddings have correct shape."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        with patch(
            "brain_go_brrr.core.features.extractor.EEGPTModel", return_value=mock_eegpt_model
        ):
            extractor = EEGPTFeatureExtractor()
            embeddings = extractor.extract_embeddings(sample_raw)

            # Should return (n_windows, 512) for single recording
            assert embeddings.shape == (3, 512)
            assert embeddings.dtype == np.float32

    def test_caching_embeddings(self, sample_raw, mock_eegpt_model, tmp_path, monkeypatch):
        """Test that embeddings are cached for efficiency."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        # Use a temporary directory for cache to ensure isolation
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir(exist_ok=True)
        monkeypatch.setenv("BGB_CACHE_DIR", str(cache_dir))

        with patch(
            "brain_go_brrr.core.features.extractor.EEGPTModel", return_value=mock_eegpt_model
        ):
            extractor = EEGPTFeatureExtractor(enable_cache=True)

            # First extraction
            embeddings1 = extractor.extract_embeddings(sample_raw)

            # Second extraction - should use cache
            embeddings2 = extractor.extract_embeddings(sample_raw)

            # Should be the same object (cached)
            assert np.array_equal(embeddings1, embeddings2)

            # Model should be called once per window on first extraction (3 windows)
            # But not called at all on second extraction (cached)
            assert mock_eegpt_model.extract_features.call_count == 3

    def test_window_extraction(self, sample_raw):
        """Test window extraction for EEGPT processing."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        extractor = EEGPTFeatureExtractor()
        windows = extractor._extract_windows(sample_raw, window_size=4.0, overlap=0.0)

        # 12 seconds / 4 seconds = 3 windows
        assert len(windows) == 3

        # Each window should be (n_channels, samples_per_window)
        assert windows[0].shape == (19, 1024)  # 4 * 256 = 1024 samples

    def test_overlapping_windows(self, sample_raw):
        """Test extraction with overlapping windows."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        extractor = EEGPTFeatureExtractor()
        windows = extractor._extract_windows(sample_raw, window_size=4.0, overlap=2.0)

        # With 50% overlap: (12 - 4) / 2 + 1 = 5 windows
        assert len(windows) == 5

    def test_preprocessing_for_eegpt(self, sample_raw):
        """Test preprocessing matches EEGPT requirements."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        extractor = EEGPTFeatureExtractor()
        preprocessed = extractor._preprocess_for_eegpt(sample_raw)

        # Should be resampled to 256 Hz (already at 256)
        assert preprocessed.info["sfreq"] == 256

        # Should be filtered 0.5-45 Hz
        # Check that data has changed (filtering applied)
        assert not np.array_equal(sample_raw.get_data(), preprocessed.get_data())

    def test_embedding_metadata(self, sample_raw, mock_eegpt_model):
        """Test that metadata is returned with embeddings."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        with patch(
            "brain_go_brrr.core.features.extractor.EEGPTModel", return_value=mock_eegpt_model
        ):
            extractor = EEGPTFeatureExtractor()
            result = extractor.extract_embeddings_with_metadata(sample_raw)

            assert "embeddings" in result
            assert "window_times" in result
            assert "sampling_rate" in result
            assert "n_channels" in result

            # Window times should match number of windows
            assert len(result["window_times"]) == 3
            assert result["window_times"][0] == (0.0, 4.0)
            assert result["window_times"][1] == (4.0, 8.0)
            assert result["window_times"][2] == (8.0, 12.0)

    def test_batch_processing(self, mock_eegpt_model):
        """Test batch processing of multiple recordings."""
        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        # Create multiple recordings
        raws = []
        for _ in range(3):
            sfreq = 256
            data = np.random.randn(19, sfreq * 8) * 50e-6  # 8 seconds each
            info = mne.create_info(
                ch_names=[f"Ch{i}" for i in range(19)], sfreq=sfreq, ch_types="eeg"
            )
            raws.append(mne.io.RawArray(data, info))

        # Mock batch output
        mock_eegpt_model.extract_features.return_value = np.random.randn(3, 2, 512).astype(
            np.float32
        )

        with patch(
            "brain_go_brrr.core.features.extractor.EEGPTModel", return_value=mock_eegpt_model
        ):
            extractor = EEGPTFeatureExtractor()
            embeddings_list = extractor.extract_batch_embeddings(raws)

            assert len(embeddings_list) == 3
            for embeddings in embeddings_list:
                assert embeddings.shape == (2, 512)  # 2 windows per recording

    @pytest.mark.integration
    def test_real_eegpt_model_loading(self):
        """Test loading real EEGPT model if available."""
        model_path = Path("data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

        if not model_path.exists():
            pytest.skip("EEGPT model not found")

        from brain_go_brrr.core.features import EEGPTFeatureExtractor

        extractor = EEGPTFeatureExtractor(model_path=model_path)
        assert extractor.model is not None
        assert hasattr(extractor.model, "extract_features")
