"""Tests for the complete EEGPT pipeline."""

from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest
import torch

from src.brain_go_brrr.models.eegpt_architecture import EEGPTConfig
from src.brain_go_brrr.models.eegpt_model import (
    EEGPTModel,
    extract_features_from_raw,
    preprocess_for_eegpt,
)


class TestEEGPTPreprocessing:
    """Test preprocessing functions for EEGPT."""

    @pytest.fixture
    def sample_raw(self):
        """Create sample raw EEG data."""
        sfreq = 512  # Original high sampling rate
        duration = 60  # 1 minute
        n_channels = 19
        n_samples = int(sfreq * duration)

        # Create realistic EEG data
        np.random.seed(42)
        data = np.random.randn(n_channels, n_samples) * 50  # 50 µV

        # Standard 10-20 channel names
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                   'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                   'Fz', 'Cz', 'Pz']
        ch_types = ['eeg'] * n_channels

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        return raw

    def test_preprocess_resampling(self, sample_raw):
        """Test that preprocessing resamples to 256 Hz."""
        processed = preprocess_for_eegpt(sample_raw)

        assert processed.info['sfreq'] == 256
        # Check data shape changed appropriately
        expected_samples = int(sample_raw.n_times * 256 / sample_raw.info['sfreq'])
        assert abs(processed.n_times - expected_samples) < 10  # Allow small difference

    def test_preprocess_keeps_original_if_256hz(self):
        """Test that 256 Hz data is not resampled."""
        # Create data already at 256 Hz
        sfreq = 256
        data = np.random.randn(3, sfreq * 10) * 50
        info = mne.create_info(['C3', 'C4', 'Cz'], sfreq=sfreq, ch_types=['eeg'] * 3)
        raw = mne.io.RawArray(data, info)

        processed = preprocess_for_eegpt(raw)

        assert processed.info['sfreq'] == 256
        assert processed.n_times == raw.n_times

    def test_preprocess_preserves_data_continuity(self, sample_raw):
        """Test that preprocessing doesn't break data continuity."""
        # Add a marker in the data
        sample_raw._data[0, 1000:1100] = 1000  # Large spike

        processed = preprocess_for_eegpt(sample_raw)

        # The spike should still be visible (though at different location due to resampling)
        assert np.max(processed._data[0]) > 500

    def test_preprocess_handles_bad_channels(self, sample_raw):
        """Test preprocessing with bad channels marked."""
        sample_raw.info['bads'] = ['T3', 'T4']

        processed = preprocess_for_eegpt(sample_raw)

        # Bad channels should be preserved
        assert processed.info['bads'] == ['T3', 'T4']
        assert processed.info['sfreq'] == 256


class TestEEGPTWindowExtraction:
    """Test window extraction for EEGPT."""

    @pytest.fixture
    def eegpt_model(self):
        """Create EEGPTModel instance with mocked components."""
        with patch('src.brain_go_brrr.models.eegpt_model.load_eegpt_encoder') as mock_load:
            # Mock the encoder
            mock_encoder = Mock()
            mock_encoder.prepare_chan_ids = Mock(return_value=torch.zeros(19))
            mock_load.return_value = mock_encoder

            # Create model without loading checkpoint
            model = EEGPTModel(checkpoint_path=None)
            model.encoder = mock_encoder

            return model

    def test_extract_windows_basic(self, eegpt_model):
        """Test basic window extraction."""
        # Create 10 seconds of data at 256 Hz
        sfreq = 256
        duration = 10
        n_channels = 19
        data = np.random.randn(n_channels, sfreq * duration)

        windows = eegpt_model.extract_windows(data, sfreq)

        # Should have 7 windows (4-second windows with 2-second overlap)
        # Windows at: 0-4, 2-6, 4-8, 6-10 (last window may be partial)
        assert len(windows) >= 3
        assert len(windows) <= 4

        # Each window should be (channels, samples)
        assert windows[0].shape == (n_channels, 4 * sfreq)

    def test_extract_windows_overlap(self, eegpt_model):
        """Test window extraction with custom overlap."""
        sfreq = 256
        duration = 10
        n_channels = 19
        data = np.random.randn(n_channels, sfreq * duration)

        # No overlap
        windows = eegpt_model.extract_windows(data, sfreq, overlap=0.0)

        # Should have exactly 2 full windows (0-4, 4-8) plus maybe partial
        assert len(windows) >= 2
        assert len(windows) <= 3

    def test_extract_windows_short_data(self, eegpt_model):
        """Test window extraction with data shorter than window."""
        sfreq = 256
        duration = 2  # Only 2 seconds
        n_channels = 19
        data = np.random.randn(n_channels, sfreq * duration)

        windows = eegpt_model.extract_windows(data, sfreq)

        # Should pad and return 1 window
        assert len(windows) == 1
        assert windows[0].shape == (n_channels, 4 * sfreq)

    def test_extract_windows_different_sampling_rate(self, eegpt_model):
        """Test window extraction handles resampling."""
        sfreq = 512  # Different from target 256 Hz
        duration = 8
        n_channels = 19
        data = np.random.randn(n_channels, sfreq * duration)

        windows = eegpt_model.extract_windows(data, sfreq)

        # Windows should be at 256 Hz
        assert windows[0].shape == (n_channels, 4 * 256)


class TestEEGPTFeatureExtraction:
    """Test feature extraction functionality."""

    @pytest.fixture
    def mock_eegpt_model(self):
        """Create fully mocked EEGPT model."""
        with patch('src.brain_go_brrr.models.eegpt_model.load_eegpt_encoder'):
            model = EEGPTModel(checkpoint_path=None)

            # Mock encoder to return features
            mock_encoder = Mock()
            mock_encoder.prepare_chan_ids = Mock(return_value=torch.zeros(19))

            # Mock forward pass to return features
            def mock_forward(x, chan_ids):
                batch_size = x.shape[0]
                # Return mock features (batch, n_patches, feature_dim)
                return torch.randn(batch_size, 8, 768)  # 8 summary tokens, 768 dim

            mock_encoder.forward = mock_forward
            mock_encoder.__call__ = mock_forward
            model.encoder = mock_encoder

            return model

    def test_extract_features_single_window(self, mock_eegpt_model):
        """Test feature extraction for single window."""
        # Create single window
        window = np.random.randn(19, 1024)  # 19 channels, 4 seconds at 256 Hz

        features = mock_eegpt_model.extract_features(window)

        # Should return numpy array
        assert isinstance(features, np.ndarray)
        assert features.shape == (8, 768)  # 8 summary tokens, 768 features

    def test_extract_features_batch(self, mock_eegpt_model):
        """Test batch feature extraction."""
        # Create batch of windows
        batch_size = 16
        windows = np.random.randn(batch_size, 19, 1024)

        features = mock_eegpt_model.extract_features_batch(windows)

        assert isinstance(features, np.ndarray)
        assert features.shape == (batch_size, 8, 768)

    def test_extract_features_with_channel_names(self, mock_eegpt_model):
        """Test feature extraction with channel names."""
        window = np.random.randn(19, 1024)
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                   'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                   'Fz', 'Cz', 'Pz']

        features = mock_eegpt_model.extract_features(window, channel_names=ch_names)

        # Should call prepare_chan_ids with channel names
        mock_eegpt_model.encoder.prepare_chan_ids.assert_called_with(ch_names)
        assert features.shape == (8, 768)

    @patch('torch.cuda.is_available', return_value=True)
    def test_extract_features_gpu(self, mock_cuda, mock_eegpt_model):
        """Test feature extraction on GPU."""
        # Set model to GPU mode
        mock_eegpt_model.device = torch.device('cuda')

        window = np.random.randn(19, 1024)
        features = mock_eegpt_model.extract_features(window)

        # Features should still be numpy on CPU
        assert isinstance(features, np.ndarray)
        assert features.shape == (8, 768)


class TestEEGPTPipeline:
    """Test complete EEGPT pipeline."""

    @pytest.fixture
    def sample_eeg_file(self, tmp_path):
        """Create a temporary EEG file."""
        # Create synthetic EEG data
        sfreq = 512
        duration = 30  # 30 seconds
        n_channels = 19

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50

        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                   'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                   'Fz', 'Cz', 'Pz']
        ch_types = ['eeg'] * n_channels

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Save as FIF (faster than EDF for testing)
        fif_path = tmp_path / "test_eeg.fif"
        raw.save(fif_path, overwrite=True)

        return fif_path

    @patch('src.brain_go_brrr.models.eegpt_model.EEGPTModel')
    def test_extract_features_from_raw(self, mock_model_class, sample_eeg_file):
        """Test the high-level feature extraction function."""
        # Setup mock model
        mock_model = Mock()
        mock_model.extract_windows.return_value = [np.random.randn(19, 1024) for _ in range(5)]
        mock_model.extract_features_batch.return_value = np.random.randn(5, 8, 768)
        mock_model_class.return_value = mock_model

        # Load raw data
        raw = mne.io.read_raw_fif(sample_eeg_file, preload=True)

        # Extract features
        with patch('src.brain_go_brrr.models.eegpt_model.Path'):
            results = extract_features_from_raw(raw, "dummy_model_path")

        assert 'features' in results
        assert 'window_times' in results
        assert 'metadata' in results

        # Check shapes
        assert len(results['features']) == 5  # 5 windows
        assert len(results['window_times']) == 5

        # Check metadata
        assert results['metadata']['sampling_rate'] == 256  # After preprocessing
        assert results['metadata']['n_channels'] == 19

    def test_abnormality_prediction_pipeline(self):
        """Test abnormality prediction with mocked model."""
        with patch('src.brain_go_brrr.models.eegpt_model.load_eegpt_encoder'):
            model = EEGPTModel(checkpoint_path=None)

            # Mock the encoder and classifier
            model.encoder = Mock()
            model.encoder.prepare_chan_ids = Mock(return_value=torch.zeros(19))
            model.encoder.forward = Mock(return_value=torch.randn(1, 8, 768))

            # Mock abnormality classifier
            mock_classifier = Mock()
            mock_classifier.forward = Mock(return_value=torch.tensor([[0.2, 0.8]]))  # Normal, Abnormal
            model.task_heads = {'abnormal': mock_classifier}

            # Create test data
            sfreq = 256
            data = np.random.randn(19, sfreq * 20) * 50
            info = mne.create_info(['C3'] * 19, sfreq=sfreq, ch_types=['eeg'] * 19)
            raw = mne.io.RawArray(data, info)

            # Run prediction
            results = model.predict_abnormality(raw)

            assert 'abnormal_probability' in results
            assert 'confidence' in results
            assert 'n_windows' in results
            assert 0 <= results['abnormal_probability'] <= 1
            assert results['confidence'] > 0

    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        with patch('src.brain_go_brrr.models.eegpt_model.load_eegpt_encoder') as mock_load:
            # Make encoder loading fail
            mock_load.side_effect = FileNotFoundError("Model not found")

            # Should handle error gracefully
            model = EEGPTModel(checkpoint_path=Path("nonexistent.ckpt"))

            # Model should still be created but encoder is None
            assert model.encoder is None

            # Create test data
            raw = mne.io.RawArray(np.random.randn(1, 256),
                                 mne.create_info(['C3'], 256, ['eeg']))

            # Should return error result
            results = model.predict_abnormality(raw)
            assert 'error' in results or results['abnormal_probability'] == 0.5


class TestEEGPTConfig:
    """Test EEGPT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EEGPTConfig()

        assert config.n_channels == 58
        assert config.n_samples == 1024
        assert config.patch_size == 64
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12

    def test_custom_config(self):
        """Test custom configuration."""
        config = EEGPTConfig(
            n_channels=32,
            n_samples=512,
            d_model=512
        )

        assert config.n_channels == 32
        assert config.n_samples == 512
        assert config.d_model == 512

        # Other values should be default
        assert config.n_layers == 12

    def test_config_validation(self):
        """Test configuration validation."""
        # Window size must be divisible by patch size
        with pytest.raises(AssertionError):
            EEGPTConfig(n_samples=1000, patch_size=64)  # 1000 not divisible by 64
