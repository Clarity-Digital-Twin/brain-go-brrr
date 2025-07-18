"""Comprehensive unit tests for EEGPTModel class.

Tests all major functionality including model initialization, data processing,
feature extraction, and abnormality prediction. Follows TDD best practices
with extensive mocking to avoid GPU dependencies and ensure fast execution.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from typing import Any

import numpy as np
import pytest
import torch
import mne

from brain_go_brrr.models.eegpt_model import EEGPTConfig, EEGPTModel, preprocess_for_eegpt


class TestEEGPTConfig:
    """Test EEGPTConfig dataclass functionality."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = EEGPTConfig()
        
        assert config.model_size == "large"
        assert config.n_summary_tokens == 4
        assert config.sampling_rate == 256
        assert config.window_duration == 4.0
        assert config.patch_size == 64
        assert config.max_channels == 58

    def test_window_samples_calculation(self):
        """Test window_samples property calculation."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        
        assert config.window_samples == 1024
        
    def test_window_samples_with_invalid_duration(self):
        """Test window_samples raises error with invalid duration."""
        config = EEGPTConfig(sampling_rate=256, window_duration=3.14159)
        
        with pytest.raises(ValueError, match="Window duration must result in integer samples"):
            _ = config.window_samples

    def test_n_patches_per_window_calculation(self):
        """Test n_patches_per_window property calculation."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0, patch_size=64)
        
        assert config.n_patches_per_window == 16  # 1024 / 64

    def test_n_patches_per_window_with_invalid_patch_size(self):
        """Test n_patches_per_window raises error with invalid patch size."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0, patch_size=100)
        
        with pytest.raises(ValueError, match="Patch size must divide window samples evenly"):
            _ = config.n_patches_per_window


class TestEEGPTModelInitialization:
    """Test EEGPTModel initialization functionality."""

    def test_initialization_without_checkpoint(self):
        """Test model initialization without checkpoint."""
        config = EEGPTConfig()
        
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        assert model.checkpoint_path is None
        assert model.config == config
        assert model.encoder is None
        assert model.is_loaded is False
        assert isinstance(model.device, torch.device)

    def test_initialization_with_checkpoint_path_auto_load_false(self):
        """Test model initialization with checkpoint path but no auto-loading."""
        config = EEGPTConfig()
        checkpoint_path = Path("test_checkpoint.ckpt")
        
        model = EEGPTModel(checkpoint_path=checkpoint_path, config=config, auto_load=False)
        
        assert model.checkpoint_path == checkpoint_path
        assert model.config == config
        assert model.encoder is None
        assert model.is_loaded is False

    @patch('brain_go_brrr.models.eegpt_model.create_eegpt_model')
    @patch.object(Path, 'exists', return_value=True)
    def test_initialization_with_auto_load_success(self, mock_exists, mock_create):
        """Test model initialization with auto-loading enabled."""
        mock_encoder = MagicMock()
        mock_create.return_value = mock_encoder
        
        config = EEGPTConfig()
        checkpoint_path = Path("test_checkpoint.ckpt")
        
        model = EEGPTModel(checkpoint_path=checkpoint_path, config=config, auto_load=True)
        
        assert model.is_loaded is True
        assert model.encoder == mock_encoder
        mock_create.assert_called_once()

    def test_initialization_with_specific_device(self):
        """Test model initialization with specific device."""
        device = torch.device("cpu")
        config = EEGPTConfig()
        
        model = EEGPTModel(checkpoint_path=None, config=config, device=device, auto_load=False)
        
        assert model.device == device

    @patch('torch.cuda.is_available', return_value=True)
    def test_device_auto_detection_cuda_available(self, mock_cuda):
        """Test device auto-detection when CUDA is available."""
        config = EEGPTConfig()
        
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        assert model.device.type == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_auto_detection_cuda_unavailable(self, mock_cuda):
        """Test device auto-detection when CUDA is unavailable."""
        config = EEGPTConfig()
        
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        assert model.device.type == "cpu"


class TestEEGPTModelLoading:
    """Test EEGPTModel checkpoint loading functionality."""

    @patch('brain_go_brrr.models.eegpt_model.create_eegpt_model')
    @patch.object(Path, 'exists', return_value=True)
    def test_load_checkpoint_success(self, mock_exists, mock_create):
        """Test successful checkpoint loading."""
        mock_encoder = MagicMock()
        mock_create.return_value = mock_encoder
        
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        checkpoint_path = Path("test_checkpoint.ckpt")
        
        result = model.load_checkpoint(checkpoint_path)
        
        assert result is True
        assert model.is_loaded is True
        assert model.encoder == mock_encoder
        assert model.checkpoint_path == checkpoint_path
        mock_create.assert_called_once_with(
            checkpoint_path=str(checkpoint_path),
            return_all_tokens=False
        )

    def test_load_checkpoint_file_not_exists(self):
        """Test checkpoint loading with non-existent file."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        checkpoint_path = Path("nonexistent_checkpoint.ckpt")
        
        result = model.load_checkpoint(checkpoint_path)
        
        assert result is False
        assert model.is_loaded is False

    @patch('brain_go_brrr.models.eegpt_model.create_eegpt_model', side_effect=Exception("Load error"))
    @patch.object(Path, 'exists', return_value=True)
    def test_load_checkpoint_creation_error(self, mock_exists, mock_create):
        """Test checkpoint loading with creation error."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        checkpoint_path = Path("test_checkpoint.ckpt")
        
        result = model.load_checkpoint(checkpoint_path)
        
        assert result is False
        assert model.is_loaded is False

    @patch('brain_go_brrr.models.eegpt_architecture.EEGTransformer')
    def test_initialize_model_architecture(self, mock_transformer):
        """Test model architecture initialization."""
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance
        
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        model._initialize_model()
        
        mock_transformer.assert_called_once()
        assert model.encoder == mock_transformer_instance
        mock_transformer_instance.to.assert_called_once_with(model.device)
        mock_transformer_instance.eval.assert_called_once()


class TestEEGPTModelWindowExtraction:
    """Test EEGPTModel window extraction functionality."""

    def test_extract_windows_basic(self):
        """Test basic window extraction."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Create data: 2 channels, 8 seconds (2 windows)
        data = np.random.randn(2, 2048)
        sampling_rate = 256
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 2
        assert windows[0].shape == (2, 1024)
        assert windows[1].shape == (2, 1024)

    def test_extract_windows_with_resampling(self):
        """Test window extraction with resampling."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Create data at 512 Hz, 4 seconds (should be resampled to 256 Hz)
        data = np.random.randn(2, 2048)  # 512 Hz * 4 seconds
        sampling_rate = 512
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 1
        assert windows[0].shape == (2, 1024)  # Resampled to 256 Hz

    def test_extract_windows_insufficient_data(self):
        """Test window extraction with insufficient data."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Create data: 2 channels, 2 seconds (insufficient for 1 window)
        data = np.random.randn(2, 512)
        sampling_rate = 256
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 0

    def test_extract_windows_exact_window_size(self):
        """Test window extraction with exact window size."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Create data: 2 channels, exactly 4 seconds (1 window)
        data = np.random.randn(2, 1024)
        sampling_rate = 256
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 1
        assert windows[0].shape == (2, 1024)

    def test_extract_windows_partial_last_window(self):
        """Test window extraction discards partial last window."""
        config = EEGPTConfig(sampling_rate=256, window_duration=4.0)
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Create data: 2 channels, 4.5 seconds (1 full window, partial second)
        data = np.random.randn(2, 1152)  # 1024 + 128 samples
        sampling_rate = 256
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 1  # Only full windows
        assert windows[0].shape == (2, 1024)


class TestEEGPTModelFeatureExtraction:
    """Test EEGPTModel feature extraction functionality."""

    def setup_method(self):
        """Set up mock model for feature extraction tests."""
        self.config = EEGPTConfig()
        self.model = EEGPTModel(checkpoint_path=None, config=self.config, auto_load=False)
        
        # Mock encoder
        self.mock_encoder = MagicMock()
        self.model.encoder = self.mock_encoder
        
        # Mock encoder output: (batch, n_summary_tokens, embed_dim)
        self.mock_features = torch.randn(1, 4, 512)
        self.mock_encoder.return_value = self.mock_features

    def test_extract_features_numpy_input(self):
        """Test feature extraction with numpy input."""
        # Create numpy window: (channels, samples)
        window = np.random.randn(19, 1024)
        
        # Mock prepare_chan_ids
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        
        features = self.model.extract_features(window)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (4, 512)  # n_summary_tokens, embed_dim
        self.mock_encoder.assert_called_once()

    def test_extract_features_tensor_input(self):
        """Test feature extraction with tensor input."""
        # Create tensor window: (channels, samples)
        window = torch.randn(19, 1024)
        
        # Mock prepare_chan_ids
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        
        features = self.model.extract_features(window)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == (4, 512)  # n_summary_tokens, embed_dim
        self.mock_encoder.assert_called_once()

    def test_extract_features_with_channel_names(self):
        """Test feature extraction with channel names."""
        window = np.random.randn(19, 1024)
        channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        
        # Mock prepare_chan_ids
        mock_chan_ids = torch.arange(19)
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=mock_chan_ids)
        
        features = self.model.extract_features(window, channel_names)
        
        self.mock_encoder.prepare_chan_ids.assert_called_once_with(channel_names)
        assert isinstance(features, np.ndarray)

    def test_extract_features_window_padding(self):
        """Test feature extraction with short window requiring padding."""
        # Create short window: (channels, samples) - less than required
        window = np.random.randn(19, 512)  # Half the required size
        
        # Mock prepare_chan_ids
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        
        features = self.model.extract_features(window)
        
        # Verify the encoder was called (padding should happen internally)
        self.mock_encoder.assert_called_once()
        call_args = self.mock_encoder.call_args[0]
        # Check that input was padded to correct size
        assert call_args[0].shape == (1, 1, 19, 1024)

    def test_extract_features_window_cropping(self):
        """Test feature extraction with long window requiring cropping."""
        # Create long window: (channels, samples) - more than required
        window = np.random.randn(19, 2048)  # Double the required size
        
        # Mock prepare_chan_ids
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        
        features = self.model.extract_features(window)
        
        # Verify the encoder was called (cropping should happen internally)
        self.mock_encoder.assert_called_once()
        call_args = self.mock_encoder.call_args[0]
        # Check that input was cropped to correct size
        assert call_args[0].shape == (1, 1, 19, 1024)

    def test_extract_features_no_model_loaded(self):
        """Test feature extraction raises error when no model is loaded."""
        model = EEGPTModel(checkpoint_path=None, config=self.config, auto_load=False)
        window = np.random.randn(19, 1024)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.extract_features(window)

    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        # Create batch of windows: (batch, channels, samples)
        windows = np.random.randn(3, 19, 1024)
        
        # Mock prepare_chan_ids
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        
        # Mock batch output
        self.mock_encoder.return_value = torch.randn(3, 4, 512)
        
        features = self.model.extract_features_batch(windows)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 4, 512)  # batch, n_summary_tokens, embed_dim
        self.mock_encoder.assert_called_once()

    def test_extract_features_batch_with_channel_names(self):
        """Test batch feature extraction with channel names."""
        windows = np.random.randn(2, 19, 1024)
        channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        
        # Mock prepare_chan_ids
        self.mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        
        # Mock batch output
        self.mock_encoder.return_value = torch.randn(2, 4, 512)
        
        features = self.model.extract_features_batch(windows, channel_names)
        
        self.mock_encoder.prepare_chan_ids.assert_called_once_with(channel_names)
        assert features.shape == (2, 4, 512)


class TestEEGPTModelAbnormalityPrediction:
    """Test EEGPTModel abnormality prediction functionality."""

    def setup_method(self):
        """Set up mock model for abnormality prediction tests."""
        self.config = EEGPTConfig()
        self.model = EEGPTModel(checkpoint_path=None, config=self.config, auto_load=False)
        
        # Mock encoder and abnormality head
        self.mock_encoder = MagicMock()
        self.model.encoder = self.mock_encoder
        
        # Mock abnormality head
        self.mock_abnormality_head = MagicMock()
        self.model.abnormality_head = self.mock_abnormality_head
        
        # Mock extract_windows and extract_features methods
        self.model.extract_windows = MagicMock()
        self.model.extract_features = MagicMock()

    @patch('brain_go_brrr.models.eegpt_model.preprocess_for_eegpt')
    def test_predict_abnormality_success(self, mock_preprocess):
        """Test successful abnormality prediction."""
        # Create mock raw data
        mock_raw = MagicMock()
        mock_raw.ch_names = ['Fp1', 'Fp2', 'C3', 'C4']
        mock_raw.info = {'sfreq': 256}
        mock_raw.get_data.return_value = np.random.randn(4, 2048)
        
        # Mock preprocessing
        mock_preprocess.return_value = mock_raw
        
        # Mock window extraction
        mock_windows = [np.random.randn(4, 1024), np.random.randn(4, 1024)]
        self.model.extract_windows.return_value = mock_windows
        
        # Mock feature extraction
        mock_features = np.random.randn(4, 512)  # n_summary_tokens, embed_dim
        self.model.extract_features.return_value = mock_features
        
        # Mock abnormality head output
        mock_logits = torch.tensor([[0.3, 0.7]])  # [normal, abnormal] logits
        self.mock_abnormality_head.return_value = mock_logits
        
        result = self.model.predict_abnormality(mock_raw)
        
        assert 'abnormality_score' in result
        assert 'confidence' in result
        assert 'window_scores' in result
        assert 'n_windows' in result
        assert 'channels_used' in result
        assert result['n_windows'] == 2
        assert result['channels_used'] == ['Fp1', 'Fp2', 'C3', 'C4']

    @patch('brain_go_brrr.models.eegpt_model.preprocess_for_eegpt')
    def test_predict_abnormality_no_windows(self, mock_preprocess):
        """Test abnormality prediction with no valid windows."""
        # Create mock raw data
        mock_raw = MagicMock()
        mock_raw.ch_names = ['Fp1', 'Fp2']
        mock_raw.info = {'sfreq': 256}
        mock_raw.get_data.return_value = np.random.randn(2, 512)  # Too short
        
        # Mock preprocessing
        mock_preprocess.return_value = mock_raw
        
        # Mock empty window extraction
        self.model.extract_windows.return_value = []
        
        result = self.model.predict_abnormality(mock_raw)
        
        assert result['abnormality_score'] == 0.0
        assert result['confidence'] == 0.0
        assert result['window_scores'] == []
        assert result['n_windows'] == 0
        assert 'error' in result

    @patch('brain_go_brrr.models.eegpt_model.preprocess_for_eegpt')
    def test_predict_abnormality_single_window(self, mock_preprocess):
        """Test abnormality prediction with single window."""
        # Create mock raw data
        mock_raw = MagicMock()
        mock_raw.ch_names = ['Fp1', 'Fp2']
        mock_raw.info = {'sfreq': 256}
        mock_raw.get_data.return_value = np.random.randn(2, 1024)
        
        # Mock preprocessing
        mock_preprocess.return_value = mock_raw
        
        # Mock single window extraction
        mock_windows = [np.random.randn(2, 1024)]
        self.model.extract_windows.return_value = mock_windows
        
        # Mock feature extraction
        mock_features = np.random.randn(4, 512)
        self.model.extract_features.return_value = mock_features
        
        # Mock abnormality head output
        mock_logits = torch.tensor([[0.2, 0.8]])  # High abnormal probability
        self.mock_abnormality_head.return_value = mock_logits
        
        result = self.model.predict_abnormality(mock_raw)
        
        assert result['n_windows'] == 1
        assert result['confidence'] == 0.8  # Default for single window
        assert len(result['window_scores']) == 1


class TestEEGPTModelRecordingProcessing:
    """Test EEGPTModel recording processing functionality."""

    def setup_method(self):
        """Set up mock model for recording processing tests."""
        self.config = EEGPTConfig()
        self.model = EEGPTModel(checkpoint_path=None, config=self.config, auto_load=False)
        
        # Mock methods
        self.model.extract_windows = MagicMock()
        self.model.extract_features_batch = MagicMock()

    def test_process_recording_basic(self):
        """Test basic recording processing."""
        # Create mock data: 19 channels, 8 seconds (2 windows)
        data = np.random.randn(19, 2048)
        sampling_rate = 256
        
        # Mock window extraction
        mock_windows = [np.random.randn(19, 1024), np.random.randn(19, 1024)]
        self.model.extract_windows.return_value = mock_windows
        
        # Mock batch feature extraction
        mock_features = np.random.randn(2, 4, 512)  # batch, n_summary_tokens, embed_dim
        self.model.extract_features_batch.return_value = mock_features
        
        result = self.model.process_recording(data, sampling_rate)
        
        assert result['n_windows'] == 2
        assert result['processing_complete'] is True
        assert len(result['features']) == 2  # Two windows worth of features
        self.model.extract_windows.assert_called_once_with(data, sampling_rate)

    def test_process_recording_with_batch_size(self):
        """Test recording processing with custom batch size."""
        # Create mock data for 3 windows
        data = np.random.randn(19, 3072)  # 3 * 1024 samples
        sampling_rate = 256
        batch_size = 2
        
        # Mock window extraction
        mock_windows = [np.random.randn(19, 1024) for _ in range(3)]
        self.model.extract_windows.return_value = mock_windows
        
        # Mock batch feature extraction (called twice: batch of 2, then batch of 1)
        self.model.extract_features_batch.side_effect = [
            np.random.randn(2, 4, 512),  # First batch
            np.random.randn(1, 4, 512)   # Second batch
        ]
        
        result = self.model.process_recording(data, sampling_rate, batch_size=batch_size)
        
        assert result['n_windows'] == 3
        assert result['processing_complete'] is True
        assert len(result['features']) == 3
        assert self.model.extract_features_batch.call_count == 2

    def test_process_recording_empty_windows(self):
        """Test recording processing with no valid windows."""
        data = np.random.randn(19, 512)  # Too short for any windows
        sampling_rate = 256
        
        # Mock empty window extraction
        self.model.extract_windows.return_value = []
        
        result = self.model.process_recording(data, sampling_rate)
        
        assert result['n_windows'] == 0
        assert result['processing_complete'] is True
        assert result['features'] == []
        self.model.extract_features_batch.assert_not_called()

    def test_process_recording_large_batch_size(self):
        """Test recording processing with batch size larger than windows."""
        data = np.random.randn(19, 1024)  # 1 window
        sampling_rate = 256
        batch_size = 10
        
        # Mock single window extraction
        mock_windows = [np.random.randn(19, 1024)]
        self.model.extract_windows.return_value = mock_windows
        
        # Mock batch feature extraction
        mock_features = np.random.randn(1, 4, 512)
        self.model.extract_features_batch.return_value = mock_features
        
        result = self.model.process_recording(data, sampling_rate, batch_size=batch_size)
        
        assert result['n_windows'] == 1
        assert len(result['features']) == 1
        self.model.extract_features_batch.assert_called_once()


class TestEEGPTModelEdgeCases:
    """Test EEGPTModel edge cases and error handling."""

    def test_cleanup_cuda_device(self):
        """Test GPU memory cleanup for CUDA device."""
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            config = EEGPTConfig()
            model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
            model.device = torch.device("cuda")
            
            model.cleanup()
            
            mock_empty_cache.assert_called_once()

    def test_cleanup_cpu_device(self):
        """Test cleanup for CPU device (should not call CUDA functions)."""
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            config = EEGPTConfig()
            model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
            model.device = torch.device("cpu")
            
            model.cleanup()
            
            mock_empty_cache.assert_not_called()

    def test_extract_features_3d_input(self):
        """Test feature extraction with 3D input (batch dimension)."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Mock encoder
        mock_encoder = MagicMock()
        model.encoder = mock_encoder
        mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        mock_encoder.return_value = torch.randn(1, 4, 512)
        
        # 3D input: (batch, channels, samples)
        window = np.random.randn(1, 19, 1024)
        
        features = model.extract_features(window)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (4, 512)

    def test_extract_features_single_channel(self):
        """Test feature extraction with single channel."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Mock encoder
        mock_encoder = MagicMock()
        model.encoder = mock_encoder
        mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(1))
        mock_encoder.return_value = torch.randn(1, 4, 512)
        
        # Single channel input
        window = np.random.randn(1, 1024)
        
        features = model.extract_features(window)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (4, 512)

    def test_extract_windows_single_sample(self):
        """Test window extraction with very short data."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Single sample
        data = np.random.randn(19, 1)
        sampling_rate = 256
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 0

    def test_extract_windows_empty_data(self):
        """Test window extraction with empty data."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Empty data
        data = np.random.randn(19, 0)
        sampling_rate = 256
        
        windows = model.extract_windows(data, sampling_rate)
        
        assert len(windows) == 0


class TestPreprocessForEEGPT:
    """Test preprocess_for_eegpt function."""

    def test_preprocess_for_eegpt_basic(self, mock_eeg_data):
        """Test basic preprocessing functionality."""
        raw = mock_eeg_data.copy()
        
        processed = preprocess_for_eegpt(raw)
        
        assert processed.info['sfreq'] == 256
        assert len(processed.ch_names) <= 58
        # Original should be unchanged
        assert raw.info['sfreq'] == 256  # mock_eeg_data already at 256 Hz

    @patch('mne.io.Raw.resample')
    def test_preprocess_for_eegpt_resampling(self, mock_resample, mock_eeg_data):
        """Test preprocessing with resampling."""
        raw = mock_eeg_data.copy()
        raw.info['sfreq'] = 512  # Different sampling rate
        
        processed = preprocess_for_eegpt(raw)
        
        mock_resample.assert_called_once_with(256)

    def test_preprocess_for_eegpt_too_many_channels(self):
        """Test preprocessing with too many channels."""
        # Create data with more than 58 channels
        ch_names = [f'CH{i:02d}' for i in range(60)]
        data = np.random.randn(60, 7680)  # 30 seconds at 256 Hz
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        processed = preprocess_for_eegpt(raw)
        
        assert len(processed.ch_names) == 58

    def test_preprocess_for_eegpt_preserves_original(self, mock_eeg_data):
        """Test that preprocessing doesn't modify original data."""
        original_sfreq = mock_eeg_data.info['sfreq']
        original_ch_names = mock_eeg_data.ch_names.copy()
        
        processed = preprocess_for_eegpt(mock_eeg_data)
        
        # Original should be unchanged
        assert mock_eeg_data.info['sfreq'] == original_sfreq
        assert mock_eeg_data.ch_names == original_ch_names
        
        # Processed should be different object
        assert processed is not mock_eeg_data


@pytest.mark.benchmark
class TestEEGPTModelPerformance:
    """Performance benchmarks for EEGPTModel (marked for optional execution)."""

    def test_extract_windows_performance(self, benchmark):
        """Benchmark window extraction performance."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Large data: 19 channels, 10 minutes at 256 Hz
        data = np.random.randn(19, 153600)
        sampling_rate = 256
        
        result = benchmark(model.extract_windows, data, sampling_rate)
        
        # Should extract 150 windows (10 minutes / 4 seconds)
        assert len(result) == 150

    def test_extract_features_performance_mock(self, benchmark):
        """Benchmark feature extraction performance with mocked encoder."""
        config = EEGPTConfig()
        model = EEGPTModel(checkpoint_path=None, config=config, auto_load=False)
        
        # Mock encoder for performance test
        mock_encoder = MagicMock()
        mock_encoder.prepare_chan_ids = MagicMock(return_value=torch.arange(19))
        mock_encoder.return_value = torch.randn(1, 4, 512)
        model.encoder = mock_encoder
        
        window = np.random.randn(19, 1024)
        
        result = benchmark(model.extract_features, window)
        
        assert result.shape == (4, 512)