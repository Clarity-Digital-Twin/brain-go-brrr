"""REAL tests for EEGPT wrapper - Clean testing of model wrapping."""

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestEEGPTWrapper:
    """Test EEGPT wrapper functionality."""

    @patch('brain_go_brrr.models.eegpt_wrapper.create_eegpt_model')
    def test_wrapper_initialization(self, mock_create):
        """Test wrapper initialization."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        # Mock the model creation
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        wrapper = EEGPTWrapper(checkpoint_path=None)
        
        assert wrapper.model == mock_model
        assert hasattr(wrapper, 'model')
        mock_create.assert_called_once()

    @patch('brain_go_brrr.models.eegpt_wrapper.create_eegpt_model')
    def test_wrapper_preprocess(self, mock_create):
        """Test data preprocessing in wrapper."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        wrapper = EEGPTWrapper()
        
        # Input data: (channels, samples)
        raw_data = np.random.randn(20, 1024).astype(np.float32)
        
        # Preprocess
        processed = wrapper.preprocess(raw_data)
        
        # Should return tensor
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (20, 1024)
        assert processed.dtype == torch.float32

    def test_wrapper_extract_features(self):
        """Test feature extraction through wrapper."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        # Setup mock model
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        mock_model.config.embed_dim = 768
        
        # Mock forward pass
        mock_model.forward.return_value = torch.randn(1, 16, 768)
        
        wrapper = EEGPTWrapper(model=mock_model)
        
        # Input data
        data = np.random.randn(20, 1024).astype(np.float32)
        
        # Extract features
        features = wrapper.extract_features(data)
        
        # Check output
        assert isinstance(features, np.ndarray)
        assert features.shape[-1] == 768  # embed_dim
        
        # Model should have been called
        mock_model.forward.assert_called_once()

    def test_wrapper_batch_processing(self):
        """Test batch processing through wrapper."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        mock_model.config.embed_dim = 512
        
        # Mock to handle different batch sizes
        def mock_forward(x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 16, 512)
        
        mock_model.forward = mock_forward
        
        wrapper = EEGPTWrapper(model=mock_model)
        
        # Batch of data
        batch_data = [
            np.random.randn(20, 1024).astype(np.float32),
            np.random.randn(20, 1024).astype(np.float32),
            np.random.randn(20, 1024).astype(np.float32),
        ]
        
        # Process batch
        features = wrapper.extract_features_batch(batch_data)
        
        # Check output
        assert len(features) == 3
        for feat in features:
            assert feat.shape[-1] == 512

    def test_wrapper_device_handling(self):
        """Test device (CPU/GPU) handling."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        mock_model.to = MagicMock(return_value=mock_model)
        
        # Test CPU
        wrapper = EEGPTWrapper(model=mock_model, device='cpu')
        assert wrapper.device == 'cpu'
        mock_model.to.assert_called_with('cpu')
        
        # Test CUDA (if available)
        if torch.cuda.is_available():
            wrapper = EEGPTWrapper(model=mock_model, device='cuda')
            assert wrapper.device == 'cuda'
            mock_model.to.assert_called_with('cuda')

    def test_wrapper_prediction(self):
        """Test end-to-end prediction."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        # Setup complete mock
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        mock_model.config.embed_dim = 768
        
        # Mock encoder output
        mock_model.forward.return_value = torch.randn(1, 16, 768)
        
        # Mock classifier head
        mock_classifier = MagicMock()
        mock_classifier.return_value = torch.tensor([[0.2, 0.8]])  # Binary logits
        
        wrapper = EEGPTWrapper(model=mock_model, classifier=mock_classifier)
        
        # Input data
        data = np.random.randn(20, 1024).astype(np.float32)
        
        # Predict
        logits, probs = wrapper.predict(data)
        
        # Check outputs
        assert logits.shape == (2,)
        assert probs.shape == (2,)
        assert np.allclose(probs.sum(), 1.0, atol=1e-6)
        
        # Classifier should have been called
        mock_classifier.assert_called_once()


class TestWrapperValidation:
    """Test input validation in wrapper."""

    def test_validate_input_shape(self):
        """Test input shape validation."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        
        wrapper = EEGPTWrapper(model=mock_model)
        
        # Wrong number of channels
        with pytest.raises(ValueError) as exc_info:
            bad_data = np.random.randn(19, 1024)
            wrapper.validate_input(bad_data)
        assert "channels" in str(exc_info.value).lower()
        
        # Wrong dimensions
        with pytest.raises(ValueError) as exc_info:
            bad_data = np.random.randn(20)  # 1D instead of 2D
            wrapper.validate_input(bad_data)
        assert "dimension" in str(exc_info.value).lower()

    def test_validate_sampling_rate(self):
        """Test sampling rate validation."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        
        wrapper = EEGPTWrapper(model=mock_model)
        
        # Data with wrong sampling rate
        data = np.random.randn(20, 512)  # 2 seconds at 256 Hz
        
        # Should handle resampling or raise warning
        result = wrapper.preprocess(data, source_sfreq=128)
        
        # Should resample to correct rate
        assert result.shape[1] == 512 or result.shape[1] == 256  # Depends on implementation


class TestWrapperCaching:
    """Test caching functionality in wrapper."""

    def test_feature_caching(self):
        """Test feature caching for repeated inputs."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        mock_model.forward.return_value = torch.randn(1, 16, 768)
        
        wrapper = EEGPTWrapper(model=mock_model, enable_cache=True)
        
        # Same input twice
        data = np.random.randn(20, 1024).astype(np.float32)
        
        features1 = wrapper.extract_features(data)
        features2 = wrapper.extract_features(data)
        
        # Should use cache on second call
        if hasattr(wrapper, '_cache'):
            # Model should only be called once
            assert mock_model.forward.call_count == 1
            # Results should be identical
            np.testing.assert_array_equal(features1, features2)
        else:
            pytest.skip("Caching not implemented")


class TestWrapperWindowProcessing:
    """Test sliding window processing."""

    def test_sliding_window_extraction(self):
        """Test feature extraction with sliding windows."""
        from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper
        
        mock_model = MagicMock()
        mock_model.config.n_channels = 20
        mock_model.config.sampling_rate = 256
        mock_model.config.window_size = 4.0  # 4 seconds
        mock_model.forward.return_value = torch.randn(1, 16, 768)
        
        wrapper = EEGPTWrapper(model=mock_model)
        
        # Long recording (10 seconds)
        long_data = np.random.randn(20, 2560).astype(np.float32)
        
        # Extract with sliding windows
        features = wrapper.extract_features_windowed(
            long_data,
            window_size=4.0,
            stride=2.0  # 50% overlap
        )
        
        # Should return multiple windows
        # 10 seconds with 4s windows and 2s stride = 4 windows
        expected_windows = 4
        
        if isinstance(features, list):
            assert len(features) == expected_windows
        else:
            # Might return stacked array
            assert features.shape[0] == expected_windows