"""Improved mock fixtures for EEGPT model testing."""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Optional


class MockEEGPTModel:
    """A more realistic mock for EEGPT model that produces meaningful embeddings."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.is_loaded = True
        self.embedding_dim = 512
        
        # Create a simple but meaningful feature extractor
        # This simulates what EEGPT might learn during pretraining
        self._init_feature_patterns()
        
    def _init_feature_patterns(self):
        """Initialize patterns that distinguish normal from abnormal EEG."""
        # These patterns would be learned by EEGPT during pretraining
        # We're creating simplified versions for testing
        
        # Pattern 1: High frequency content (might indicate muscle artifacts)
        self.high_freq_pattern = np.random.randn(self.embedding_dim).astype(np.float32)
        self.high_freq_pattern[:50] = 2.0  # Strong activation in first 50 dims
        
        # Pattern 2: Spike patterns (might indicate epileptiform activity)
        self.spike_pattern = np.random.randn(self.embedding_dim).astype(np.float32)
        self.spike_pattern[100:150] = -1.5  # Different activation pattern
        
        # Pattern 3: Slow wave patterns (might indicate slowing)
        self.slow_wave_pattern = np.random.randn(self.embedding_dim).astype(np.float32)
        self.slow_wave_pattern[200:250] = 1.8
        
        # Normal pattern (balanced activity)
        self.normal_pattern = np.random.randn(self.embedding_dim).astype(np.float32) * 0.5
        
    def extract_features(self, window_tensor: torch.Tensor) -> np.ndarray:
        """Extract features from EEG window.
        
        This mock analyzes basic statistics of the input and returns
        embeddings that would distinguish different EEG patterns.
        """
        # Convert to numpy for analysis
        window_np = window_tensor.cpu().numpy()
        
        # Analyze window characteristics
        # Shape: (batch, channels, time)
        std_per_channel = np.std(window_np, axis=-1)
        mean_std = np.mean(std_per_channel)
        max_std = np.max(std_per_channel)
        
        # Check for high frequency content (simplified)
        if len(window_np.shape) > 2:
            diff = np.diff(window_np, axis=-1)
            high_freq_score = np.mean(np.abs(diff))
        else:
            high_freq_score = 0
        
        # Create embedding based on characteristics
        embedding = self.normal_pattern.copy()
        
        # Mix in patterns based on window characteristics
        if max_std > 3.0:  # High amplitude
            embedding = 0.5 * embedding + 0.5 * self.high_freq_pattern
        elif mean_std < 0.1:  # Very low amplitude (might be suppression)
            embedding = 0.7 * embedding + 0.3 * self.slow_wave_pattern
        elif high_freq_score > 1.5:  # High frequency content
            embedding = 0.6 * embedding + 0.4 * self.spike_pattern
            
        # Add some noise for realism
        embedding += np.random.randn(self.embedding_dim).astype(np.float32) * 0.1
        
        # Normalize (EEGPT embeddings are typically normalized)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Return with correct shape (batch_size, embedding_dim)
        return embedding.reshape(1, -1)


class MockClassifierHead(nn.Module):
    """A mock classifier head with meaningful weights for testing."""
    
    def __init__(self, input_dim: int = 512, hidden_dim1: int = 256, 
                 hidden_dim2: int = 128, num_classes: int = 2):
        super().__init__()
        
        # Create the same architecture as the real classifier
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, num_classes)
        )
        
        # Initialize with meaningful weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to produce meaningful classifications."""
        with torch.no_grad():
            # Set weights so that certain embedding patterns lead to abnormal classification
            # This simulates a trained classifier
            
            # First layer: detect patterns
            first_layer = self.layers[0]
            # Make first 50 neurons sensitive to high frequency pattern
            first_layer.weight[:50, :50] = 0.1
            # Make next 50 sensitive to spike pattern
            first_layer.weight[50:100, 100:150] = -0.1
            # Make next 50 sensitive to slow waves
            first_layer.weight[100:150, 200:250] = 0.1
            
            # Output layer: combine patterns for classification
            output_layer = self.layers[-1]
            # Normal class (index 0) - low activation from abnormal patterns
            output_layer.weight[0, :50] = -0.1  # Negative weight for high freq
            output_layer.weight[0, 50:100] = -0.1  # Negative weight for spikes
            # Abnormal class (index 1) - high activation from abnormal patterns
            output_layer.weight[1, :50] = 0.1  # Positive weight for high freq
            output_layer.weight[1, 50:100] = 0.1  # Positive weight for spikes
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier."""
        return self.layers(x)


def create_mock_detector_with_realistic_model():
    """Create a detector with properly mocked EEGPT and classifier."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from services.abnormality_detector import AbnormalityDetector
    
    # Create mock model
    mock_model = MockEEGPTModel(device="cpu")
    
    # Create wrapper that behaves like the real model
    model_wrapper = MagicMock()
    model_wrapper.extract_features = mock_model.extract_features
    model_wrapper.is_loaded = True
    
    with patch('services.abnormality_detector.EEGPTModel') as mock_model_class, \
         patch('services.abnormality_detector.ModelConfig'):
        mock_model_class.return_value = model_wrapper
        
        detector = AbnormalityDetector(
            model_path=Path("fake/path.ckpt"),
            device="cpu"
        )
        
        # Replace classifier with our mock classifier
        detector.classifier = MockClassifierHead()
        detector.classifier.eval()
        
        # Set the model
        detector.model = model_wrapper
        
        return detector