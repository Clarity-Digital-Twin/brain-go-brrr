"""Test doubles (fakes) for clean, maintainable tests.

These fakes provide lightweight, behavior-focused alternatives to heavy mocking.
They implement just enough functionality to test the behavior we care about,
without coupling tests to implementation details.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class FakeEEGPTBackbone:
    """Lightweight fake for EEGPT backbone model."""
    
    def __init__(self, feature_dim: int = 2048, n_summary_tokens: int = 1):
        self.feature_dim = feature_dim
        self._n_summary_tokens = n_summary_tokens
        self.is_loaded = True
        self.config = type('Config', (), {'model_path': Path('/fake/model.ckpt')})()
    
    @property
    def n_summary_tokens(self) -> int:
        return self._n_summary_tokens
    
    def extract_features(self, data: np.ndarray, channel_names: Optional[List[str]] = None) -> np.ndarray:
        """Return consistent fake features for testing."""
        batch_size = data.shape[0] if len(data.shape) > 1 else 1
        return np.ones((batch_size, self.feature_dim), dtype=np.float32) * 0.5
    
    def to(self, device: str):
        """Fake device movement."""
        return self


class FakeClassifierHead(nn.Module):
    """Lightweight fake classifier for abnormality detection."""
    
    def __init__(self, input_dim: int = 2048, n_classes: int = 2, deterministic: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.deterministic = deterministic
        # Simple linear layer for actual forward pass
        self.linear = nn.Linear(input_dim, n_classes)
        if deterministic:
            # Set weights to produce predictable outputs
            with torch.no_grad():
                self.linear.weight.fill_(0.1)
                self.linear.bias.fill_(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return consistent predictions for testing."""
        if self.deterministic:
            # Return slightly different values for binary classification
            batch_size = x.shape[0]
            # Normal class gets 0.3, abnormal gets 0.7 confidence
            return torch.tensor([[0.7, 0.3]] * batch_size, dtype=torch.float32)
        return self.linear(x)


class FakeRedis:
    """In-memory fake Redis for testing cache behavior."""
    
    def __init__(self):
        self.storage: Dict[str, bytes] = {}
        self.call_count = {'get': 0, 'set': 0, 'delete': 0, 'exists': 0}
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value from fake storage."""
        self.call_count['get'] += 1
        return self.storage.get(key)
    
    def set(self, key: str, value: bytes, ex: Optional[int] = None) -> bool:
        """Set value in fake storage."""
        self.call_count['set'] += 1
        self.storage[key] = value
        return True
    
    def delete(self, key: str) -> int:
        """Delete key from fake storage."""
        self.call_count['delete'] += 1
        if key in self.storage:
            del self.storage[key]
            return 1
        return 0
    
    def exists(self, key: str) -> int:
        """Check if key exists."""
        self.call_count['exists'] += 1
        return 1 if key in self.storage else 0
    
    def flushdb(self) -> bool:
        """Clear all storage."""
        self.storage.clear()
        return True


class FakeEdfReader:
    """Fake EDF reader for testing without real files."""
    
    def __init__(self, n_channels: int = 19, n_samples: int = 10000, sfreq: float = 256.0):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sfreq = sfreq
        self.is_open = False
        self.channel_labels = [f"EEG{i}" for i in range(n_channels)]
    
    def __enter__(self):
        self.is_open = True
        return self
    
    def __exit__(self, *args):
        self.is_open = False
    
    def getNSamples(self) -> List[int]:
        """Return sample counts per channel."""
        return [self.n_samples] * self.n_channels
    
    def getSampleFrequency(self, channel: Optional[int] = None) -> List[float]:
        """Return sampling frequency."""
        if channel is not None:
            return self.sfreq
        return [self.sfreq] * self.n_channels
    
    def getSignalLabels(self) -> List[str]:
        """Return channel labels."""
        return self.channel_labels
    
    def readSignal(self, channel: int, start: int = 0, n: Optional[int] = None) -> np.ndarray:
        """Return fake signal data."""
        if n is None:
            n = self.n_samples - start
        # Return realistic EEG-like data (small amplitude noise)
        return np.random.randn(n) * 10  # ~10 μV amplitude


class FakeFeatureExtractor:
    """Fake feature extractor for pipeline tests."""
    
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
        self.device = "cpu"
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract fake features."""
        batch_size = len(data) if isinstance(data, list) else data.shape[0]
        return np.zeros((batch_size, self.feature_dim), dtype=np.float32)
    
    def extract_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Alias for extract."""
        return self.extract(data)
    
    def extract_embeddings_with_metadata(self, raw: Any) -> Dict[str, Any]:
        """Extract embeddings with metadata for pipeline."""
        # Fake some embeddings
        n_windows = 10
        return {
            "embeddings": np.zeros((n_windows, self.feature_dim), dtype=np.float32),
            "window_times": np.arange(n_windows) * 4.0,
            "metadata": {
                "n_windows": n_windows,
                "feature_dim": self.feature_dim,
                "sampling_rate": 256.0
            }
        }


class FakeSleepAnalyzer:
    """Fake sleep analyzer for pipeline tests."""
    
    def __init__(self):
        self.yasa_version = "0.6.0"
    
    def run_full_sleep_analysis(self, raw: Any) -> Dict[str, Any]:
        """Return fake sleep analysis results."""
        return {
            'hypnogram': ['W', 'N1', 'N2', 'N3', 'N2', 'REM'],
            'sleep_efficiency': 85.5,
            'total_sleep_time': 420,  # minutes
            'sleep_stages': {
                'W': 15.0,
                'N1': 5.0,
                'N2': 45.0,
                'N3': 20.0,
                'REM': 15.0
            }
        }
    
    def stage_sleep(self, raw: Any, **kwargs) -> Dict[str, Any]:
        """Stage sleep for pipeline compatibility."""
        return self.run_full_sleep_analysis(raw)
    
    def predict_proba(self, raw: Any) -> np.ndarray:
        """Return fake sleep stage probabilities."""
        n_epochs = 100  # Fake number of epochs
        # 5 stages: W, N1, N2, N3, REM
        proba = np.random.rand(n_epochs, 5)
        # Normalize to sum to 1
        return proba / proba.sum(axis=1, keepdims=True)


class FakeMNERaw:
    """Minimal fake for MNE Raw object."""
    
    def __init__(self, n_channels: int = 19, duration: float = 20.0, sfreq: float = 256.0):
        self.n_channels = n_channels
        self.duration = duration
        self.ch_names = [f'EEG{i}' for i in range(n_channels)]  # Add as attribute
        self.info = {'sfreq': sfreq, 'ch_names': self.ch_names}
        self.times = np.arange(0, duration, 1/sfreq)
        self._data = np.random.randn(n_channels, len(self.times)) * 1e-6  # μV scale
    
    def get_data(self, picks=None, start=0, stop=None) -> np.ndarray:
        """Return fake EEG data."""
        if stop is None:
            stop = len(self.times)
        if picks is None:
            return self._data[:, start:stop]
        return self._data[picks, start:stop]
    
    def copy(self):
        """Return a copy of self."""
        return FakeMNERaw(self.n_channels, self.duration, self.info['sfreq'])
    
    def filter(self, l_freq, h_freq, **kwargs):
        """Fake filtering."""
        return self
    
    def resample(self, sfreq, **kwargs):
        """Fake resampling."""
        self.info['sfreq'] = sfreq
        return self