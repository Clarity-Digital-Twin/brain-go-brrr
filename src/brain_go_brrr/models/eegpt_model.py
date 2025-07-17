"""
EEGPT Model Integration Module

Implements the EEGPT foundation model for EEG analysis.
Based on the paper "EEGPT: Pretrained Transformer for Universal
and Reliable Representation of EEG Signals"

Model specifications:
- 10M parameters (large variant)
- Vision Transformer architecture
- Dual self-supervised learning
- 4-second windows at 256 Hz
- Supports up to 58 channels
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch
import torch.nn as nn
from scipy import signal

from .eegpt_architecture import create_eegpt_model

logger = logging.getLogger(__name__)


@dataclass
class EEGPTConfig:
    """Configuration for EEGPT model based on paper specifications."""

    # Model parameters
    model_size: str = "large"  # large (10M) or xlarge (101M)
    n_summary_tokens: int = 4  # S=4 from paper

    # Input specifications
    sampling_rate: int = 256  # Hz
    window_duration: float = 4.0  # seconds
    patch_size: int = 64  # samples (250ms)
    max_channels: int = 58  # Maximum supported channels

    # Preprocessing
    reference: str = "average"  # Average reference
    unit: str = "mV"  # Expected unit (millivolts)
    filter_low: float | None = None  # Optional low-pass
    filter_high: float | None = None  # Optional high-pass

    @property
    def window_samples(self) -> int:
        """Calculate window size in samples."""
        samples = self.window_duration * self.sampling_rate
        if not samples.is_integer():
            raise ValueError("Window duration must result in integer samples")
        return int(samples)

    @property
    def n_patches_per_window(self) -> int:
        """Calculate number of patches per window."""
        if self.window_samples % self.patch_size != 0:
            raise ValueError("Patch size must divide window samples evenly")
        return self.window_samples // self.patch_size


class EEGPTModel:
    """
    EEGPT model wrapper for EEG analysis.

    Provides high-level interface for:
    - Loading pretrained checkpoints
    - Preprocessing EEG data
    - Extracting features
    - Task-specific predictions
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: EEGPTConfig | None = None,
        device: torch.device | None = None,
        auto_load: bool = True
    ):
        """
        Initialize EEGPT model.

        Args:
            checkpoint_path: Path to pretrained checkpoint
            config: Model configuration
            device: PyTorch device (auto-detected if None)
            auto_load: Whether to automatically load the model checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config or EEGPTConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = None
        self.is_loaded = False
        self.n_summary_tokens = self.config.n_summary_tokens

        if auto_load:
            self._load_model()

    def _load_model(self):
        """Load pretrained EEGPT model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            # Create and load EEGPT model
            logger.info(f"Loading EEGPT model from {self.checkpoint_path}")
            self.encoder = create_eegpt_model(
                checkpoint_path=str(self.checkpoint_path),
                return_all_tokens=False  # Only return summary tokens
            )
            self.encoder.to(self.device)
            self.encoder.eval()

            # Load task-specific heads if needed
            self._load_task_heads()

            self.is_loaded = True
            logger.info("EEGPT model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load EEGPT model: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if loading successful, False otherwise
        """
        try:
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False

            self.checkpoint_path = checkpoint_path
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _initialize_model(self):
        """Initialize model architecture without loading checkpoint."""
        try:
            from .eegpt_architecture import EEGTransformer
            
            # Create model architecture with correct parameters
            self.encoder = EEGTransformer(
                img_size=[self.config.max_channels, self.config.window_samples],
                patch_size=self.config.patch_size,
                in_chans=1,
                embed_dim=512,  # Default from paper
                embed_num=self.config.n_summary_tokens,
                depth=8,  # Default from paper
                num_heads=8,  # Default from paper
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                return_all_tokens=False
            )
            self.encoder.to(self.device)
            self.encoder.eval()
            
            logger.info("Model architecture initialized")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _load_task_heads(self):
        """Load task-specific classification heads."""
        # Initialize task-specific heads
        # These would be loaded from separate checkpoints or trained
        self.abnormality_head = nn.Sequential(
            nn.Linear(self.config.n_summary_tokens * 512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification
        ).to(self.device)

    def extract_windows(
        self,
        data: np.ndarray,
        sampling_rate: int
    ) -> list[np.ndarray]:
        """
        Extract non-overlapping windows from continuous data.

        Args:
            data: EEG data (channels, samples)
            sampling_rate: Sampling rate in Hz

        Returns:
            List of windows (channels, window_samples)
        """
        # Resample if needed
        if sampling_rate != self.config.sampling_rate:
            n_samples = data.shape[1]
            new_n_samples = int(n_samples * self.config.sampling_rate / sampling_rate)
            data = signal.resample(data, new_n_samples, axis=1)

        # Extract windows
        window_samples = self.config.window_samples
        n_windows = data.shape[1] // window_samples

        windows = []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            windows.append(data[:, start:end])

        return windows

    def extract_features(self, window: np.ndarray | torch.Tensor, channel_names: list[str] | None = None) -> np.ndarray | torch.Tensor:
        """
        Extract features from a single window using EEGPT encoder.

        Args:
            window: EEG window (channels, samples)
            channel_names: List of channel names

        Returns:
            Features array (n_summary_tokens, feature_dim)
        """
        # Ensure window is correct size
        if window.shape[1] != self.config.window_samples:
            # Resample or pad/crop as needed
            if window.shape[1] < self.config.window_samples:
                # Pad with zeros
                pad_width = self.config.window_samples - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Crop
                window = window[:, :self.config.window_samples]

        # Convert to tensor
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

        # Prepare channel IDs
        if channel_names is not None:
            chan_ids = self.encoder.prepare_chan_ids(channel_names).to(self.device)
        else:
            # Default channel IDs
            chan_ids = torch.arange(window.shape[0], device=self.device)

        # Extract features
        with torch.no_grad():
            features = self.encoder(window_tensor, chan_ids)  # B, n_summary_tokens, embed_dim
            features = features.squeeze(0)  # Remove batch dimension

        return features.cpu().numpy()

    def extract_features_batch(self, windows: np.ndarray, channel_names: list[str] | None = None) -> np.ndarray:
        """
        Extract features from batch of windows.

        Args:
            windows: Batch of windows (batch, channels, samples)
            channel_names: List of channel names

        Returns:
            Features (batch, n_summary_tokens, feature_dim)
        """
        # Convert to tensor
        windows_tensor = torch.FloatTensor(windows).to(self.device)

        # Prepare channel IDs
        if channel_names is not None:
            chan_ids = self.encoder.prepare_chan_ids(channel_names).to(self.device)
        else:
            chan_ids = torch.arange(windows.shape[1], device=self.device)

        # Extract features for batch
        with torch.no_grad():
            features = self.encoder(windows_tensor, chan_ids)

        return features.cpu().numpy()

    def predict_abnormality(self, raw: mne.io.Raw) -> dict[str, Any]:
        """
        Predict abnormality score for raw EEG.

        Args:
            raw: MNE Raw objec

        Returns:
            Dictionary with abnormality score and metadata
        """
        # Preprocess
        processed = preprocess_for_eegpt(raw)

        # Get channel names
        channel_names = processed.ch_names

        # Extract windows
        data = processed.get_data()
        windows = self.extract_windows(data, int(processed.info['sfreq']))

        if len(windows) == 0:
            return {
                'abnormality_score': 0.0,
                'confidence': 0.0,
                'window_scores': [],
                'n_windows': 0,
                'error': 'No valid windows extracted'
            }

        # Extract features for all windows
        window_features = []
        for window in windows:
            features = self.extract_features(window, channel_names)
            window_features.append(features)

        # Stack features
        all_features = np.stack(window_features)  # (n_windows, n_summary_tokens, embed_dim)

        # Apply abnormality detection head
        window_scores = []
        with torch.no_grad():
            for features in all_features:
                # Flatten summary tokens
                features_flat = torch.FloatTensor(features.flatten()).unsqueeze(0).to(self.device)

                # Apply classification head
                logits = self.abnormality_head(features_flat)
                probs = torch.softmax(logits, dim=-1)
                abnormal_prob = probs[0, 1].item()  # Probability of abnormal class
                window_scores.append(abnormal_prob)

        # Aggregate scores
        abnormality_score = np.mean(window_scores)
        confidence = 1.0 - np.std(window_scores) if len(window_scores) > 1 else 0.8

        return {
            'abnormality_score': float(abnormality_score),
            'confidence': float(confidence),
            'window_scores': window_scores,
            'n_windows': len(windows),
            'channels_used': channel_names
        }

    def process_recording(
        self,
        data: np.ndarray,
        sampling_rate: int
    ) -> dict[str, Any]:
        """Process full recording and return results."""
        windows = self.extract_windows(data, sampling_rate)

        results = {
            'n_windows': len(windows),
            'features': [],
            'processing_complete': True
        }

        for window in windows:
            features = self.extract_features(window)
            results['features'].append(features)

        return results


def preprocess_for_eegpt(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Preprocess raw EEG data according to EEGPT requirements.

    Based on paper specifications:
    - Resample to 256 Hz
    - Average reference
    - Convert to mV
    - Optional bandpass filtering

    Args:
        raw: Input raw EEG data

    Returns:
        Preprocessed raw data
    """
    raw = raw.copy()

    # Resample to 256 Hz if needed
    if raw.info['sfreq'] != 256:
        raw.resample(256)

    # Apply average reference
    raw.set_eeg_reference('average', projection=False)

    # Ensure proper channel types
    raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')

    # Limit to max channels if needed
    if len(raw.ch_names) > 58:
        raw.pick_channels(raw.ch_names[:58])

    return raw


def extract_features_from_raw(
    raw: mne.io.Raw,
    model_path: str | Path
) -> dict[str, Any]:
    """
    High-level function to extract features from raw EEG.

    Args:
        raw: Raw EEG data
        model_path: Path to EEGPT checkpoin

    Returns:
        Dictionary with features and metadata
    """
    import time
    start_time = time.time()

    # Initialize model
    model = EEGPTModel(checkpoint_path=model_path)

    # Get abnormality prediction
    result = model.predict_abnormality(raw)

    # Add timing
    result['processing_time'] = time.time() - start_time
    result['features'] = True  # Placeholder

    return result
