"""EEGPT Model Integration Module.

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
import numpy.typing as npt
import torch
import torch.nn as nn
from scipy import signal

from ..core.config import ModelConfig
from .eegpt_architecture import EEGTransformer, create_eegpt_model

logger = logging.getLogger(__name__)


@dataclass
class EEGPTConfig:
    """Configuration for EEGPT model based on paper specifications."""

    # Model parameters
    model_size: str = "large"  # large (10M) or xlarge (101M)
    n_summary_tokens: int = 4  # S=4 from paper
    embed_dim: int = 512  # Embedding dimension

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
    """EEGPT Model wrapper for inference and feature extraction."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize EEGPT model.

        Args:
            config: Model configuration (preferred)
            checkpoint_path: Backward compatibility - path to checkpoint
            device: Backward compatibility - device to use
            auto_load: Backward compatibility - whether to auto-load model
        """
        # Handle backward compatibility
        if config is None and checkpoint_path is not None:
            # Create config from legacy parameters
            config = ModelConfig()
            if checkpoint_path:
                config.model_path = Path(checkpoint_path)
            if device:
                config.device = device

        self.config = config or ModelConfig()

        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.encoder: EEGTransformer | None = None
        self.abnormality_head: nn.Module | None = None
        self.is_loaded = False

        # Performance optimization: cache channel index mappings
        self._ch_idx_cache: dict[tuple[str, ...], torch.Tensor] = {}

        self.logger = logging.getLogger(__name__)

        # Auto-load for backward compatibility
        if auto_load and self.config.model_path.exists():
            try:
                self.load_model()
            except Exception as e:
                self.logger.warning(f"Auto-load failed: {e}")

    def _get_cached_channel_ids(self, channel_names: list[str]) -> torch.Tensor:
        """Get channel IDs with caching for performance."""
        key = tuple(channel_names)
        if key not in self._ch_idx_cache:
            if self.encoder is None:
                raise RuntimeError("Encoder not loaded")
            # Create and cache the channel IDs
            chan_ids = self.encoder.prepare_chan_ids(channel_names)
            self._ch_idx_cache[key] = chan_ids.to(self.device)
        return self._ch_idx_cache[key]

    @property
    def n_summary_tokens(self) -> int:
        """Backward compatibility property."""
        return self.config.n_summary_tokens

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from config."""
        return self.config.embed_dim

    @property
    def checkpoint_path(self) -> Path:
        """Backward compatibility property."""
        return self.config.model_path

    @property
    def window_samples(self) -> int:
        """Backward compatibility property."""
        return self.config.window_samples

    def load_model(self) -> None:
        """Load the EEGPT model from checkpoint."""
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.config.model_path}")

        # Load model architecture with real checkpoint
        self.encoder = create_eegpt_model(str(self.config.model_path))

        if self.encoder is None:
            raise RuntimeError("Failed to load EEGPT encoder")

        self.encoder.to(self.device)
        self.encoder.eval()

        # Initialize abnormality detection head (trainable on top of frozen EEGPT)
        self.abnormality_head = nn.Sequential(
            nn.Linear(self.config.embed_dim * self.config.n_summary_tokens, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),  # Binary classification
        ).to(self.device)

        self.is_loaded = True
        self.logger.info(f"✅ Loaded real EEGPT model from {self.config.model_path}")

    def extract_features(
        self, data: npt.NDArray[np.float64], channel_names: list[str]
    ) -> npt.NDArray[np.float64]:
        """Extract features from EEG data using EEGPT encoder."""
        if not self.is_loaded or self.encoder is None:
            self.load_model()

        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")

        # Data should be shape (channels, time_samples)
        n_channels, n_samples = data.shape

        # Reshape data into patches
        # EEGPT expects patches of size 64 samples (250ms at 256Hz)
        patch_size = self.config.patch_size
        n_patches = n_samples // patch_size

        if n_samples % patch_size != 0:
            # Truncate to fit exact patches
            data = data[:, : n_patches * patch_size]

        # Reshape to (n_channels, n_patches, patch_size)
        data_patched = data.reshape(n_channels, n_patches, patch_size)

        # Rearrange to (n_patches, n_channels, patch_size) for the model
        data_rearranged = np.transpose(data_patched, (1, 0, 2))

        # Flatten patches: (n_patches * n_channels, patch_size)
        data_flattened = data_rearranged.reshape(-1, patch_size)

        # Convert to tensor and add batch dimension
        data_tensor = torch.FloatTensor(data_flattened).unsqueeze(0).to(self.device)

        # Prepare channel IDs with caching for performance
        chan_ids = self._get_cached_channel_ids(channel_names)

        with torch.no_grad():
            if self.encoder is not None:  # Additional null check for mypy
                features = self.encoder(data_tensor, chan_ids)
                # Average features across all patches to get summary
                features = features.squeeze(0)  # Remove batch dim
                # Reshape back to (n_patches, n_channels, embed_dim)
                features = features.reshape(n_patches, n_channels, -1)
                # Average across patches and channels to get summary tokens
                # For now, return mean features (simplified)
                summary_features = features.mean(dim=(0, 1))  # Shape: (embed_dim,)
                # Repeat to get n_summary_tokens
                summary_features = summary_features.unsqueeze(0).expand(
                    self.config.n_summary_tokens, -1
                )
                result: npt.NDArray[np.float64] = summary_features.cpu().numpy()
                return result

        # This should never be reached due to the encoder None check above
        return np.zeros((self.config.n_summary_tokens, self.config.embed_dim), dtype=np.float64)  # type: ignore[unreachable]

    def predict_abnormality(self, raw: "mne.io.Raw") -> dict[str, Any]:  # Use string annotation
        """Predict abnormality from raw EEG data with streaming support."""
        if not self.is_loaded:
            try:
                self.load_model()
            except Exception as e:
                # Return error result if model loading fails
                return {
                    "abnormal_probability": 0.5,  # Unknown
                    "confidence": 0.0,
                    "window_scores": [],
                    "n_windows": 0,
                    "used_streaming": False,
                    "error": str(e),
                }

        # Preprocess data
        processed = preprocess_for_eegpt(raw)

        # Get channel names
        channel_names = processed.ch_names

        # Check duration to decide on streaming (configurable threshold)
        duration = processed.times[-1]
        use_streaming = duration > self.config.streaming_threshold

        if use_streaming:
            # Use streaming for large files

            window_scores = []
            n_windows_processed = 0

            # Create a temporary streamer-compatible object
            # EDFStreamer expects file path, but we have raw data
            # So we'll process windows directly
            sfreq = int(processed.info["sfreq"])
            window_duration = 4.0  # 4 second windows
            step_duration = 2.0  # 2 second step (50% overlap)

            window_samples = int(window_duration * sfreq)
            step_samples = int(step_duration * sfreq)

            # Get data once for streaming
            data = processed.get_data()
            n_samples = data.shape[1]

            # Process windows with overlap
            for start_idx in range(0, n_samples - window_samples + 1, step_samples):
                end_idx = start_idx + window_samples
                window_data = data[:, start_idx:end_idx]

                # Extract features
                features = self.extract_features(window_data, channel_names)

                # Apply abnormality detection
                if self.abnormality_head is not None:
                    with torch.no_grad():
                        features_flat = (
                            torch.FloatTensor(features.flatten()).unsqueeze(0).to(self.device)
                        )
                        logits = self.abnormality_head(features_flat)
                        probs = torch.softmax(logits, dim=-1)
                        abnormal_prob = probs[0, 1].item()
                        window_scores.append(abnormal_prob)

                n_windows_processed += 1

            # Aggregate scores
            abnormality_score = np.mean(window_scores) if window_scores else 0.0
            confidence = 1.0 - np.std(window_scores) if len(window_scores) > 1 else 0.8

            return {
                "abnormal_probability": float(abnormality_score),
                "confidence": float(confidence),
                "window_scores": window_scores,
                "n_windows": len(window_scores),
                "mean_score": float(abnormality_score),
                "std_score": float(np.std(window_scores)) if window_scores else 0.0,
                "used_streaming": True,
                "n_windows_processed": n_windows_processed,
                "metadata": {
                    "duration": duration,
                    "n_channels": len(channel_names),
                    "sampling_rate": sfreq,
                },
            }
        else:
            # Original non-streaming logic for small files
            data = processed.get_data()
            windows = self.extract_windows(data, int(processed.info["sfreq"]))

            if len(windows) == 0:
                return {
                    "abnormal_probability": 0.0,
                    "confidence": 0.0,
                    "window_scores": [],
                    "n_windows": 0,
                    "used_streaming": False,
                    "error": "No valid windows extracted",
                }

            # Extract features for all windows
            window_features = []
            for window in windows:
                features = self.extract_features(window, channel_names)
                window_features.append(features)

            # Features extracted for all windows

            # Apply abnormality detection head
            window_scores = []
            if self.abnormality_head is not None:
                with torch.no_grad():
                    for features in window_features:
                        features_flat = (
                            torch.FloatTensor(features.flatten()).unsqueeze(0).to(self.device)
                        )
                        logits = self.abnormality_head(features_flat)
                        probs = torch.softmax(logits, dim=-1)
                        abnormal_prob = probs[0, 1].item()
                        window_scores.append(abnormal_prob)

            # Aggregate scores
            abnormality_score = np.mean(window_scores)
            confidence = 1.0 - np.std(window_scores) if len(window_scores) > 1 else 0.8

            return {
                "abnormal_probability": float(abnormality_score),
                "confidence": float(confidence),
                "window_scores": window_scores,
                "n_windows": len(windows),
                "used_streaming": False,
                "channels_used": channel_names,
                "metadata": {
                    "duration": processed.times[-1],
                    "n_channels": len(channel_names),
                    "sampling_rate": processed.info["sfreq"],
                },
            }

    def process_recording(
        self, file_path: str | Path, analysis_type: str = "abnormality"
    ) -> dict[str, Any]:
        """Process a complete EEG recording."""
        try:
            import mne
        except ImportError as e:
            raise ImportError("MNE-Python is required for EEG processing") from e

        file_path = Path(file_path)

        # Load raw data
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

        if analysis_type == "abnormality":
            return self.predict_abnormality(raw)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    def extract_windows(self, data: np.ndarray, sampling_rate: int) -> list[np.ndarray]:
        """Extract non-overlapping windows from continuous data.

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

    def extract_features_batch(
        self, windows: np.ndarray, channel_names: list[str] | None = None
    ) -> np.ndarray:
        """Extract features from batch of windows.

        Args:
            windows: Batch of windows (batch, channels, samples)
            channel_names: List of channel names

        Returns:
            Features (batch, n_summary_tokens, feature_dim)
        """
        batch_size, n_channels, n_samples = windows.shape

        # Process each window individually for now
        # TODO: Optimize for true batch processing
        batch_features = []

        for i in range(batch_size):
            window = windows[i]
            if channel_names is None:
                ch_names = [f"EEG{j:03d}" for j in range(n_channels)]
            else:
                ch_names = channel_names

            features = self.extract_features(window, ch_names)
            batch_features.append(features)

        return np.stack(batch_features, axis=0)

    def cleanup(self) -> None:
        """Clean up GPU memory if using CUDA."""
        if self.device.type == "cuda":
            # Clear any cached allocations
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory cache")


def preprocess_for_eegpt(
    raw: "mne.io.Raw",  # Use string annotation for untyped module
    target_sfreq: int = 256,
    l_freq: float = 0.5,
    h_freq: float = 50.0,
    notch_freq: float | list | None = None,
) -> "mne.io.Raw":  # Use string annotation
    """Preprocess EEG data for EEGPT model."""
    # Mark unused parameters as intentionally unused for now
    _ = l_freq
    _ = h_freq
    _ = notch_freq

    raw = raw.copy()

    # Resample to 256 Hz if needed
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)

    # Apply average reference
    raw.set_eeg_reference("average", projection=False)

    # Ensure proper channel types
    raw.pick_types(meg=False, eeg=True, eog=False, exclude="bads")

    # Limit to max channels if needed
    if len(raw.ch_names) > 58:
        raw.pick_channels(raw.ch_names[:58])

    return raw


def extract_features_from_raw(raw: mne.io.Raw, model_path: str | Path) -> dict[str, Any]:
    """High-level function to extract features from raw EEG.

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
    result["processing_time"] = time.time() - start_time
    result["features"] = True  # Placeholder

    return result
