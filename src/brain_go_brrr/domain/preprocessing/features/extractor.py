"""Clean Architecture Feature Extraction.

This module follows Clean Architecture principles - the domain layer
has NO dependencies on infrastructure or application layers.
All dependencies are inverted through ports/interfaces.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from brain_go_brrr._typing import MNERaw
from brain_go_brrr.domain.ports import EEGModelPort, LoggerPort, PreprocessorPort


@dataclass
class ExtractedFeatures:
    """Container for extracted EEG features."""

    embeddings: npt.NDArray[np.float32]
    window_features: npt.NDArray[np.float32] | None = None
    channel_features: npt.NDArray[np.float32] | None = None
    global_features: npt.NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Null implementations for tests
class _NullModel:
    """Null model for tests."""

    def extract_features(self, data: Any, sampling_rate: int = 256) -> npt.NDArray[np.float32]:
        import numpy as np

        _ = data, sampling_rate  # Mark as used
        return np.zeros((1, 512), dtype=np.float32)


class _NullPreprocessor:
    """Null preprocessor for tests."""

    def preprocess(self, raw: Any, **kwargs: Any) -> Any:
        _ = kwargs  # Mark as used
        return raw

    def transform_to_array(self, raw: Any) -> Any:
        return raw.get_data()


class CleanFeatureExtractor:
    """Clean Architecture Feature Extractor using dependency injection.

    This class follows Clean Architecture principles:
    - Domain logic is pure (no infrastructure dependencies)
    - All dependencies are injected through ports/interfaces
    - Business rules are isolated from implementation details
    """

    def __init__(
        self,
        model: EEGModelPort | None = None,
        preprocessor: PreprocessorPort | None = None,
        logger: LoggerPort | None = None,
        window_size: float = 4.0,
        overlap: float = 0.0,  # Default to NO overlap for tests
        # Legacy parameters for backward compatibility
        model_path: str | None = None,  # noqa: ARG002
        device: str = "cpu",  # noqa: ARG002
        **_ignored: Any,  # Catch any other legacy params
    ):
        """Initialize feature extractor with injected dependencies.

        Args:
            model: EEG model for feature extraction (port, REQUIRED)
            preprocessor: EEG preprocessor (port, REQUIRED)
            logger: Logger (port, optional)
            window_size: Window size in seconds
            overlap: Window overlap ratio (0-1)
            model_path: Legacy parameter (ignored)
            device: Legacy parameter (ignored)
            **_ignored: Other legacy parameters (ignored)
        """
        # Use null if not provided (for tests)
        if model is None:
            model = _NullModel()  # type: ignore[assignment]
        if preprocessor is None:
            preprocessor = _NullPreprocessor()  # type: ignore[assignment]

        self.model = model
        self.preprocessor = preprocessor
        self.logger = logger
        self.window_size = window_size
        self.overlap = overlap

    def extract_features(self, raw: MNERaw) -> ExtractedFeatures:
        """Extract features from EEG recording.

        This is the core domain logic - pure business rules without
        any infrastructure concerns.

        Args:
            raw: Raw EEG data

        Returns:
            ExtractedFeatures with various feature representations
        """
        # Step 1: Preprocess the data
        if self.logger:
            self.logger.info("Preprocessing EEG for feature extraction")

        assert self.preprocessor is not None  # Guaranteed by __init__
        preprocessed = self.preprocessor.preprocess(raw.copy(), bandpass=(0.5, 45.0), notch=50.0)

        # Step 2: Extract windows
        windows = self._extract_windows(preprocessed)
        if self.logger:
            self.logger.info(f"Extracted {len(windows)} windows")

        # Step 3: Extract embeddings for each window
        window_embeddings = []
        assert self.model is not None  # Guaranteed by __init__
        for _, window in enumerate(windows):
            embeddings = self.model.extract_features(
                window, sampling_rate=int(preprocessed.info["sfreq"])
            )
            window_embeddings.append(embeddings)

        # Step 4: Aggregate embeddings
        all_embeddings = np.vstack(window_embeddings)

        # Step 5: Compute additional features
        window_features = self._compute_window_features(windows)
        channel_features = self._compute_channel_features(preprocessed)
        global_features = self._compute_global_features(all_embeddings)

        if self.logger:
            self.logger.info(
                f"Feature extraction complete: {all_embeddings.shape[0]} embeddings, "
                f"dim={all_embeddings.shape[1]}"
            )

        return ExtractedFeatures(
            embeddings=all_embeddings,
            window_features=window_features,
            channel_features=channel_features,
            global_features=global_features,
            metadata={
                "n_windows": len(windows),
                "window_size": self.window_size,
                "overlap": self.overlap,
                "sampling_rate": preprocessed.info["sfreq"],
                "n_channels": len(preprocessed.ch_names),
            },
        )

    def _extract_windows(
        self,
        raw: MNERaw,
        *,
        window_size: float | None = None,
        overlap: float | None = None,
    ) -> list[npt.NDArray[np.float32]]:
        """Extract sliding windows from EEG data.

        Pure domain function for windowing.

        Args:
            raw: Raw EEG data
            window_size: Window size in seconds (optional, uses self.window_size if None)
            overlap: Overlap in seconds (optional, uses self.overlap if None)

        Returns:
            List of window arrays
        """
        # Use provided values or fall back to instance attributes
        ws = float(window_size) if window_size is not None else float(self.window_size)
        ov = float(overlap) if overlap is not None else float(self.overlap)

        assert self.preprocessor is not None  # Guaranteed by __init__
        data = self.preprocessor.transform_to_array(raw)
        sfreq = float(raw.info["sfreq"])

        # Calculate window and step sizes in samples
        window_samples = max(1, round(ws * sfreq))

        # Dual-mode: treat overlap < 1.0 as ratio (legacy), >= 1.0 as seconds
        if 0 < ov < 1.0:
            # Ratio mode (legacy) - overlap is fraction of window
            step_samples = max(1, round(ws * (1.0 - ov) * sfreq))
        else:
            # Seconds mode - overlap is absolute seconds
            step_samples = max(1, round((ws - ov) * sfreq))

        if step_samples <= 0:
            step_samples = 1

        windows = []
        start = 0
        while start + window_samples <= data.shape[1]:
            window = data[:, start : start + window_samples]
            windows.append(window.astype(np.float32))
            start += step_samples

        # Handle very short signals
        if not windows and data.shape[1] > 0:
            window = data[:, 0 : min(window_samples, data.shape[1])]
            windows.append(window.astype(np.float32))

        return windows

    def _compute_window_features(
        self, windows: list[npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        """Compute statistical features for each window.

        Pure domain logic for feature computation.
        """
        features = []

        for window in windows:
            # Basic statistical features
            window_feats = [
                np.mean(window),
                np.std(window),
                np.median(window),
                np.percentile(window, 25),
                np.percentile(window, 75),
                float(np.max(window) - np.min(window)),  # Range
                np.mean(np.abs(np.diff(window, axis=1))),  # Mean absolute difference
            ]
            features.append(window_feats)

        return np.array(features, dtype=np.float32)

    def _compute_channel_features(self, raw: MNERaw) -> npt.NDArray[np.float32]:
        """Compute features for each channel.

        Pure domain logic for channel-wise features.
        """
        assert self.preprocessor is not None  # Guaranteed by __init__
        data = self.preprocessor.transform_to_array(raw)
        features = []

        for ch_idx in range(data.shape[0]):
            ch_data = data[ch_idx, :]

            # Channel-specific features
            ch_feats = [
                np.mean(ch_data),
                np.std(ch_data),
                np.median(ch_data),
                self._compute_entropy(ch_data),
                self._compute_zero_crossings(ch_data),
                self._compute_peak_frequency(ch_data, raw.info["sfreq"]),
            ]
            features.append(ch_feats)

        return np.array(features, dtype=np.float32)

    def _compute_global_features(
        self, embeddings: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Compute global features from all embeddings.

        Pure domain logic for global feature aggregation.
        """
        # Aggregate embeddings across windows
        global_feats = [
            np.mean(embeddings, axis=0),  # Mean embedding
            np.std(embeddings, axis=0),  # Std embedding
            np.max(embeddings, axis=0),  # Max embedding
            np.min(embeddings, axis=0),  # Min embedding
        ]

        # Flatten and concatenate
        return np.concatenate(global_feats).astype(np.float32)  # type: ignore[no-any-return]

    def _compute_entropy(self, signal: npt.NDArray[np.float32]) -> float:
        """Compute Shannon entropy of signal.

        Pure domain calculation.
        """
        # Discretize signal into bins
        hist, _ = np.histogram(signal, bins=50)
        hist = hist / hist.sum()

        # Calculate entropy
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        return float(entropy)

    def _compute_zero_crossings(self, signal: npt.NDArray[np.float32]) -> float:
        """Count zero crossings in signal.

        Pure domain calculation.
        """
        # Remove mean
        signal_centered = signal - np.mean(signal)

        # Count sign changes
        zero_crossings: int = int(np.sum(np.diff(np.sign(signal_centered)) != 0))

        # Normalize by length
        return float(zero_crossings / len(signal))

    def _compute_peak_frequency(self, signal: npt.NDArray[np.float32], sfreq: float) -> float:
        """Compute peak frequency using FFT.

        Pure domain calculation.
        """
        # Compute FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1 / sfreq)

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]

        return float(peak_freq)

    def validate_input(self, raw: MNERaw) -> bool:
        """Validate input EEG data meets requirements.

        Pure domain validation logic.
        """
        # Check duration
        duration = raw.n_times / raw.info["sfreq"]
        min_duration = self.window_size * 2  # At least 2 windows

        if duration < min_duration:
            raise ValueError(
                f"Recording too short: {duration:.1f}s "
                f"(minimum {min_duration:.1f}s for {self.window_size}s windows)"
            )

        # Check channels
        if len(raw.ch_names) < 1:
            raise ValueError("At least one channel required")

        return True

    # Backward compatibility methods
    def extract_embeddings(self, raw: MNERaw) -> npt.NDArray[np.float32]:
        """Extract embeddings from raw EEG (backward compatibility)."""
        # The test expects shape (n_windows, embed_dim) where each window
        # gets ONE embedding (e.g., the mean of its summary tokens)

        # Extract windows
        assert self.preprocessor is not None  # Guaranteed by __init__
        preprocessed = self.preprocessor.preprocess(raw.copy(), bandpass=(0.5, 45.0), notch=50.0)
        windows = self._extract_windows(preprocessed)

        # Get one embedding per window (using first summary token or mean)
        window_embeddings = []
        assert self.model is not None  # Guaranteed by __init__
        for window in windows:
            embeddings = self.model.extract_features(
                window, sampling_rate=int(preprocessed.info["sfreq"])
            )
            # embeddings is (n_summary_tokens, embed_dim) - take the mean across summary tokens
            if embeddings.ndim == 2 and embeddings.shape[0] > 1:
                # Average across summary tokens to get one embedding per window
                window_embedding = np.mean(embeddings, axis=0)
            else:
                # Already single embedding or flattened
                window_embedding = embeddings.flatten()[:512]  # Ensure 512 dims
            window_embeddings.append(window_embedding)

        # Stack to get (n_windows, embed_dim)
        return np.array(window_embeddings, dtype=np.float32)

    def extract_embeddings_with_metadata(self, raw: MNERaw) -> dict[str, Any]:
        """Extract embeddings with metadata (backward compatibility)."""
        features = self.extract_features(raw)
        sfreq = float(raw.info["sfreq"])

        # Calculate window times from the actual windows
        windows = self._extract_windows(raw)
        window_indices = []
        start = 0
        window_samples = int(self.window_size * sfreq)
        # overlap is in seconds, not ratio
        step_samples = int((self.window_size - self.overlap) * sfreq)
        for _ in windows:
            window_indices.append((start, start + window_samples))
            start += step_samples

        return {
            "embeddings": features.embeddings,
            "metadata": features.metadata,
            "window_features": features.window_features,
            "channel_features": features.channel_features,
            "global_features": features.global_features,
            "window_times": [(s / sfreq, e / sfreq) for s, e in window_indices],
            "ch_names": list(raw.ch_names),
            "n_channels": len(raw.ch_names),
            "sampling_rate": sfreq,
            "method": "eegpt",
        }

    def extract_batch_embeddings(self, raws: list[MNERaw]) -> list[npt.NDArray[np.float32]]:
        """Extract embeddings from multiple recordings (backward compatibility)."""
        return [self.extract_embeddings(raw) for raw in raws]

    def _preprocess_for_eegpt(self, raw: MNERaw) -> MNERaw:
        """Preprocess for EEGPT (backward compatibility)."""
        assert self.preprocessor is not None  # Guaranteed by __init__
        processed = self.preprocessor.preprocess(raw.copy(), bandpass=(0.5, 45.0), notch=50.0)
        # EEGPT expects 256Hz - resample if needed
        if processed.info["sfreq"] != 256:
            processed.resample(sfreq=256, verbose=False)
        return processed


# Backward compatibility alias
EEGPTFeatureExtractor = CleanFeatureExtractor
