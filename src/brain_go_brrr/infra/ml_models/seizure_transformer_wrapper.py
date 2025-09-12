"""SeizureTransformer wrapper with safe imports and overlap aggregation.

This wrapper expects either:
- An externally provided model instance (dependency injection), or
- A pip-installable package exposing `build_seizure_transformer(n_channels: int)`.

Weights loading is optional. Inputs are Volts in `(C, T)` at 256 Hz by default.
IMPORTANT: The reference implementation requires UNIPOLAR montage. Ensure your inputs
are referential/unipolar (not bipolar channel pairs) before calling `predict`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

# SciPy dependencies are imported lazily inside methods to avoid hard dependency


class SeizureTransformerWrapper:
    def __init__(
        self,
        model: nn.Module | None = None,
        build_fn: Callable[[int], nn.Module] | None = None,
        weights_path: Path | str | None = None,
        n_channels: int = 19,
        fs: int = 256,
        window_samples: int = 15360,  # 60s @ 256Hz
        overlap_ratio: float = 0.0,  # No overlap by default (matches reference)
        device: torch.device | None = None,
    ) -> None:
        """Initialize the SeizureTransformer wrapper.

        Args:
            model: Pre-instantiated SeizureTransformer model (optional).
            build_fn: Function to build the model given n_channels (optional).
            weights_path: Path to model weights checkpoint (optional).
            n_channels: Number of input channels (default: 19).
            fs: Sampling frequency in Hz (default: 256).
            window_samples: Window size in samples (default: 15360 for 60s @ 256Hz).
            overlap_ratio: Overlap ratio for sliding windows (default: 0.0).
            device: Torch device for computation (default: auto-detect).
        """
        self.fs = fs
        self.window_samples = window_samples
        self.overlap_ratio = overlap_ratio
        self.n_channels = n_channels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Preprocessing params from reference implementation
        self.lowcut = 0.5
        self.highcut = 120
        self._notch_coeffs: (
            tuple[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None
        ) = None  # lazy-init (b,a) tuples for 1 Hz and 60 Hz

        if model is not None:
            self.model = model
        else:
            if build_fn is None:
                # Try to import the reference model without mutating sys.path
                try:  # pragma: no cover - optional integration
                    from wu_2025.architecture import SeizureTransformer  # type: ignore

                    self.model = SeizureTransformer(
                        in_channels=self.n_channels,
                        in_samples=self.window_samples,
                        drop_rate=0.1,
                    )
                except Exception as e:  # pragma: no cover
                    raise ImportError(
                        "Could not import `wu_2025.architecture.SeizureTransformer`. "
                        "Pass a `model` or a `build_fn`, or install the reference package on PYTHONPATH."
                    ) from e
            else:
                self.model = build_fn(self.n_channels)

        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                # Note: Reference implementation doesn't use weights_only
                # since model weights contain custom objects
                ckpt = torch.load(p, map_location="cpu", weights_only=False)  # nosec:weights_only - model contains architecture
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                else:
                    state_dict = ckpt
                self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_clip(self, eeg_clip: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply preprocessing from reference implementation."""
        from scipy.signal import butter, iirnotch, lfilter  # lazy import

        # Bandpass filter
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(3, [low, high], btype='band')
        filtered = lfilter(b, a, eeg_clip, axis=1)

        # Notch filters (lazy init coefficients)
        if self._notch_coeffs is None:
            notch_1 = iirnotch(1, Q=30, fs=self.fs)
            notch_60 = iirnotch(60, Q=30, fs=self.fs)
            self._notch_coeffs = (notch_1, notch_60)
        (n1_b, n1_a), (n60_b, n60_a) = self._notch_coeffs
        filtered = lfilter(n1_b, n1_a, filtered, axis=1)
        filtered = lfilter(n60_b, n60_a, filtered, axis=1)

        return filtered.astype(np.float32)  # type: ignore[no-any-return]

    def _postprocess(self, predictions: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply post-processing from reference implementation."""
        from scipy.ndimage import binary_closing, binary_opening  # lazy import

        # Threshold at 0.8 (from reference)
        binary = (predictions > 0.8).astype(int)

        # Morphological opening to remove short bursts
        structure = np.ones(5, dtype=bool)
        binary = binary_opening(binary, structure=structure).astype(int)

        # Morphological closing to fill gaps
        binary = binary_closing(binary, structure=structure).astype(int)

        # Remove events shorter than 2 seconds
        min_samples = int(2.0 * self.fs)
        is_seizure = False
        start_idx = 0

        for i in range(len(binary)):
            if not is_seizure and binary[i] == 1:
                is_seizure = True
                start_idx = i
            elif is_seizure and (binary[i] == 0 or i == len(binary) - 1):
                end_idx = i if binary[i] == 0 else i + 1
                length = end_idx - start_idx
                if length < min_samples:
                    binary[start_idx:end_idx] = 0
                is_seizure = False

        return binary.astype(np.float32)  # type: ignore[no-any-return]

    @torch.no_grad()
    def predict(
        self, eeg: npt.NDArray[np.float32], apply_postprocessing: bool = True
    ) -> npt.NDArray[np.float32]:
        """Return per-sample seizure predictions for a single recording.

        Args:
            eeg: Array of shape (C, T) in Volts at `fs` Hz.
            apply_postprocessing: Whether to apply morphological filtering and thresholding.

        Returns:
            predictions: Array of shape (T,) with values in [0, 1] (or binary if postprocessed).
        """
        assert eeg.ndim == 2, "expected (C, T)"
        c, t = int(eeg.shape[0]), int(eeg.shape[1])
        if c != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {c}")

        # Z-score normalization per channel (from reference)
        eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (
            np.std(eeg, axis=1, keepdims=True) + 1e-8
        )

        # Resample if needed
        if self.fs != 256:
            from scipy.signal import resample  # lazy import

            new_n_samples = int(t * 256.0 / self.fs)
            eeg = resample(eeg, new_n_samples, axis=1)
            t = new_n_samples

        # Apply preprocessing
        eeg = self._preprocess_clip(eeg)

        # Calculate stride
        stride = int(self.window_samples * (1 - self.overlap_ratio))

        # Collect all predictions
        all_outputs = []

        # Process windows
        for start_idx in range(0, t, stride):
            if start_idx + self.window_samples > t:
                # Last window: pad if needed
                clip = eeg[:, start_idx:]
                pad_length = self.window_samples - clip.shape[1]
                if pad_length > 0:
                    clip = np.pad(clip, ((0, 0), (0, pad_length)), mode='constant')
            else:
                clip = eeg[:, start_idx : start_idx + self.window_samples]

            # Convert to tensor and predict
            x = torch.from_numpy(clip).unsqueeze(0).float().to(self.device)

            # Model returns per-timestep predictions for the window
            output = self.model(x)  # Shape: [1, window_samples]
            all_outputs.append(output.cpu().numpy())

            # Break if we've covered the whole recording
            if start_idx + self.window_samples >= t:
                break

        # Concatenate and flatten predictions
        if all_outputs:
            predictions = np.concatenate([o.flatten() for o in all_outputs])[:t]
        else:
            predictions = np.zeros(t, dtype=np.float32)

        # Apply post-processing if requested
        if apply_postprocessing:
            predictions = self._postprocess(predictions)

        return predictions  # type: ignore[no-any-return]


__all__ = ["SeizureTransformerWrapper"]
