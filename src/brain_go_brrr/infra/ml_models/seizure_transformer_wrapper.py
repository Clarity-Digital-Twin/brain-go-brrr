"""SeizureTransformer wrapper with safe imports and overlap aggregation.

This wrapper expects either:
- An externally provided model instance (dependency injection), or
- A pip-installable package exposing `build_seizure_transformer(n_channels: int)`.

Weights loading is optional. Inputs are Volts in `(C, T)` at 256 Hz by default.
IMPORTANT: The reference implementation requires UNIPOLAR montage. Ensure your inputs
are referential/unipolar (not bipolar channel pairs) before calling `predict`.

This wrapper uses the SSOT preprocessing and post-processing utilities from
seizure_transformer_utils to ensure consistency across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from brain_go_brrr.infra.ml_models.seizure_transformer_utils import (
    SeizurePreprocessor,
    SeizurePostProcessor,
    prepare_channels,
    CANONICAL_CHANNELS,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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
        
        # Use SSOT preprocessing and post-processing utilities
        self.preprocessor = SeizurePreprocessor(target_fs=self.fs)
        self.postprocessor = SeizurePostProcessor(fs=self.fs)

        if model is not None:
            self.model = model
        else:
            if build_fn is None:
                # Use the local implementation by default
                from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer

                self.model = SeizureTransformer(
                    in_channels=self.n_channels,
                    in_samples=self.window_samples,
                    drop_rate=0.1,
                )
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

    def _ensure_canonical_channels(
        self, 
        eeg: npt.NDArray[np.float32],
        channel_names: list[str] | None = None
    ) -> npt.NDArray[np.float32]:
        """Ensure EEG data is in canonical channel order.
        
        If channel names are provided, reorder/pad to canonical order.
        Otherwise, assume data is already in correct order and just validate shape.
        """
        if channel_names is not None:
            # Use the SSOT utility to prepare channels
            prepared, _ = prepare_channels(eeg, channel_names, CANONICAL_CHANNELS[:self.n_channels])
            return prepared
        else:
            # Just ensure correct number of channels
            c = eeg.shape[0]
            if c == self.n_channels:
                return eeg
            elif c > self.n_channels:
                return eeg[:self.n_channels, :]
            else:
                # Pad with zeros
                padded = np.zeros((self.n_channels, eeg.shape[1]), dtype=np.float32)
                padded[:c, :] = eeg
                return padded

    @torch.no_grad()
    def predict(
        self, 
        eeg: npt.NDArray[np.float32], 
        fs_original: int | None = None,
        channel_names: list[str] | None = None,
        apply_postprocessing: bool = True
    ) -> npt.NDArray[np.float32]:
        """Return per-sample seizure predictions for a single recording.

        Args:
            eeg: Array of shape (C, T) in Volts.
            fs_original: Original sampling rate (default: self.fs).
            channel_names: Channel names for reordering to canonical (optional).
            apply_postprocessing: Whether to apply morphological filtering and thresholding.

        Returns:
            predictions: Array of shape (T,) with values in [0, 1] (or binary if postprocessed).
        """
        assert eeg.ndim == 2, "expected (C, T)"
        
        # Use provided fs or default
        if fs_original is None:
            fs_original = self.fs
        
        # Ensure canonical channel order if names provided
        eeg = self._ensure_canonical_channels(eeg, channel_names)
        
        # Apply SSOT preprocessing (z-score → resample → bandpass → notch)
        eeg = self.preprocessor.preprocess(eeg, fs_original)
        t = eeg.shape[1]

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

        # Apply SSOT post-processing if requested
        if apply_postprocessing:
            predictions = self.postprocessor.postprocess(predictions)

        return predictions  # type: ignore[no-any-return]


__all__ = ["SeizureTransformerWrapper"]
