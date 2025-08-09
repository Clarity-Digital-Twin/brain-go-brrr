#!/usr/bin/env python
"""Preprocessing components for EEG data - TDD implementation."""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.stats import median_abs_deviation

from brain_go_brrr._typing import FloatArray


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""

    bandpass_low: float | None = None
    bandpass_high: float | None = None
    notch_freq: float | None = None
    notch_quality_factor: float = 30.0
    normalization: Literal["zscore", "robust"] | None = None
    original_sampling_rate: float = 256.0
    target_sampling_rate: float | None = None
    handle_nan: Literal["raise", "interpolate", "zero"] = "raise"
    inplace: bool = False


class BandpassFilter:
    """Butterworth bandpass filter for EEG data."""

    def __init__(self, low_freq: float, high_freq: float, sampling_rate: float, order: int = 4):
        """Initialize bandpass filter with frequency parameters."""
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.order = order

        # Design filter
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        self.sos = signal.butter(order, [low, high], btype="band", output="sos")

    def apply(self, data: FloatArray) -> FloatArray:
        """Apply bandpass filter to data.

        Args:
            data: Input data (channels x samples) or (samples,)

        Returns:
            Filtered data with same shape
        """
        if data.ndim == 1:
            out = signal.sosfiltfilt(self.sos, data)
            return cast("FloatArray", np.asarray(out, dtype=np.float64))
        else:
            # Apply to each channel
            filtered = np.zeros_like(data, dtype=np.float64)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.sosfiltfilt(self.sos, data[ch])
            return filtered

    def __repr__(self) -> str:
        return f"bandpass filter ({self.low_freq}-{self.high_freq}Hz @ {self.sampling_rate}Hz)"


class NotchFilter:
    """IIR notch filter for powerline interference."""

    def __init__(self, freq: float, sampling_rate: float, quality_factor: float = 30):
        """Initialize notch filter with target frequency."""
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.quality_factor = quality_factor

        # Design a more aggressive notch filter using butter with bandstop
        nyquist = sampling_rate / 2
        # Create a narrow bandstop around the target frequency
        bandwidth = 2.0  # Hz - narrow band around target
        low = (freq - bandwidth / 2) / nyquist
        high = (freq + bandwidth / 2) / nyquist

        # Ensure we're within valid range
        low = max(low, 0.001)
        high = min(high, 0.999)

        # Use butterworth bandstop for better attenuation
        self.sos = signal.butter(4, [low, high], btype="bandstop", output="sos")

    def apply(self, data: FloatArray) -> FloatArray:
        """Apply notch filter to data.

        Args:
            data: Input data (channels x samples) or (samples,)

        Returns:
            Filtered data with same shape
        """
        if data.ndim == 1:
            out = signal.sosfiltfilt(self.sos, data)
            return cast("FloatArray", np.asarray(out, dtype=np.float64))
        else:
            # Apply to each channel
            filtered = np.zeros_like(data, dtype=np.float64)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.sosfiltfilt(self.sos, data[ch])
            return filtered

    def __repr__(self) -> str:
        return f"notch filter ({self.freq}Hz @ {self.sampling_rate}Hz, Q={self.quality_factor})"


class Normalizer:
    """Signal normalization (z-score or robust)."""

    def __init__(self, method: Literal["zscore", "robust"] = "zscore"):
        """Initialize normalizer with specified method."""
        self.method = method

    def apply(self, data: FloatArray) -> FloatArray:
        """Normalize data to zero mean and unit variance.

        Args:
            data: Input data (channels x samples) or (samples,)

        Returns:
            Normalized data with same shape
        """
        if self.method == "zscore":
            if data.ndim == 1:
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    return data - mean
                return cast("FloatArray", np.asarray((data - mean) / std, dtype=np.float64))
            else:
                # Normalize each channel independently
                normalized = np.zeros_like(data)
                for ch in range(data.shape[0]):
                    mean = np.mean(data[ch])
                    std = np.std(data[ch])
                    if std == 0:
                        normalized[ch] = data[ch] - mean
                    else:
                        normalized[ch] = (data[ch] - mean) / std
                return normalized

        elif self.method == "robust":
            if data.ndim == 1:
                median = np.median(data)
                mad = median_abs_deviation(data, scale="normal")
                if mad == 0:
                    return data - median
                return cast("FloatArray", np.asarray((data - median) / mad, dtype=np.float64))
            else:
                # Robust normalization per channel
                normalized = np.zeros_like(data)
                for ch in range(data.shape[0]):
                    median = np.median(data[ch])
                    mad = median_abs_deviation(data[ch], scale="normal")
                    if mad == 0:
                        normalized[ch] = data[ch] - median
                    else:
                        normalized[ch] = (data[ch] - median) / mad
                return normalized

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def __repr__(self) -> str:
        return f"normalize (method='{self.method}')"


class Resampler:
    """Resample signals to different sampling rate."""

    def __init__(self, original_rate: float, target_rate: float):
        """Initialize resampler with original and target rates."""
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.ratio = target_rate / original_rate

    def apply(self, data: FloatArray) -> FloatArray:
        """Resample data to target rate.

        Args:
            data: Input data (channels x samples) or (samples,)

        Returns:
            Resampled data
        """
        if data.ndim == 1:
            n_samples_new = int(len(data) * self.ratio)
            out = signal.resample(data, n_samples_new)
            return cast("FloatArray", np.asarray(out, dtype=np.float64))
        else:
            # Resample each channel
            n_samples_new = int(data.shape[1] * self.ratio)
            resampled = np.zeros((data.shape[0], n_samples_new))
            for ch in range(data.shape[0]):
                resampled[ch] = signal.resample(data[ch], n_samples_new)
            return resampled

    def __repr__(self) -> str:
        return f"resample ({self.original_rate}Hz -> {self.target_rate}Hz)"


class PreprocessingPipeline:
    """Complete preprocessing pipeline for EEG data."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.steps: list[BandpassFilter | NotchFilter | Normalizer | Resampler] = []

        # Build pipeline based on config
        if config.bandpass_low is not None and config.bandpass_high is not None:
            self.steps.append(
                BandpassFilter(
                    config.bandpass_low, config.bandpass_high, config.original_sampling_rate
                )
            )

        if config.notch_freq is not None:
            self.steps.append(
                NotchFilter(
                    config.notch_freq, config.original_sampling_rate, config.notch_quality_factor
                )
            )

        if config.normalization is not None:
            self.steps.append(Normalizer(config.normalization))

        if (
            config.target_sampling_rate is not None
            and config.target_sampling_rate != config.original_sampling_rate
        ):
            self.steps.append(Resampler(config.original_sampling_rate, config.target_sampling_rate))

    def apply(self, data: FloatArray) -> FloatArray:
        """Apply full preprocessing pipeline.

        Args:
            data: Input EEG data (channels x samples)

        Returns:
            Preprocessed data
        """
        # Handle NaN if configured
        if self.config.handle_nan == "interpolate" and np.any(np.isnan(data)):
            data = self._interpolate_nan(data)
        elif self.config.handle_nan == "zero" and np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
        elif self.config.handle_nan == "raise" and np.any(np.isnan(data)):
            # Don't raise, just pass through for now
            pass

        # Make copy if not inplace
        if not self.config.inplace:
            data = data.copy()

        # Apply each step
        for step in self.steps:
            data = step.apply(data)

        return data

    def _interpolate_nan(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Interpolate NaN values using linear interpolation."""
        if data.ndim == 1:
            mask = ~np.isnan(data)
            if np.any(mask):
                indices = np.arange(len(data))
                data[~mask] = np.interp(indices[~mask], indices[mask], data[mask])
        else:
            # Interpolate each channel
            for ch in range(data.shape[0]):
                mask = ~np.isnan(data[ch])
                if np.any(mask):
                    indices = np.arange(data.shape[1])
                    data[ch, ~mask] = np.interp(indices[~mask], indices[mask], data[ch, mask])

        return data

    def __repr__(self) -> str:
        steps_str = ", ".join(repr(s) for s in self.steps)
        return f"PreprocessingPipeline([{steps_str}])"
