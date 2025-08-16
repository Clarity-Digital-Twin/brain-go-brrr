"""EDF file validation for quality and compatibility checks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ValidationResult:
    """Result of EDF validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]


class EDFValidator:
    """Validates EDF files for processing requirements."""

    # Supported sampling rates
    VALID_SAMPLING_RATES = [100, 128, 200, 250, 256, 500, 512, 1000]

    def __init__(
        self,
        min_duration_seconds: float = 60.0,
        min_channels: int = 19,
        max_amplitude_v: float = 1e-3,  # 1mV
    ):
        """Initialize EDF validator with parameters."""
        self.min_duration_seconds = min_duration_seconds
        self.min_channels = min_channels
        self.max_amplitude_v = max_amplitude_v

    def validate(self, file_path: Path) -> ValidationResult:
        """Validate EDF file from path.

        Args:
            file_path: Path to EDF file

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # Check file exists
        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Check file extension
        if file_path.suffix.lower() != ".edf":
            errors.append(f"Invalid file extension: {file_path.suffix}, expected .edf")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Would load and validate actual EDF data here
        # For now, return validation result
        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata=metadata
        )

    def validate_data(self, edf_data: Any) -> ValidationResult:
        """Validate loaded EDF data.

        Args:
            edf_data: Loaded EDF data object

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # Extract metadata
        duration_seconds = edf_data.duration
        sfreq = edf_data.sfreq
        n_channels = edf_data.n_channels

        metadata["duration_seconds"] = duration_seconds
        metadata["sampling_rate"] = sfreq
        metadata["n_channels"] = n_channels

        # Check duration
        if duration_seconds < self.min_duration_seconds:
            errors.append(
                f"Recording too short: {duration_seconds}s, "
                f"minimum required: {self.min_duration_seconds}s"
            )

        # Check sampling rate
        if sfreq not in self.VALID_SAMPLING_RATES:
            errors.append(
                f"Unsupported sampling rate: {sfreq}Hz, "
                f"supported rates: {self.VALID_SAMPLING_RATES}"
            )

        # Check channel count
        if n_channels < self.min_channels:
            errors.append(f"Too few channels: {n_channels}, minimum required: {self.min_channels}")

        # Check data quality
        data = edf_data.data

        # Check for NaN values
        if np.any(np.isnan(data)):
            errors.append("Data contains NaN values")

        # Check for infinite values
        if np.any(np.isinf(data)):
            errors.append("Data contains infinite values")

        # Check for extreme amplitudes (ignoring NaN values)
        max_amplitude: float = float(np.nanmax(np.abs(data)))
        if not np.isnan(max_amplitude) and max_amplitude > self.max_amplitude_v:
            warnings.append(f"Extreme amplitude detected: {max_amplitude * 1e6:.1f}µV")

        # Check for flat channels
        flat_channels = []
        for i in range(n_channels):
            channel_std = np.nanstd(data[i, :])
            if channel_std < 1e-10:  # Essentially zero variance
                flat_channels.append(i)

        if flat_channels:
            warnings.append(f"Flat channels detected: {flat_channels}")
            metadata["flat_channels"] = flat_channels

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings, metadata=metadata
        )
