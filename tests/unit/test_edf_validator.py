"""Test-driven development for EDF file validation.
Tests written BEFORE implementation.
"""
import tempfile
from pathlib import Path

import numpy as np


class TestEDFValidator:
    """Test EDF file validation."""

    def test_validates_file_exists(self):
        """Should fail validation if file doesn't exist."""
        # Given
        non_existent = Path("/tmp/does_not_exist.edf")

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate(non_existent)

        # Then
        assert result.is_valid is False
        assert "not found" in result.errors[0].lower()

    def test_validates_file_extension(self):
        """Should only accept .edf files."""
        # Given
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            wrong_extension = Path(f.name)

            # When
            from brain_go_brrr.core.edf_validator import EDFValidator
            validator = EDFValidator()
            result = validator.validate(wrong_extension)

            # Then
            assert result.is_valid is False
            assert "extension" in result.errors[0].lower()

    def test_validates_minimum_duration(self):
        """Should require minimum 60 seconds duration."""
        # Given - mock EDF with short duration
        mock_edf_data = MockEDFData(duration_seconds=30, sfreq=256, n_channels=20)

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator(min_duration_seconds=60)
        result = validator.validate_data(mock_edf_data)

        # Then
        assert result.is_valid is False
        assert "short" in result.errors[0].lower()
        assert "30" in result.errors[0]  # Should mention actual duration

    def test_validates_sampling_rate(self):
        """Should validate sampling rate is supported."""
        # Given - non-standard sampling rate
        mock_edf_data = MockEDFData(duration_seconds=300, sfreq=333, n_channels=20)

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate_data(mock_edf_data)

        # Then
        assert result.is_valid is False
        assert "sampling rate" in result.errors[0].lower()
        assert "333" in result.errors[0]

    def test_accepts_standard_sampling_rates(self):
        """Should accept common EEG sampling rates."""
        # Given
        valid_rates = [100, 128, 200, 250, 256, 500, 512, 1000]

        for sfreq in valid_rates:
            mock_edf = MockEDFData(duration_seconds=300, sfreq=sfreq, n_channels=20)

            # When
            from brain_go_brrr.core.edf_validator import EDFValidator
            validator = EDFValidator()
            result = validator.validate_data(mock_edf)

            # Then - no sampling rate errors
            assert not any("sampling rate" in err.lower() for err in result.errors)

    def test_validates_channel_count(self):
        """Should require minimum number of channels."""
        # Given - too few channels
        mock_edf_data = MockEDFData(duration_seconds=300, sfreq=256, n_channels=5)

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator(min_channels=19)
        result = validator.validate_data(mock_edf_data)

        # Then
        assert result.is_valid is False
        assert "channels" in result.errors[0].lower()
        assert "5" in result.errors[0]  # Actual count
        assert "19" in result.errors[0]  # Required count

    def test_detects_nan_values(self):
        """Should detect NaN values in data."""
        # Given
        mock_edf = MockEDFData(duration_seconds=60, sfreq=256, n_channels=20)
        mock_edf.data[5, 1000:1100] = np.nan  # Inject NaNs

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate_data(mock_edf)

        # Then
        assert result.is_valid is False
        assert "nan" in result.errors[0].lower()

    def test_detects_infinite_values(self):
        """Should detect infinite values in data."""
        # Given
        mock_edf = MockEDFData(duration_seconds=60, sfreq=256, n_channels=20)
        mock_edf.data[10, 500] = np.inf

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate_data(mock_edf)

        # Then
        assert result.is_valid is False
        assert "infinite" in result.errors[0].lower()

    def test_warns_on_extreme_amplitudes(self):
        """Should warn about extreme amplitude values."""
        # Given - extreme but not infinite values
        mock_edf = MockEDFData(duration_seconds=60, sfreq=256, n_channels=20)
        mock_edf.data[0, :] = 5000e-6  # 5mV - very high for EEG

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate_data(mock_edf)

        # Then
        assert result.is_valid is True  # Warning, not error
        assert len(result.warnings) > 0
        assert "amplitude" in result.warnings[0].lower()

    def test_detects_flat_channels(self):
        """Should detect flat (zero variance) channels."""
        # Given
        mock_edf = MockEDFData(duration_seconds=60, sfreq=256, n_channels=20)
        mock_edf.data[3, :] = 0.0  # Flat channel
        mock_edf.data[7, :] = 1e-6  # Constant non-zero

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate_data(mock_edf)

        # Then
        assert result.is_valid is True  # Just warning
        assert len(result.warnings) > 0
        assert "flat" in result.warnings[0].lower()
        assert result.metadata["flat_channels"] == [3, 7]

    def test_validates_complete_valid_file(self):
        """Should pass validation for good EDF file."""
        # Given - all parameters good
        mock_edf = MockEDFData(
            duration_seconds=300,
            sfreq=256,
            n_channels=20,
            amplitude_uv=50  # Normal EEG amplitude
        )

        # When
        from brain_go_brrr.core.edf_validator import EDFValidator
        validator = EDFValidator()
        result = validator.validate_data(mock_edf)

        # Then
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.metadata["duration_seconds"] == 300
        assert result.metadata["sampling_rate"] == 256
        assert result.metadata["n_channels"] == 20


class TestValidationResult:
    """Test validation result structure."""

    def test_result_contains_all_fields(self):
        """Validation result should have all necessary fields."""
        # Given/When
        from brain_go_brrr.core.edf_validator import ValidationResult
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["High amplitude detected"],
            metadata={"duration_seconds": 300}
        )

        # Then
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "metadata")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)


# Mock data for testing
class MockEDFData:
    """Mock EDF data for testing validation logic."""

    def __init__(
        self,
        duration_seconds: float,
        sfreq: float,
        n_channels: int,
        amplitude_uv: float = 50.0
    ):
        self.duration = duration_seconds
        self.sfreq = sfreq
        self.n_channels = n_channels

        # Generate realistic EEG data
        n_samples = int(duration_seconds * sfreq)
        self.data = np.random.randn(n_channels, n_samples) * amplitude_uv * 1e-6

        # Channel names
        self.ch_names = [f"CH{i+1}" for i in range(n_channels)]

    @property
    def n_times(self):
        return self.data.shape[1]
