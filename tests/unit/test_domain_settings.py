"""Tests for domain settings - REAL BEHAVIORAL TESTS, NO MOCKING."""

import pytest

from brain_go_brrr.domain.abnormal.settings import (
    AbnormalitySettings,
    FeatureSettings,
    QualitySettings,
)


class TestAbnormalitySettings:
    """Test AbnormalitySettings value object behavior."""

    def test_default_settings_are_valid(self):
        """Test default settings pass validation."""
        settings = AbnormalitySettings()
        settings.validate()  # Should not raise

        # Check defaults are sensible
        assert settings.abnormal_threshold == 0.5
        assert settings.confidence_threshold == 0.7
        assert settings.urgent_threshold > settings.expedite_threshold
        assert settings.expedite_threshold > settings.routine_threshold

    def test_immutability(self):
        """Test settings are truly immutable (frozen dataclass)."""
        settings = AbnormalitySettings()

        with pytest.raises(AttributeError):
            settings.abnormal_threshold = 0.9

    def test_validate_abnormal_threshold_bounds(self):
        """Test abnormal_threshold must be in [0,1]."""
        # Below bounds
        with pytest.raises(ValueError, match="abnormal_threshold must be in"):
            settings = AbnormalitySettings(abnormal_threshold=-0.1)
            settings.validate()

        # Above bounds
        with pytest.raises(ValueError, match="abnormal_threshold must be in"):
            settings = AbnormalitySettings(abnormal_threshold=1.1)
            settings.validate()

        # Edge cases - should work
        settings = AbnormalitySettings(abnormal_threshold=0.0)
        settings.validate()

        settings = AbnormalitySettings(abnormal_threshold=1.0)
        settings.validate()

    def test_validate_confidence_threshold_bounds(self):
        """Test confidence_threshold must be in [0,1]."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            settings = AbnormalitySettings(confidence_threshold=1.5)
            settings.validate()

        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            settings = AbnormalitySettings(confidence_threshold=-0.5)
            settings.validate()

    def test_validate_triage_threshold_ordering(self):
        """Test triage thresholds must be properly ordered."""
        # Urgent <= Expedite (should fail)
        with pytest.raises(ValueError, match="urgent_threshold must be > expedite_threshold"):
            settings = AbnormalitySettings(
                urgent_threshold=0.8,
                expedite_threshold=0.8,
                routine_threshold=0.7,
            )
            settings.validate()

        # Expedite <= Routine (should fail)
        with pytest.raises(ValueError, match="expedite_threshold must be > routine_threshold"):
            settings = AbnormalitySettings(
                urgent_threshold=0.95,
                expedite_threshold=0.7,
                routine_threshold=0.7,
            )
            settings.validate()

        # Valid ordering
        settings = AbnormalitySettings(
            urgent_threshold=0.95,
            expedite_threshold=0.85,
            routine_threshold=0.70,
        )
        settings.validate()  # Should pass

    def test_validate_window_duration(self):
        """Test window_duration must be positive."""
        with pytest.raises(ValueError, match="window_duration must be positive"):
            settings = AbnormalitySettings(window_duration=0.0)
            settings.validate()

        with pytest.raises(ValueError, match="window_duration must be positive"):
            settings = AbnormalitySettings(window_duration=-1.0)
            settings.validate()

        # Valid duration
        settings = AbnormalitySettings(window_duration=4.0)
        settings.validate()

    def test_validate_window_overlap(self):
        """Test window_overlap must be in [0,1)."""
        # Too low
        with pytest.raises(ValueError, match="window_overlap must be in"):
            settings = AbnormalitySettings(window_overlap=-0.1)
            settings.validate()

        # Too high (>=1)
        with pytest.raises(ValueError, match="window_overlap must be in"):
            settings = AbnormalitySettings(window_overlap=1.0)
            settings.validate()

        # Valid values
        settings = AbnormalitySettings(window_overlap=0.0)
        settings.validate()

        settings = AbnormalitySettings(window_overlap=0.5)
        settings.validate()

        settings = AbnormalitySettings(window_overlap=0.99)
        settings.validate()

    def test_all_parameters_configurable(self):
        """Test all parameters can be configured."""
        settings = AbnormalitySettings(
            abnormal_threshold=0.6,
            confidence_threshold=0.8,
            min_confidence=0.4,
            urgent_threshold=0.96,
            expedite_threshold=0.86,
            routine_threshold=0.71,
            window_duration=5.0,
            window_overlap=0.25,
            min_windows=5,
            min_quality_score=0.6,
            artifact_threshold=0.4,
        )

        # Verify all set correctly
        assert settings.abnormal_threshold == 0.6
        assert settings.confidence_threshold == 0.8
        assert settings.min_confidence == 0.4
        assert settings.urgent_threshold == 0.96
        assert settings.expedite_threshold == 0.86
        assert settings.routine_threshold == 0.71
        assert settings.window_duration == 5.0
        assert settings.window_overlap == 0.25
        assert settings.min_windows == 5
        assert settings.min_quality_score == 0.6
        assert settings.artifact_threshold == 0.4

        # Should still validate
        settings.validate()


class TestQualitySettings:
    """Test QualitySettings value object behavior."""

    def test_default_settings(self):
        """Test default quality settings are sensible."""
        settings = QualitySettings()

        # Check defaults
        assert settings.flat_channel_threshold == 1e-6
        assert settings.noise_multiplier == 5.0
        assert settings.min_unique_values == 100

        # Artifact thresholds in microvolts
        assert settings.high_amplitude_threshold == 100e-6
        assert settings.jump_threshold == 50e-6

        # Weights should sum to 1.0
        assert settings.bad_channel_weight + settings.artifact_weight == 1.0

        # Minimum requirements
        assert settings.min_channels == 4
        assert settings.min_duration_seconds == 10.0
        assert settings.min_sampling_rate == 50.0

    def test_immutability(self):
        """Test quality settings are immutable."""
        settings = QualitySettings()

        with pytest.raises(AttributeError):
            settings.min_channels = 10

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        settings = QualitySettings(
            flat_channel_threshold=1e-7,
            high_amplitude_threshold=150e-6,
            jump_threshold=75e-6,
        )

        assert settings.flat_channel_threshold == 1e-7
        assert settings.high_amplitude_threshold == 150e-6
        assert settings.jump_threshold == 75e-6

    def test_weight_configuration(self):
        """Test weight configuration for quality scoring."""
        # Custom weights (don't have to sum to 1.0 necessarily)
        settings = QualitySettings(
            bad_channel_weight=0.7,
            artifact_weight=0.3,
        )

        assert settings.bad_channel_weight == 0.7
        assert settings.artifact_weight == 0.3

    def test_minimum_requirements(self):
        """Test minimum requirement configuration."""
        settings = QualitySettings(
            min_channels=8,
            min_duration_seconds=30.0,
            min_sampling_rate=128.0,
        )

        assert settings.min_channels == 8
        assert settings.min_duration_seconds == 30.0
        assert settings.min_sampling_rate == 128.0


class TestFeatureSettings:
    """Test FeatureSettings value object behavior."""

    def test_default_frequency_bands(self):
        """Test default frequency bands match standard EEG bands."""
        settings = FeatureSettings()

        # Standard EEG frequency bands
        assert settings.delta_band == (0.5, 4.0)
        assert settings.theta_band == (4.0, 8.0)
        assert settings.alpha_band == (8.0, 13.0)
        assert settings.beta_band == (13.0, 30.0)
        assert settings.gamma_band == (30.0, 45.0)

    def test_default_preprocessing_parameters(self):
        """Test default preprocessing parameters."""
        settings = FeatureSettings()

        # Bandpass filter
        assert settings.bandpass_low == 0.5
        assert settings.bandpass_high == 45.0

        # Notch filter for line noise
        assert settings.notch_freq == 50.0  # European standard

    def test_immutability(self):
        """Test feature settings are immutable."""
        settings = FeatureSettings()

        with pytest.raises(AttributeError):
            settings.window_size = 8.0

        # Tuples are immutable too
        with pytest.raises(AttributeError):
            settings.delta_band = (0.1, 3.5)

    def test_custom_frequency_bands(self):
        """Test custom frequency band configuration."""
        settings = FeatureSettings(
            delta_band=(0.1, 3.5),
            theta_band=(3.5, 7.5),
            alpha_band=(7.5, 12.5),
            beta_band=(12.5, 28.0),
            gamma_band=(28.0, 50.0),
        )

        assert settings.delta_band == (0.1, 3.5)
        assert settings.theta_band == (3.5, 7.5)
        assert settings.alpha_band == (7.5, 12.5)
        assert settings.beta_band == (12.5, 28.0)
        assert settings.gamma_band == (28.0, 50.0)

    def test_window_parameters(self):
        """Test window parameter configuration."""
        settings = FeatureSettings(
            window_size=8.0,
            window_overlap=0.75,
        )

        assert settings.window_size == 8.0
        assert settings.window_overlap == 0.75

    def test_fft_and_entropy_parameters(self):
        """Test FFT and entropy parameter configuration."""
        settings = FeatureSettings(
            n_fft_bins=512,
            entropy_bins=100,
        )

        assert settings.n_fft_bins == 512
        assert settings.entropy_bins == 100

    def test_notch_freq_for_us_standard(self):
        """Test configuring notch frequency for US power grid."""
        settings = FeatureSettings(
            notch_freq=60.0,  # US standard
        )

        assert settings.notch_freq == 60.0

    def test_all_parameters_configurable(self):
        """Test all feature settings parameters are configurable."""
        settings = FeatureSettings(
            window_size=2.0,
            window_overlap=0.25,
            delta_band=(0.3, 3.8),
            theta_band=(3.8, 7.8),
            alpha_band=(7.8, 12.8),
            beta_band=(12.8, 29.0),
            gamma_band=(29.0, 48.0),
            n_fft_bins=128,
            entropy_bins=25,
            bandpass_low=0.3,
            bandpass_high=48.0,
            notch_freq=60.0,
        )

        # Verify all set
        assert settings.window_size == 2.0
        assert settings.window_overlap == 0.25
        assert settings.delta_band == (0.3, 3.8)
        assert settings.theta_band == (3.8, 7.8)
        assert settings.alpha_band == (7.8, 12.8)
        assert settings.beta_band == (12.8, 29.0)
        assert settings.gamma_band == (29.0, 48.0)
        assert settings.n_fft_bins == 128
        assert settings.entropy_bins == 25
        assert settings.bandpass_low == 0.3
        assert settings.bandpass_high == 48.0
        assert settings.notch_freq == 60.0


class TestSettingsInteraction:
    """Test interaction between different settings classes."""

    def test_settings_independence(self):
        """Test settings classes are independent of each other."""
        abnormal = AbnormalitySettings(window_duration=4.0)
        feature = FeatureSettings(window_size=8.0)

        # Different window parameters shouldn't affect each other
        assert abnormal.window_duration == 4.0
        assert feature.window_size == 8.0

    def test_combined_usage_pattern(self):
        """Test typical usage pattern with multiple settings."""
        # Configuration for a complete analysis pipeline
        abnormal_settings = AbnormalitySettings(
            abnormal_threshold=0.6,
            urgent_threshold=0.9,
            expedite_threshold=0.8,
            routine_threshold=0.6,
        )

        quality_settings = QualitySettings(
            min_channels=8,
            min_duration_seconds=20.0,
        )

        feature_settings = FeatureSettings(
            window_size=4.0,
            bandpass_high=40.0,
        )

        # Validate abnormal settings
        abnormal_settings.validate()

        # Use settings together (simulating a pipeline)
        assert abnormal_settings.window_duration == feature_settings.window_size
        assert quality_settings.min_duration_seconds >= abnormal_settings.window_duration

    def test_settings_hashability(self):
        """Test settings can be used as dict keys (frozen dataclass)."""
        settings1 = AbnormalitySettings(abnormal_threshold=0.5)
        settings2 = AbnormalitySettings(abnormal_threshold=0.5)
        settings3 = AbnormalitySettings(abnormal_threshold=0.6)

        # Can use as dict keys
        config_map = {
            settings1: "config1",
            settings3: "config3",
        }

        # Equal settings have same hash
        assert hash(settings1) == hash(settings2)
        assert settings1 == settings2

        # Different settings have different hash (probably)
        assert settings1 != settings3

        # Can retrieve with equal instance
        assert config_map[settings2] == "config1"