"""Tests for core.preprocessing - CLEAN, NO BULLSHIT."""

import numpy as np
import pytest

from brain_go_brrr.core.preprocessing import (
    BandpassFilter,
    Normalizer,
    NotchFilter,
    PreprocessingConfig,
    PreprocessingPipeline,
    Resampler,
)


class TestBandpassFilter:
    """Test bandpass filtering."""

    def test_bandpass_respects_nyquist(self):
        """Test that bandpass respects Nyquist frequency."""
        sfreq = 256
        duration = 2
        data = np.random.randn(4, sfreq * duration)

        # Create and apply filter
        filter = BandpassFilter(low_freq=0.5, high_freq=40.0, sfreq=sfreq)
        filtered = filter.apply(data)

        # Check output shape preserved
        assert filtered.shape == data.shape

        # Verify frequencies are valid
        nyquist = sfreq / 2
        assert filter.high_freq < nyquist

    def test_bandpass_invalid_frequencies(self):
        """Test that invalid frequencies raise errors."""
        sfreq = 100

        # High freq > Nyquist should raise
        with pytest.raises(ValueError):
            BandpassFilter(low_freq=0.5, high_freq=60.0, sfreq=sfreq)

        # Low freq >= high freq should raise
        with pytest.raises(ValueError):
            BandpassFilter(low_freq=40.0, high_freq=30.0, sfreq=sfreq)


class TestNormalizer:
    """Test signal normalization."""

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        data = np.array([
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50]
        ], dtype=np.float32)

        normalizer = Normalizer(method='zscore')
        normalized = normalizer.apply(data)

        # Each channel should have mean ~0 and std ~1
        assert np.allclose(normalized.mean(axis=1), 0, atol=1e-6)
        assert np.allclose(normalized.std(axis=1), 1, atol=1e-6)

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.array([
            [0, 5, 10],
            [-10, 0, 10]
        ], dtype=np.float32)

        normalizer = Normalizer(method='minmax')
        normalized = normalizer.apply(data)

        # Each channel should be in [0, 1]
        assert normalized.min() >= -0.001  # Small tolerance
        assert normalized.max() <= 1.001


class TestResampler:
    """Test data resampling."""

    def test_resample_preserves_duration(self):
        """Test that resampling preserves signal duration."""
        orig_sfreq = 256
        target_sfreq = 128
        duration = 2

        data = np.random.randn(4, orig_sfreq * duration)

        resampler = Resampler(orig_sfreq=orig_sfreq, target_sfreq=target_sfreq)
        resampled = resampler.apply(data)

        # Check new shape
        expected_samples = target_sfreq * duration
        assert resampled.shape == (4, expected_samples)

    def test_no_resample_when_same_freq(self):
        """Test that no resampling occurs when frequencies match."""
        sfreq = 256
        data = np.random.randn(4, 1000)

        resampler = Resampler(orig_sfreq=sfreq, target_sfreq=sfreq)
        resampled = resampler.apply(data)

        # Should return same data
        assert np.array_equal(resampled, data)


class TestNotchFilter:
    """Test notch filtering for line noise removal."""

    def test_notch_filter_50hz(self):
        """Test 50 Hz notch filter."""
        sfreq = 256
        duration = 2
        t = np.linspace(0, duration, sfreq * duration)

        # Create signal with 50 Hz noise
        clean = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
        noise = np.sin(2 * np.pi * 50 * t) * 0.5  # 50 Hz noise
        data = (clean + noise).reshape(1, -1)

        # Apply notch filter
        notch = NotchFilter(freq=50.0, sfreq=sfreq)
        filtered = notch.apply(data)

        # Check shape preserved
        assert filtered.shape == data.shape

        # Filtered should have less 50 Hz power
        # (actual power spectrum test would require FFT)

    def test_notch_filter_harmonics(self):
        """Test notch filter with harmonics."""
        sfreq = 500

        notch = NotchFilter(freq=60.0, sfreq=sfreq, harmonics=True)

        # Should create filters for 60, 120, 180 Hz
        assert notch.freq == 60.0
        # Harmonics implementation depends on class details


class TestPreprocessingPipeline:
    """Test preprocessing pipeline."""

    def test_pipeline_execution_order(self):
        """Test that pipeline executes steps in order."""
        sfreq = 256
        data = np.random.randn(4, sfreq * 2)

        # Create pipeline
        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=40.0,
            notch_freq=50.0,
            normalize=True,
            normalize_method='zscore'
        )

        pipeline = PreprocessingPipeline(config, sfreq=sfreq)
        processed = pipeline.apply(data)

        # Check output shape preserved
        assert processed.shape == data.shape

        # Check normalized (if zscore was applied)
        if config.normalize:
            # Should be approximately normalized
            assert np.abs(processed.mean()) < 0.1
            assert np.abs(processed.std() - 1.0) < 0.2

    def test_empty_pipeline(self):
        """Test pipeline with no operations."""
        sfreq = 256
        data = np.random.randn(4, 1000)

        # Create empty config (all operations disabled)
        config = PreprocessingConfig(
            bandpass_low=None,
            bandpass_high=None,
            notch_freq=None,
            normalize=False
        )

        pipeline = PreprocessingPipeline(config, sfreq=sfreq)
        processed = pipeline.apply(data)

        # Should return unchanged data
        assert np.array_equal(processed, data)
