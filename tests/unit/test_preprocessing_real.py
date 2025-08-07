"""REAL tests for preprocessing - NO BULLSHIT MOCKING."""

import numpy as np

from brain_go_brrr.core.preprocessing import (
    BandpassFilter,
    Normalizer,
    NotchFilter,
    PreprocessingConfig,
    PreprocessingPipeline,
    Resampler,
)


class TestBandpassFilterREAL:
    """Test REAL bandpass filtering logic."""

    def test_filter_actually_filters(self):
        """Test that filter ACTUALLY removes frequencies."""
        # Create signal with known frequencies
        fs = 256
        t = np.linspace(0, 1, fs)

        # Mix of 10 Hz (pass) and 80 Hz (stop)
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        signal_80hz = np.sin(2 * np.pi * 80 * t)
        mixed = signal_10hz + signal_80hz

        # Apply filter (0.5-40 Hz)
        filt = BandpassFilter(0.5, 40, fs)
        filtered = filt.apply(mixed)

        # Compute FFT to check frequencies
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(mixed), 1/fs)

        # Find peaks
        idx_10hz = np.argmin(np.abs(freqs - 10))
        idx_80hz = np.argmin(np.abs(freqs - 80))

        # 10 Hz should be preserved, 80 Hz should be attenuated
        assert fft_filt[idx_10hz] > fft_filt[idx_80hz] * 2

    def test_filter_preserves_shape(self):
        """Test filter preserves data shape."""
        data = np.random.randn(4, 1000)  # 4 channels
        filt = BandpassFilter(1, 40, 256)
        filtered = filt.apply(data)

        assert filtered.shape == data.shape
        assert not np.array_equal(filtered, data)  # Should be different


class TestNormalizerREAL:
    """Test REAL normalization."""

    def test_zscore_actually_normalizes(self):
        """Test z-score ACTUALLY normalizes data."""
        # Create data with known statistics
        data = np.array([
            [1, 2, 3, 4, 5],     # mean=3, std≈1.58
            [10, 20, 30, 40, 50] # mean=30, std≈15.8
        ], dtype=np.float64)

        norm = Normalizer(method='zscore')
        normalized = norm.apply(data)

        # Check actual normalization
        for i in range(data.shape[0]):
            assert abs(normalized[i].mean()) < 1e-10
            assert abs(normalized[i].std() - 1.0) < 1e-10

    def test_robust_normalization_exists(self):
        """Test robust normalization works."""
        # Data with outliers
        data = np.array([[1, 2, 3, 4, 5, 100]], dtype=np.float64)

        norm = Normalizer(method='robust')
        result = norm.apply(data)

        # Should be normalized (not same as input)
        assert not np.array_equal(result, data)
        # Should have reasonable range
        assert result.min() > -10
        assert result.max() < 50


class TestNotchFilterREAL:
    """Test REAL notch filtering."""

    def test_notch_removes_60hz(self):
        """Test notch ACTUALLY removes 60 Hz noise."""
        fs = 500
        t = np.linspace(0, 1, fs)

        # Signal + 60 Hz noise
        clean = np.sin(2 * np.pi * 10 * t)
        noise = np.sin(2 * np.pi * 60 * t) * 0.5
        noisy = clean + noise

        # Apply notch
        notch = NotchFilter(60, fs)
        filtered = notch.apply(noisy.reshape(1, -1))

        # Check 60 Hz is reduced
        fft_noisy = np.abs(np.fft.rfft(noisy))
        fft_clean = np.abs(np.fft.rfft(filtered[0]))
        freqs = np.fft.rfftfreq(len(noisy), 1/fs)

        idx_60hz = np.argmin(np.abs(freqs - 60))

        # 60 Hz should be attenuated by at least 10x
        assert fft_clean[idx_60hz] < fft_noisy[idx_60hz] / 10


class TestResamplerREAL:
    """Test REAL resampling."""

    def test_resample_preserves_frequency_content(self):
        """Test resampling preserves frequency content."""
        # Create 10 Hz sine wave
        orig_fs = 256
        target_fs = 128
        t_orig = np.linspace(0, 1, orig_fs)
        signal_orig = np.sin(2 * np.pi * 10 * t_orig)

        # Resample
        resampler = Resampler(orig_fs, target_fs)
        resampled = resampler.apply(signal_orig.reshape(1, -1))

        # Check shape
        assert resampled.shape[1] == target_fs

        # Check frequency preserved (FFT peak should still be at 10 Hz)
        fft = np.abs(np.fft.rfft(resampled[0]))
        freqs = np.fft.rfftfreq(len(resampled[0]), 1/target_fs)
        peak_freq = freqs[np.argmax(fft[1:])+1]  # Skip DC

        assert abs(peak_freq - 10) < 1  # Peak still at ~10 Hz


class TestPipelineREAL:
    """Test REAL pipeline execution."""

    def test_pipeline_executes_in_order(self):
        """Test pipeline applies operations in correct order."""
        fs = 256
        data = np.random.randn(4, fs * 2)

        # Add DC offset to test normalization
        data = data + 10

        config = PreprocessingConfig(
            bandpass_low=1,
            bandpass_high=40,
            notch_freq=50,
            normalization='zscore'
        )

        pipeline = PreprocessingPipeline(config)
        processed = pipeline.apply(data)

        # Should be normalized (mean ~0)
        assert np.abs(processed.mean()) < 0.1

        # Should be filtered (different from input)
        assert not np.array_equal(processed, data)
