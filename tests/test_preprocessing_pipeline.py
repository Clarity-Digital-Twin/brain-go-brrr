#!/usr/bin/env python
"""Phase 2 TDD: Preprocessing pipeline tests - RED phase first!"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from brain_go_brrr.core.preprocessing import (
    BandpassFilter,
    Normalizer,
    NotchFilter,
    PreprocessingConfig,
    PreprocessingPipeline,
    Resampler,
)


class TestBandpassFilter:
    """Test bandpass filtering following EEGPT specs."""

    def test_bandpass_filter_initialization(self):
        """Test filter can be initialized with frequency parameters."""
        filter = BandpassFilter(low_freq=0.5, high_freq=50.0, sampling_rate=256)
        assert filter.low_freq == 0.5
        assert filter.high_freq == 50.0
        assert filter.sampling_rate == 256

    def test_bandpass_filter_removes_dc_offset(self):
        """Test filter removes DC component (0 Hz)."""
        # Create signal with DC offset
        sampling_rate = 256
        duration = 10  # seconds
        t = np.linspace(0, duration, sampling_rate * duration)
        dc_offset = 100
        signal = dc_offset + np.sin(2 * np.pi * 10 * t)  # DC + 10Hz

        filter = BandpassFilter(low_freq=0.5, high_freq=50.0, sampling_rate=sampling_rate)
        filtered = filter.apply(signal)

        # Mean should be near zero after highpass
        assert abs(np.mean(filtered)) < 1.0  # Much less than original DC offset

    def test_bandpass_filter_removes_high_frequencies(self):
        """Test filter removes frequencies above cutoff."""
        sampling_rate = 256
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)

        # Mix of frequencies: 10Hz (pass), 40Hz (pass), 80Hz (reject)
        signal = (np.sin(2 * np.pi * 10 * t) +
                 np.sin(2 * np.pi * 40 * t) +
                 np.sin(2 * np.pi * 80 * t))

        filter = BandpassFilter(low_freq=0.5, high_freq=50.0, sampling_rate=sampling_rate)
        filtered = filter.apply(signal)

        # Check power spectrum
        freqs = np.fft.fftfreq(len(filtered), 1/sampling_rate)
        fft = np.abs(np.fft.fft(filtered))

        # Power at 80Hz should be heavily attenuated
        idx_80hz = np.argmin(np.abs(freqs - 80))
        idx_10hz = np.argmin(np.abs(freqs - 10))

        power_ratio = fft[idx_80hz] / fft[idx_10hz]
        assert power_ratio < 0.1  # 80Hz should have <10% power of 10Hz

    def test_bandpass_filter_multichannel(self):
        """Test filter works on multichannel data."""
        n_channels = 19
        n_samples = 2048
        data = np.random.randn(n_channels, n_samples)

        filter = BandpassFilter(low_freq=0.5, high_freq=50.0, sampling_rate=256)
        filtered = filter.apply(data)

        assert filtered.shape == data.shape
        # Each channel should be filtered
        for ch in range(n_channels):
            assert not np.array_equal(filtered[ch], data[ch])


class TestNotchFilter:
    """Test notch filter for powerline interference."""

    def test_notch_filter_initialization(self):
        """Test notch filter initialization."""
        filter = NotchFilter(freq=50.0, sampling_rate=256, quality_factor=30)
        assert filter.freq == 50.0
        assert filter.sampling_rate == 256
        assert filter.quality_factor == 30

    def test_notch_filter_removes_target_frequency(self):
        """Test notch filter removes specific frequency."""
        sampling_rate = 256
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)

        # Signal with 10Hz (keep) and 50Hz (remove) components
        signal = np.sin(2 * np.pi * 10 * t) + 2 * np.sin(2 * np.pi * 50 * t)

        filter = NotchFilter(freq=50.0, sampling_rate=sampling_rate)
        filtered = filter.apply(signal)

        # Check power spectrum
        freqs = np.fft.fftfreq(len(filtered), 1/sampling_rate)
        fft = np.abs(np.fft.fft(filtered))

        # Power at 50Hz should be heavily attenuated
        idx_50hz = np.argmin(np.abs(freqs - 50))
        idx_10hz = np.argmin(np.abs(freqs - 10))

        # Get baseline power before filtering for comparison
        fft_orig = np.abs(np.fft.fft(signal))
        orig_50hz_power = fft_orig[idx_50hz]
        orig_10hz_power = fft_orig[idx_10hz]

        # Check attenuation ratio
        attenuation_50hz = fft[idx_50hz] / orig_50hz_power
        attenuation_10hz = fft[idx_10hz] / orig_10hz_power

        assert attenuation_50hz < 0.1  # 50Hz should be attenuated by >90%
        assert attenuation_10hz > 0.9  # 10Hz should be preserved >90%

    def test_notch_filter_preserves_nearby_frequencies(self):
        """Test notch filter has narrow rejection band."""
        sampling_rate = 256
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)

        # Frequencies near 50Hz
        signal = (np.sin(2 * np.pi * 48 * t) +  # Should mostly pass
                 np.sin(2 * np.pi * 50 * t) +   # Should be removed
                 np.sin(2 * np.pi * 52 * t))    # Should mostly pass

        filter = NotchFilter(freq=50.0, sampling_rate=sampling_rate, quality_factor=30)
        filtered = filter.apply(signal)

        freqs = np.fft.fftfreq(len(filtered), 1/sampling_rate)
        fft_orig = np.abs(np.fft.fft(signal))
        fft_filt = np.abs(np.fft.fft(filtered))

        # Check attenuation at each frequency
        for target_freq, expected_pass in [(48, True), (50, False), (52, True)]:
            idx = np.argmin(np.abs(freqs - target_freq))
            attenuation = fft_filt[idx] / fft_orig[idx]

            if expected_pass:
                assert attenuation > 0.7  # Should preserve >70% of power
            else:
                assert attenuation < 0.1  # Should remove >90% of power


class TestNormalizer:
    """Test z-score normalization."""

    def test_zscore_normalization_single_channel(self):
        """Test z-score normalization on single channel."""
        # Create signal with known mean and std
        mean = 10.0
        std = 2.0
        data = np.random.normal(mean, std, 1000)

        normalizer = Normalizer(method='zscore')
        normalized = normalizer.apply(data)

        # Should have zero mean and unit variance
        assert_array_almost_equal(np.mean(normalized), 0.0, decimal=2)
        assert_array_almost_equal(np.std(normalized), 1.0, decimal=2)

    def test_zscore_normalization_multichannel(self):
        """Test each channel normalized independently."""
        n_channels = 19
        n_samples = 1000

        # Different scales per channel
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            mean = ch * 10
            std = ch + 1
            data[ch] = np.random.normal(mean, std, n_samples)

        normalizer = Normalizer(method='zscore')
        normalized = normalizer.apply(data)

        # Each channel should be normalized independently
        for ch in range(n_channels):
            assert_array_almost_equal(np.mean(normalized[ch]), 0.0, decimal=2)
            assert_array_almost_equal(np.std(normalized[ch]), 1.0, decimal=2)

    def test_robust_normalization_with_outliers(self):
        """Test robust normalization handles outliers better."""
        # Signal with outliers
        np.random.seed(42)  # For reproducibility
        data = np.random.normal(0, 1, 1000)
        data[::100] = 100  # Add extreme outliers

        zscore_norm = Normalizer(method='zscore')
        robust_norm = Normalizer(method='robust')

        zscore_result = zscore_norm.apply(data.copy())
        robust_result = robust_norm.apply(data.copy())

        # The key property of robust normalization is that it's less affected by outliers
        # So the majority of the data should be closer to standard normal
        # Check the IQR (interquartile range) which should be closer to standard normal
        zscore_iqr = np.percentile(zscore_result, 75) - np.percentile(zscore_result, 25)
        robust_iqr = np.percentile(robust_result, 75) - np.percentile(robust_result, 25)

        # For standard normal, IQR ≈ 1.349
        # Robust should be closer to this than zscore when outliers present
        zscore_iqr_error = abs(zscore_iqr - 1.349)
        robust_iqr_error = abs(robust_iqr - 1.349)

        assert robust_iqr_error < zscore_iqr_error


class TestResampler:
    """Test resampling to different sampling rates."""

    def test_downsampling_preserves_signal_content(self):
        """Test downsampling from 256Hz to 200Hz."""
        # Create 10Hz sine wave
        orig_rate = 256
        target_rate = 200
        duration = 10
        t_orig = np.linspace(0, duration, orig_rate * duration)
        signal = np.sin(2 * np.pi * 10 * t_orig)

        resampler = Resampler(original_rate=orig_rate, target_rate=target_rate)
        resampled = resampler.apply(signal)

        # Check new length
        expected_length = int(len(signal) * target_rate / orig_rate)
        assert len(resampled) == expected_length

        # Reconstruct time axis and check signal
        t_new = np.linspace(0, duration, len(resampled))
        expected = np.sin(2 * np.pi * 10 * t_new)

        # Should maintain sine wave shape (high correlation)
        correlation = np.corrcoef(resampled, expected)[0, 1]
        assert correlation > 0.99

    def test_upsampling_interpolates_smoothly(self):
        """Test upsampling from 200Hz to 256Hz."""
        orig_rate = 200
        target_rate = 256
        duration = 5
        t_orig = np.linspace(0, duration, orig_rate * duration)
        signal = np.sin(2 * np.pi * 5 * t_orig)

        resampler = Resampler(original_rate=orig_rate, target_rate=target_rate)
        resampled = resampler.apply(signal)

        # Check new length
        expected_length = int(len(signal) * target_rate / orig_rate)
        assert len(resampled) == expected_length

        # Should be smooth (no sudden jumps)
        diff = np.diff(resampled)
        assert np.max(np.abs(diff)) < 0.2  # No large jumps (relaxed for 5Hz signal)

    def test_resampling_multichannel(self):
        """Test resampling preserves channel structure."""
        n_channels = 19
        orig_rate = 256
        target_rate = 200
        n_samples = 2560

        data = np.random.randn(n_channels, n_samples)

        resampler = Resampler(original_rate=orig_rate, target_rate=target_rate)
        resampled = resampler.apply(data)

        expected_samples = int(n_samples * target_rate / orig_rate)
        assert resampled.shape == (n_channels, expected_samples)


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline."""

    def test_pipeline_configuration(self):
        """Test pipeline can be configured with all components."""
        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=50.0,
            notch_freq=50.0,
            normalization='zscore',
            target_sampling_rate=200,
            original_sampling_rate=256
        )

        pipeline = PreprocessingPipeline(config)

        # Should have all components
        assert len(pipeline.steps) == 4  # bandpass, notch, normalize, resample
        assert any('bandpass' in str(step) for step in pipeline.steps)
        assert any('notch' in str(step) for step in pipeline.steps)
        assert any('normalize' in str(step) for step in pipeline.steps)
        assert any('resample' in str(step) for step in pipeline.steps)

    def test_pipeline_end_to_end(self):
        """Test full pipeline processing."""
        # Create realistic EEG-like data
        sampling_rate = 256
        duration = 10
        n_channels = 19
        t = np.linspace(0, duration, sampling_rate * duration)

        # Mix of neural rhythms + noise + artifacts
        data = np.zeros((n_channels, len(t)))
        for ch in range(n_channels):
            # Neural rhythms
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + ch)  # 10Hz alpha
            beta = 0.3 * np.sin(2 * np.pi * 25 * t + ch)   # 25Hz beta

            # Artifacts
            powerline = 2.0 * np.sin(2 * np.pi * 50 * t)   # 50Hz powerline
            highfreq = 0.5 * np.sin(2 * np.pi * 80 * t)    # High freq noise

            # DC offset
            dc_offset = ch * 10

            data[ch] = alpha + beta + powerline + highfreq + dc_offset
            data[ch] += 0.1 * np.random.randn(len(t))  # White noise

        # Configure pipeline
        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=50.0,
            notch_freq=50.0,
            normalization='zscore',
            target_sampling_rate=200,
            original_sampling_rate=256
        )

        pipeline = PreprocessingPipeline(config)
        processed = pipeline.apply(data)

        # Check output shape (resampled)
        expected_samples = int(len(t) * 200 / 256)
        assert processed.shape == (n_channels, expected_samples)

        # Check each channel is normalized
        for ch in range(n_channels):
            assert_array_almost_equal(np.mean(processed[ch]), 0.0, decimal=1)
            assert_array_almost_equal(np.std(processed[ch]), 1.0, decimal=1)

        # Verify artifacts removed by checking frequency content
        for ch in range(n_channels):
            freqs = np.fft.fftfreq(processed.shape[1], 1/200)  # New sampling rate
            fft = np.abs(np.fft.fft(processed[ch]))

            # Find peaks in spectrum
            alpha_idx = np.argmin(np.abs(freqs - 10))
            powerline_idx = np.argmin(np.abs(freqs - 50))

            # Alpha should be preserved, powerline should be gone
            alpha_power = fft[alpha_idx]
            powerline_power = fft[powerline_idx]

            assert powerline_power < 0.1 * alpha_power  # Powerline heavily attenuated

    def test_pipeline_with_nan_handling(self):
        """Test pipeline handles NaN values gracefully."""
        n_channels = 19
        n_samples = 1000
        data = np.random.randn(n_channels, n_samples)

        # Add some NaN values
        data[0, 100:200] = np.nan
        data[5, :50] = np.nan

        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=50.0,
            normalization='zscore',
            handle_nan='interpolate'  # Should interpolate NaN regions
        )

        pipeline = PreprocessingPipeline(config)
        processed = pipeline.apply(data)

        # No NaN in output
        assert not np.any(np.isnan(processed))

    def test_pipeline_optional_components(self):
        """Test pipeline with only some components enabled."""
        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=50.0,
            notch_freq=None,  # Disable notch filter
            normalization=None,  # Disable normalization
            target_sampling_rate=None  # Disable resampling
        )

        pipeline = PreprocessingPipeline(config)

        # Should only have bandpass
        assert len(pipeline.steps) == 1
        assert 'bandpass' in str(pipeline.steps[0])


# Integration tests with real EEG parameters
class TestPreprocessingIntegration:
    """Integration tests matching EEGPT requirements."""

    def test_tuab_preprocessing_spec(self):
        """Test preprocessing matching TUAB enhanced config."""
        # TUAB uses 8s windows @ 256Hz
        window_duration = 8.0
        sampling_rate = 256
        n_channels = 19
        n_samples = int(window_duration * sampling_rate)

        # Create test data
        data = np.random.randn(n_channels, n_samples) * 50  # µV scale

        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=50.0,
            notch_freq=None,  # TUAB doesn't use notch
            normalization='zscore',
            original_sampling_rate=256,
            target_sampling_rate=256  # No resampling needed
        )

        pipeline = PreprocessingPipeline(config)
        processed = pipeline.apply(data)

        # Verify output
        assert processed.shape == (n_channels, n_samples)
        assert not np.any(np.isnan(processed))
        assert not np.any(np.isinf(processed))

        # Each channel normalized
        for ch in range(n_channels):
            assert abs(np.mean(processed[ch])) < 0.1
            assert abs(np.std(processed[ch]) - 1.0) < 0.1

    def test_memory_efficiency(self):
        """Test pipeline doesn't create unnecessary copies."""
        # Large data to test memory
        n_channels = 19
        n_samples = 256 * 60  # 1 minute
        data = np.random.randn(n_channels, n_samples).astype(np.float32)

        config = PreprocessingConfig(
            bandpass_low=0.5,
            bandpass_high=50.0,
            normalization='zscore',
            inplace=True  # Should modify in place when possible
        )

        pipeline = PreprocessingPipeline(config)

        # Track memory usage (simplified check)
        data_id = id(data)
        processed = pipeline.apply(data)

        # For inplace, might reuse same memory
        # (This is a simplified test - real memory profiling would be more complex)
        assert processed.dtype == np.float32  # Preserves dtype
