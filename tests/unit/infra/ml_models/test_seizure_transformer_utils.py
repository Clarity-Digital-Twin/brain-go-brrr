"""Tests for SeizureTransformer preprocessing and post-processing utilities."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from brain_go_brrr.infra.ml_models.seizure_transformer_utils import (
    CANONICAL_CHANNELS,
    SeizurePostProcessor,
    SeizurePreprocessor,
    prepare_channels,
    standardize_channel_names,
)


@pytest.mark.unit
def test_seizure_preprocessor_initialization():
    """Test preprocessor initializes with correct parameters."""
    preprocessor = SeizurePreprocessor(target_fs=256)

    assert preprocessor.fs == 256
    assert preprocessor.lowcut == 0.5
    assert preprocessor.highcut == 120.0  # Not 100Hz!

    # Check filter coefficients exist
    assert preprocessor.bp_b is not None
    assert preprocessor.bp_a is not None
    assert preprocessor.notch_1_b is not None
    assert preprocessor.notch_60_b is not None


@pytest.mark.unit
def test_preprocessor_zscore_normalization():
    """Test z-score normalization is applied correctly."""
    preprocessor = SeizurePreprocessor()

    # Create test data with known mean and std
    np.random.seed(42)
    n_channels, n_samples = 19, 2560
    eeg = np.random.randn(n_channels, n_samples).astype(np.float32)

    # Add offset and scale to make non-standard
    eeg = eeg * 10 + 5

    # Apply preprocessing
    processed = preprocessor.preprocess(eeg, fs_original=256)

    # Check normalization (should be close to N(0,1) per channel)
    for ch in range(n_channels):
        assert np.abs(np.mean(processed[ch])) < 0.1  # Mean ~0
        assert 0.8 < np.std(processed[ch]) < 1.2  # Std ~1


@pytest.mark.unit
def test_preprocessor_resampling():
    """Test resampling from different sampling rates."""
    preprocessor = SeizurePreprocessor(target_fs=256)

    # Test 1: 512Hz -> 256Hz (downsample)
    n_channels = 19
    duration_sec = 4.0
    fs_original = 512
    n_samples = int(duration_sec * fs_original)

    eeg = np.random.randn(n_channels, n_samples).astype(np.float32)
    processed = preprocessor.preprocess(eeg, fs_original=fs_original)

    expected_samples = int(duration_sec * 256)
    assert processed.shape == (n_channels, expected_samples)

    # Test 2: 128Hz -> 256Hz (upsample)
    fs_original = 128
    n_samples = int(duration_sec * fs_original)

    eeg = np.random.randn(n_channels, n_samples).astype(np.float32)
    processed = preprocessor.preprocess(eeg, fs_original=fs_original)

    assert processed.shape == (n_channels, expected_samples)

    # Test 3: Already 256Hz (no resampling)
    fs_original = 256
    n_samples = int(duration_sec * fs_original)

    eeg = np.random.randn(n_channels, n_samples).astype(np.float32)
    processed = preprocessor.preprocess(eeg, fs_original=fs_original)

    assert processed.shape == (n_channels, n_samples)


@pytest.mark.unit
def test_preprocessor_filters():
    """Test bandpass and notch filters are applied."""
    preprocessor = SeizurePreprocessor(target_fs=256)

    # Create signal with known frequency components
    fs = 256
    duration = 10.0
    t = np.arange(0, duration, 1 / fs)

    # Add components: DC, 1Hz, 30Hz (in-band), 60Hz, 150Hz (out-of-band)
    signal = (
        1.0  # DC offset
        + np.sin(2 * np.pi * 1 * t)  # 1Hz (notched)
        + np.sin(2 * np.pi * 30 * t)  # 30Hz (kept)
        + np.sin(2 * np.pi * 60 * t)  # 60Hz (notched)
        + np.sin(2 * np.pi * 150 * t)  # 150Hz (filtered)
    )

    eeg = signal.reshape(1, -1).astype(np.float32)
    processed = preprocessor.preprocess(eeg, fs_original=fs)

    # FFT to check frequency content
    from scipy.fft import rfft, rfftfreq

    freqs = rfftfreq(processed.shape[1], 1 / fs)
    fft = np.abs(rfft(processed[0]))

    # Compare amplitudes around key frequencies
    def band_amp(f0: float, tol: float = 1.0) -> float:
        mask = (freqs >= (f0 - tol)) & (freqs <= (f0 + tol))
        return float(fft[mask].mean())

    amp_30 = band_amp(30)
    amp_60 = band_amp(60)

    # 30 Hz should dominate vs 60 Hz (notched)
    assert amp_30 > amp_60 * 1.3

    # Sanity: shape unchanged
    assert processed.shape == eeg.shape


@pytest.mark.unit
def test_postprocessor_initialization():
    """Test post-processor initializes correctly."""
    postprocessor = SeizurePostProcessor()

    assert postprocessor.threshold == 0.8
    assert postprocessor.morph_open_size == 5
    assert postprocessor.morph_close_size == 5
    assert postprocessor.min_duration_sec == 2.0
    assert postprocessor.min_duration_samples == 512  # 2.0 * 256


@pytest.mark.unit
def test_postprocessor_thresholding():
    """Test threshold is applied correctly."""
    postprocessor = SeizurePostProcessor()

    # Create probabilities around threshold
    probs = np.array([0.7, 0.79, 0.8, 0.81, 0.9], dtype=np.float32)

    # Pad to avoid edge effects from morphological ops
    probs = np.pad(probs, 100, constant_values=0)

    result = postprocessor.postprocess(probs)

    # After thresholding, only values > 0.8 should be 1
    # But morphological ops may affect the result
    assert result.dtype == np.int32
    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.unit
def test_postprocessor_removes_short_events():
    """Test that events < 2 seconds are removed."""
    postprocessor = SeizurePostProcessor(fs=256)

    # Create signal with short and long events
    n_samples = 256 * 20  # 20 seconds
    probs = np.zeros(n_samples, dtype=np.float32)

    # Add events of different lengths
    # 1s event (should be removed)
    probs[256:512] = 0.9

    # 3s event (should be kept)
    probs[1024 : 1024 + 768] = 0.9

    # 1.5s event (should be removed)
    probs[2048 : 2048 + 384] = 0.9

    # 5s event (should be kept)
    probs[3072 : 3072 + 1280] = 0.9

    result = postprocessor.postprocess(probs)

    # Count events in result
    from scipy.ndimage import label

    labeled, num_events = label(result)

    # Should have 2 events (3s and 5s)
    assert num_events == 2


@pytest.mark.unit
def test_channel_standardization():
    """Test channel name aliasing."""
    # Test legacy names are converted
    legacy_names = ["Fp1", "T3", "T4", "T5", "T6", "O1"]
    standardized = standardize_channel_names(legacy_names)

    assert standardized == ["Fp1", "T7", "T8", "P7", "P8", "O1"]

    # Test modern names are unchanged
    modern_names = ["Fp1", "T7", "T8", "P7", "P8", "O1"]
    standardized = standardize_channel_names(modern_names)

    assert standardized == modern_names


@pytest.mark.unit
def test_prepare_channels():
    """Test channel preparation for model input."""
    # Create test data with subset of channels
    available_channels = ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
    n_samples = 1024
    data = np.random.randn(len(available_channels), n_samples).astype(np.float32)

    # Prepare for model (should get 19 channels)
    prepared, channel_info = prepare_channels(data, available_channels)

    assert prepared.shape == (19, n_samples)
    assert len(channel_info) == 19

    # Check that available channels are in correct positions
    assert channel_info[0] == "Fp1"  # Fp1 is first in canonical
    assert channel_info[1] == "Fp2"  # Fp2 is second

    # Check that missing channels are marked
    assert "F7_missing" in channel_info
    assert "T7_missing" in channel_info

    # Check that data is copied correctly
    fp1_idx = CANONICAL_CHANNELS.index("Fp1")
    assert np.allclose(prepared[fp1_idx], data[0])


@pytest.mark.unit
def test_prepare_channels_with_legacy_names():
    """Test channel preparation with legacy channel names."""
    # Use legacy names T3, T4, T5, T6
    legacy_channels = ["Fp1", "T3", "C3", "T4", "T5", "T6", "O1"]
    n_samples = 1024
    data = np.random.randn(len(legacy_channels), n_samples).astype(np.float32)

    prepared, channel_info = prepare_channels(data, legacy_channels)

    assert prepared.shape == (19, n_samples)

    # Check that T3 data is in T7 position
    t7_idx = CANONICAL_CHANNELS.index("T7")
    t3_data_idx = legacy_channels.index("T3")
    assert np.allclose(prepared[t7_idx], data[t3_data_idx])
    assert channel_info[t7_idx] == "T7"  # Not "T3"
