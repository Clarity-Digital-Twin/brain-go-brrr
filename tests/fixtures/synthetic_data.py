"""Synthetic data fixtures for testing without real datasets."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_sleep_raw(mne_mod):
    """Create synthetic 5-minute EEG for sleep staging tests.

    This replaces real Sleep-EDF data for fast unit tests.
    """
    import mne

    # Standard sleep montage channels
    ch_names = [
        'Fpz-Cz', 'Pz-Oz',  # Sleep-EDF style
        'EOG horizontal',
        'Resp oro-nasal',
        'EMG submental',
        'Temp rectal',
        'Event marker'
    ]

    sfreq = 100  # Standard for sleep
    duration = 300  # 5 minutes
    n_times = sfreq * duration

    # Create realistic-looking data
    np.random.seed(42)
    data = []

    # EEG channels with alpha/theta rhythms
    for _i in range(2):
        # Mix of frequencies typical in sleep
        t = np.arange(n_times) / sfreq
        signal = (
            10e-6 * np.sin(2 * np.pi * 10 * t) +  # Alpha
            15e-6 * np.sin(2 * np.pi * 4 * t) +   # Theta
            5e-6 * np.random.randn(n_times)       # Noise
        )
        data.append(signal)

    # EOG channel (eye movements)
    data.append(50e-6 * np.random.randn(n_times))

    # Respiration (slow oscillation)
    t = np.arange(n_times) / sfreq
    data.append(100e-6 * np.sin(2 * np.pi * 0.25 * t))

    # EMG (muscle activity)
    data.append(20e-6 * np.random.randn(n_times))

    # Temperature (very slow drift)
    data.append(np.linspace(0, 10e-6, n_times))

    # Event marker (zeros)
    data.append(np.zeros(n_times))

    data = np.array(data)

    # Create channel types
    ch_types = ['eeg', 'eeg', 'eog', 'resp', 'emg', 'misc', 'stim']

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    raw = mne.io.RawArray(data, info)

    # Add annotations for sleep stages (mock)
    onset = np.arange(0, duration, 30)  # Every 30 seconds
    duration_annot = [30] * len(onset)
    stages = ['W', 'N1', 'N2', 'N3', 'N2', 'REM'] * (len(onset) // 6 + 1)
    stages = stages[:len(onset)]

    annotations = mne.Annotations(
        onset=onset,
        duration=duration_annot,
        description=stages
    )
    raw.set_annotations(annotations)

    return raw


@pytest.fixture
def synthetic_tuab_raw(mne_mod):
    """Create synthetic TUAB-style EEG data.

    20 channels, 10-20 montage, for abnormality detection.
    """
    import mne

    # Standard TUAB channels (20 channels)
    ch_names = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'FZ', 'CZ', 'PZ', 'OZ'
    ]

    sfreq = 256  # TUAB sampling rate
    duration = 60  # 1 minute for tests
    n_times = sfreq * duration

    # Create data with some "abnormal" patterns
    np.random.seed(42)
    data = []

    for _i, ch in enumerate(ch_names):
        # Base EEG signal
        signal = 20e-6 * np.random.randn(n_times)

        # Add spikes to some channels (simulate abnormality)
        if ch in ['T3', 'T4']:  # Temporal channels
            spike_times = np.random.choice(n_times, 10, replace=False)
            signal[spike_times] += 100e-6  # Add spikes

        # Add slow waves to others
        if ch in ['F3', 'F4']:
            t = np.arange(n_times) / sfreq
            signal += 30e-6 * np.sin(2 * np.pi * 2 * t)  # Delta waves

        data.append(signal)

    data = np.array(data)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )

    raw = mne.io.RawArray(data, info)

    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    return raw


@pytest.fixture
def tuab_mini_dataset(tmp_path):
    """Create minimal TUAB dataset structure for testing.

    Creates:
    - 2 normal files
    - 2 abnormal files
    - Proper directory structure
    """
    import mne

    # Create directory structure
    tuab_dir = tmp_path / "tuab_mini"
    normal_dir = tuab_dir / "normal"
    abnormal_dir = tuab_dir / "abnormal"

    normal_dir.mkdir(parents=True)
    abnormal_dir.mkdir(parents=True)

    # Standard TUAB channels
    ch_names = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'FZ', 'CZ', 'PZ', 'OZ'
    ]

    sfreq = 256
    duration = 10  # Very short for speed

    # Create normal files
    for i in range(2):
        data = 20e-6 * np.random.randn(20, sfreq * duration)
        info = mne.create_info(ch_names, sfreq, 'eeg')
        raw = mne.io.RawArray(data, info)

        fname = normal_dir / f"normal_{i:03d}.edf"
        raw.export(str(fname), fmt='edf', overwrite=True)

    # Create abnormal files (with spikes)
    for i in range(2):
        data = 20e-6 * np.random.randn(20, sfreq * duration)
        # Add abnormal patterns
        spike_times = np.random.choice(sfreq * duration, 20, replace=False)
        data[5:8, spike_times] += 100e-6  # Spikes in some channels

        info = mne.create_info(ch_names, sfreq, 'eeg')
        raw = mne.io.RawArray(data, info)

        fname = abnormal_dir / f"abnormal_{i:03d}.edf"
        raw.export(str(fname), fmt='edf', overwrite=True)

    return tuab_dir


@pytest.fixture
def mock_eegpt_features():
    """Create mock EEGPT feature output."""
    # EEGPT outputs: [batch, n_summary_tokens, embed_dim]
    # Typical: [B, 4, 512]
    batch_size = 8
    n_summary_tokens = 4
    embed_dim = 512

    features = np.random.randn(batch_size, n_summary_tokens, embed_dim).astype(np.float32)
    return features
