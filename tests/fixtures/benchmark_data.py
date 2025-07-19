"""Benchmark data fixtures for performance testing.

This module provides realistic EEG data samples for benchmarking:
- Single 4-second windows
- Batch processing data
- Full recording samples
"""

from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture(scope="session")
def benchmark_edf_path() -> Path | None:
    """Get path to a benchmark EDF file from Sleep-EDF dataset."""
    # Use a smaller PSG file for benchmarks
    base_path = Path(__file__).parent.parent.parent
    edf_path = base_path / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"

    if not edf_path.exists():
        # Return None if data not available - tests will use synthetic data
        return None
    return edf_path


@pytest.fixture(scope="session")
def benchmark_raw_data(benchmark_edf_path) -> mne.io.Raw | None:
    """Load benchmark EEG data if available."""
    if benchmark_edf_path is None:
        return None

    # Load and preprocess to standard format
    raw = mne.io.read_raw_edf(benchmark_edf_path, preload=True)

    # Standardize to common sampling rate
    if raw.info["sfreq"] != 256:
        raw.resample(256)

    # Apply standard preprocessing
    raw.filter(0.5, 50, fir_design="firwin")

    return raw


@pytest.fixture
def realistic_single_window(
    benchmark_raw_data,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Get a realistic 4-second EEG window from actual data."""
    if benchmark_raw_data is not None:
        # Extract 4-second window from real data
        start_time = 60.0  # Start at 1 minute to avoid initial artifacts
        raw_segment = benchmark_raw_data.copy().crop(tmin=start_time, tmax=start_time + 4.0)

        data = raw_segment.get_data()
        ch_names = raw_segment.ch_names

        # Convert to float32 and ensure correct shape
        data = data.astype(np.float32)

        # If we have more than 19 channels, select subset
        if len(ch_names) > 19:
            # Select common 10-20 channels
            standard_channels = [
                "Fp1",
                "Fp2",
                "F3",
                "F4",
                "C3",
                "C4",
                "P3",
                "P4",
                "O1",
                "O2",
                "F7",
                "F8",
                "T3",
                "T4",
                "T5",
                "T6",
                "Fz",
                "Cz",
                "Pz",
            ]
            picks = []
            selected_names = []
            for ch in standard_channels:
                if ch in ch_names:
                    picks.append(ch_names.index(ch))
                    selected_names.append(ch)

            if len(picks) >= 19:
                data = data[picks[:19]]
                ch_names = selected_names[:19]
            else:
                # Pad with zeros if needed
                n_pad = 19 - len(picks)
                data = np.vstack([data[picks], np.zeros((n_pad, data.shape[1]))])
                ch_names = selected_names + [f"PAD{i}" for i in range(n_pad)]

        return data[:19, :1024], ch_names[:19]

    else:
        # Fall back to synthetic data
        np.random.seed(42)
        data = np.random.randn(19, 1024).astype(np.float32) * 20e-6

        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
        ]

        return data, ch_names


@pytest.fixture
def realistic_batch_windows(
    benchmark_raw_data,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Get a batch of realistic 4-second EEG windows."""
    batch_size = 32

    if benchmark_raw_data is not None:
        # Extract multiple windows from real data
        windows = []
        ch_names = None

        for i in range(batch_size):
            # Space windows throughout recording
            start_time = 60.0 + i * 10.0  # 10-second spacing

            if start_time + 4.0 > benchmark_raw_data.times[-1]:
                # Not enough data, use synthetic
                break

            raw_segment = benchmark_raw_data.copy().crop(tmin=start_time, tmax=start_time + 4.0)

            data = raw_segment.get_data()

            if ch_names is None:
                ch_names = raw_segment.ch_names
                # Handle channel selection like in single window
                if len(ch_names) > 19:
                    standard_channels = [
                        "Fp1",
                        "Fp2",
                        "F3",
                        "F4",
                        "C3",
                        "C4",
                        "P3",
                        "P4",
                        "O1",
                        "O2",
                        "F7",
                        "F8",
                        "T3",
                        "T4",
                        "T5",
                        "T6",
                        "Fz",
                        "Cz",
                        "Pz",
                    ]
                    picks = []
                    selected_names = []
                    for ch in standard_channels:
                        if ch in ch_names:
                            picks.append(ch_names.index(ch))
                            selected_names.append(ch)

                    if len(picks) >= 19:
                        ch_names = selected_names[:19]
                        picks = picks[:19]
                    else:
                        # Use all available channels
                        picks = list(range(min(19, len(ch_names))))
                        ch_names = ch_names[:19]
                else:
                    picks = list(range(len(ch_names)))

            # Apply channel selection
            window_data = data[picks] if picks else data

            # Ensure correct shape
            if window_data.shape[0] < 19:
                # Pad with zeros
                n_pad = 19 - window_data.shape[0]
                window_data = np.vstack([window_data, np.zeros((n_pad, window_data.shape[1]))])

            windows.append(window_data[:19, :1024])

        if len(windows) == batch_size:
            return np.stack(windows).astype(np.float32), ch_names[:19]

    # Fall back to synthetic data
    np.random.seed(42)
    data = np.random.randn(batch_size, 19, 1024).astype(np.float32) * 20e-6

    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ]

    return data, ch_names


@pytest.fixture
def realistic_twenty_min_recording(
    benchmark_raw_data,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Get a realistic 20-minute EEG recording."""
    target_duration = 20 * 60  # 20 minutes in seconds
    target_samples = target_duration * 256  # at 256 Hz

    if benchmark_raw_data is not None:
        # Use actual recording data
        available_duration = benchmark_raw_data.times[-1]

        if available_duration >= target_duration:
            # We have enough data
            raw_segment = benchmark_raw_data.copy().crop(tmin=0, tmax=target_duration)
        else:
            # Use all available data
            raw_segment = benchmark_raw_data.copy()

        data = raw_segment.get_data()
        ch_names = raw_segment.ch_names

        # Handle channel selection
        if len(ch_names) > 19:
            standard_channels = [
                "Fp1",
                "Fp2",
                "F3",
                "F4",
                "C3",
                "C4",
                "P3",
                "P4",
                "O1",
                "O2",
                "F7",
                "F8",
                "T3",
                "T4",
                "T5",
                "T6",
                "Fz",
                "Cz",
                "Pz",
            ]
            picks = []
            selected_names = []
            for ch in standard_channels:
                if ch in ch_names:
                    picks.append(ch_names.index(ch))
                    selected_names.append(ch)

            if len(picks) >= 19:
                data = data[picks[:19]]
                ch_names = selected_names[:19]
            else:
                # Pad with zeros
                n_pad = 19 - len(picks)
                data = np.vstack([data[picks], np.zeros((n_pad, data.shape[1]))])
                ch_names = selected_names + [f"PAD{i}" for i in range(n_pad)]

        # Ensure correct length
        if data.shape[1] < target_samples:
            # Pad with synthetic data if needed
            n_pad_samples = target_samples - data.shape[1]
            synthetic_pad = np.random.randn(19, n_pad_samples) * 20e-6
            data = np.hstack([data[:19], synthetic_pad])
        else:
            data = data[:19, :target_samples]

        return data.astype(np.float32), ch_names[:19]

    else:
        # Fall back to synthetic data
        np.random.seed(42)
        data = np.random.randn(19, target_samples).astype(np.float32) * 20e-6

        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
        ]

        return data, ch_names
