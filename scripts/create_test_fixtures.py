"""Create synthetic EEG test fixtures for TDD.

This script creates realistic synthetic EEG test fixtures that simulate TUAB data
without requiring the full dataset download.
"""

import json
from pathlib import Path

import mne
import numpy as np

# Suppress MNE verbose output
mne.set_log_level("ERROR")


def create_synthetic_eeg(
    duration: float = 30.0,
    sfreq: int = 256,
    n_channels: int = 19,
    is_abnormal: bool = False,
    seed: int = None,
) -> mne.io.RawArray:
    """Create realistic synthetic EEG data.

    Args:
        duration: Duration in seconds
        sfreq: Sampling frequency
        n_channels: Number of channels
        is_abnormal: Whether to simulate abnormal patterns
        seed: Random seed for reproducibility

    Returns:
        MNE Raw object with simulated EEG
    """
    if seed is not None:
        np.random.seed(seed)

    # Standard 10-20 channel names
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
    ][:n_channels]

    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq

    # Initialize data array
    data = np.zeros((n_channels, n_samples))

    # Generate realistic EEG patterns
    for ch_idx in range(n_channels):
        # Base rhythms
        # Alpha (8-12 Hz) - strongest in occipital channels
        if ch_names[ch_idx] in ["O1", "O2", "P3", "P4"]:
            alpha_amp = 20e-6 if not is_abnormal else 5e-6  # Reduced alpha in abnormal
        else:
            alpha_amp = 10e-6
        data[ch_idx] += alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)

        # Beta (12-30 Hz) - frontal
        if ch_names[ch_idx] in ["Fp1", "Fp2", "F3", "F4"]:
            beta_amp = 8e-6
        else:
            beta_amp = 5e-6
        data[ch_idx] += beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)

        # Theta (4-8 Hz)
        theta_amp = 15e-6 if not is_abnormal else 25e-6  # Increased theta in abnormal
        data[ch_idx] += theta_amp * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)

        # Delta (0.5-4 Hz)
        delta_amp = 20e-6 if not is_abnormal else 40e-6  # Increased delta in abnormal
        data[ch_idx] += delta_amp * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)

        # Add abnormal patterns if specified
        if is_abnormal:
            # Simulate generalized slowing
            data[ch_idx] += 30e-6 * np.sin(2 * np.pi * 3 * t)

            # Add some spike-like transients
            n_spikes = np.random.randint(5, 15)
            spike_times = np.random.choice(n_samples, n_spikes, replace=False)
            for spike_time in spike_times:
                # Create spike waveform
                spike_duration = int(0.07 * sfreq)  # 70ms spike
                if spike_time + spike_duration < n_samples:
                    spike_wave = 50e-6 * np.exp(-np.linspace(0, 5, spike_duration))
                    data[ch_idx, spike_time : spike_time + spike_duration] += spike_wave

            # Add rhythmic discharges in temporal channels
            if ch_names[ch_idx] in ["T3", "T4", "T5", "T6"]:
                # 3 Hz rhythmic discharge
                discharge_amp = 40e-6 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
                data[ch_idx] += discharge_amp * np.sin(2 * np.pi * 3 * t)

        # Add realistic background noise
        data[ch_idx] += np.random.randn(n_samples) * 3e-6

        # Add 50 Hz powerline noise (will be filtered out)
        data[ch_idx] += 2e-6 * np.sin(2 * np.pi * 50 * t)

    # Create info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Set standard montage
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False)

    return raw


def create_fixture(
    output_path: Path,
    duration: float = 30.0,
    is_abnormal: bool = False,
    seed: int = None,
):
    """Create and save a synthetic EEG fixture.

    Args:
        output_path: Path to save fixture
        duration: Duration in seconds
        is_abnormal: Whether to create abnormal EEG
        seed: Random seed for reproducibility
    """
    # Create synthetic EEG
    raw = create_synthetic_eeg(
        duration=duration, sfreq=256, n_channels=19, is_abnormal=is_abnormal, seed=seed
    )

    # Save as FIF format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_path, overwrite=True)

    print(f"Created fixture: {output_path.name}")
    print(f"  - Type: {'abnormal' if is_abnormal else 'normal'}")
    print(f"  - Channels: {len(raw.ch_names)}")
    print(f"  - Duration: {raw.times[-1]:.1f}s")
    print(f"  - Size: {output_path.stat().st_size / 1024:.1f} KB")

    return raw


def main():
    """Create synthetic test fixtures for abnormality detection."""
    # Define fixtures to create
    fixtures = [
        {
            "output": "tests/fixtures/eeg/tuab_001_norm_30s.fif",
            "label": 0,  # normal
            "is_abnormal": False,
            "duration": 30.0,
            "seed": 42,
        },
        {
            "output": "tests/fixtures/eeg/tuab_002_norm_30s.fif",
            "label": 0,  # normal
            "is_abnormal": False,
            "duration": 30.0,
            "seed": 43,
        },
        {
            "output": "tests/fixtures/eeg/tuab_003_abnorm_30s.fif",
            "label": 1,  # abnormal
            "is_abnormal": True,
            "duration": 30.0,
            "seed": 44,
        },
        {
            "output": "tests/fixtures/eeg/tuab_004_abnorm_30s.fif",
            "label": 1,  # abnormal
            "is_abnormal": True,
            "duration": 30.0,
            "seed": 45,
        },
        # Edge cases
        {
            "output": "tests/fixtures/eeg/tuab_005_short_10s.fif",
            "label": 0,  # normal
            "is_abnormal": False,
            "duration": 10.0,
            "seed": 46,
        },
    ]

    labels = {}

    print("Creating synthetic EEG test fixtures...")
    print("=" * 50)

    for fixture in fixtures:
        output_path = Path(fixture["output"])

        # Create fixture
        raw = create_fixture(
            output_path,
            duration=fixture["duration"],
            is_abnormal=fixture["is_abnormal"],
            seed=fixture["seed"],
        )

        # Store label
        labels[output_path.name] = {
            "label": fixture["label"],
            "label_name": "abnormal" if fixture["label"] == 1 else "normal",
            "channels": len(raw.ch_names),
            "sfreq": raw.info["sfreq"],
            "duration": float(raw.times[-1]),
        }

        # Also create 5s version for faster tests
        if fixture["duration"] == 30.0:
            short_name = output_path.name.replace("30s", "5s")
            short_path = output_path.parent / short_name

            raw_short = raw.copy().crop(0, 5)
            raw_short.save(short_path, overwrite=True)

            labels[short_name] = {
                "label": fixture["label"],
                "label_name": "abnormal" if fixture["label"] == 1 else "normal",
                "channels": len(raw_short.ch_names),
                "sfreq": raw_short.info["sfreq"],
                "duration": float(raw_short.times[-1]),
            }
            print(f"  - Also created 5s version: {short_name}")

        print()

    # Save labels file
    labels_path = Path("tests/fixtures/eeg/labels.json")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nCreated labels file: {labels_path}")
    print(f"Total fixtures created: {len(labels)}")

    # Create README
    readme_content = """# EEG Test Fixtures

Synthetic EEG test fixtures for unit testing the abnormality detection system.

## Files
- `tuab_001_norm_30s.fif` - Normal EEG, 30 seconds
- `tuab_002_norm_30s.fif` - Normal EEG, 30 seconds
- `tuab_003_abnorm_30s.fif` - Abnormal EEG with slowing and spikes
- `tuab_004_abnorm_30s.fif` - Abnormal EEG with rhythmic discharges
- `tuab_005_short_10s.fif` - Short normal EEG for edge case testing

Each 30s fixture also has a 5s version for faster tests.

## Note
These are synthetic fixtures for testing only. Use real TUAB data for validation.
"""

    readme_path = labels_path.parent / "README.md"
    with readme_path.open("w") as f:
        f.write(readme_content)

    print(f"Created README: {readme_path}")


if __name__ == "__main__":
    main()
