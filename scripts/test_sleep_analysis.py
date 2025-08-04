#!/usr/bin/env python
"""Test sleep analysis pipeline with Sleep-EDF data.

This script handles the lack of channel positions in Sleep-EDF files.
"""

import sys
from pathlib import Path

import mne
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_go_brrr  # noqa: E402.core.logger import get_logger
from brain_go_brrr  # noqa: E402.core.sleep import SleepAnalyzer

logger = get_logger(__name__)


def add_standard_montage(raw):
    """Add standard montage to raw data if channels match standard names."""
    # Map Sleep-EDF channels to standard 10-20 names
    channel_mapping = {
        "EEG Fpz-Cz": "Fpz",
        "EEG Pz-Oz": "Pz",
        "EOG horizontal": "EOG",
        "Resp oro-nasal": "RESP",
        "EMG submental": "EMG",
        "Temp rectal": "TEMP",
        "Event marker": "MISC",
    }

    # Rename channels to standard names
    raw.rename_channels(channel_mapping)

    # Pick only EEG channels for analysis
    eeg_channels = [ch for ch in raw.ch_names if ch in ["Fpz", "Pz"]]
    if eeg_channels:
        raw.pick_channels(eeg_channels)

        # Try to set a basic montage
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")
            logger.info("✓ Standard montage applied")
        except Exception as e:
            logger.warning(f"Could not set montage: {e}")

    return raw


def test_sleep_analysis():
    """Test sleep analysis on Sleep-EDF data."""
    print("\n" + "=" * 60)
    print("SLEEP ANALYSIS TEST WITH SLEEP-EDF DATA")
    print("=" * 60)

    # Find Sleep-EDF files
    sleep_edf_dir = Path("data/datasets/external/sleep-edf/sleep-cassette")
    edf_files = list(sleep_edf_dir.glob("*-PSG.edf"))[:2]  # Test first 2 files

    if not edf_files:
        logger.error("No Sleep-EDF files found!")
        return

    analyzer = SleepAnalyzer()

    for edf_file in edf_files:
        print(f"\n--- Testing: {edf_file.name} ---")

        try:
            # Load data
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            logger.info(
                f"Loaded: {len(raw.ch_names)} channels, {raw.n_times / raw.info['sfreq'] / 60:.1f} minutes"
            )

            # Add montage and pick EEG channels
            raw = add_standard_montage(raw)

            # Basic preprocessing for sleep analysis
            # Sleep analysis typically uses 0.3-35 Hz bandpass
            raw.filter(0.3, 35, fir_design="firwin", verbose=False)

            # Run sleep analysis
            logger.info("Running YASA sleep staging...")
            sleep_results = analyzer.run_full_sleep_analysis(raw)

            # Display results
            print("\n✓ Sleep Analysis Results:")
            print(f"  - Total recording time: {sleep_results['total_recording_time']:.1f} min")
            print(f"  - Total sleep time: {sleep_results['total_sleep_time']:.1f} min")
            print(f"  - Sleep efficiency: {sleep_results['sleep_efficiency']:.1f}%")
            print(f"  - Sleep latency: {sleep_results['sleep_latency']:.1f} min")
            print(f"  - REM latency: {sleep_results['rem_latency']:.1f} min")
            print("\n  Sleep Stage Distribution:")
            print(f"  - Wake: {sleep_results['wake_percentage']:.1f}%")
            print(f"  - N1: {sleep_results['n1_percentage']:.1f}%")
            print(f"  - N2: {sleep_results['n2_percentage']:.1f}%")
            print(f"  - N3: {sleep_results['n3_percentage']:.1f}%")
            print(f"  - REM: {sleep_results['rem_percentage']:.1f}%")

            # Check hypnogram
            hypnogram = sleep_results["hypnogram"]
            print("\n  Hypnogram stats:")
            print(f"  - Number of epochs: {len(hypnogram)}")
            print(f"  - Unique stages: {set(hypnogram)}")

            # Visualize first 100 epochs
            print("\n  First 100 epochs visualization:")
            stage_symbols = {"W": "▁", "N1": "▃", "N2": "▅", "N3": "▇", "REM": "▆"}
            viz = "".join(stage_symbols.get(stage, "?") for stage in hypnogram[:100])
            print(f"  {viz}")

        except Exception as e:
            logger.error(f"✗ Sleep analysis failed: {e}")
            import traceback

            traceback.print_exc()


def test_yasa_direct():
    """Test YASA directly without our wrapper."""
    print("\n" + "=" * 60)
    print("DIRECT YASA TEST")
    print("=" * 60)

    try:
        import yasa

        print(f"✓ YASA version: {yasa.__version__}")

        # Test with synthetic data
        print("\nTesting with synthetic EEG data...")
        sfreq = 100
        duration = 300  # 5 minutes
        times = np.arange(0, duration, 1 / sfreq)

        # Create synthetic EEG with sleep-like patterns
        # Mix of different frequency components
        eeg = (
            50 * np.sin(2 * np.pi * 1.5 * times)  # Delta (deep sleep)
            + 30 * np.sin(2 * np.pi * 10 * times)  # Alpha (relaxed)
            + 20 * np.sin(2 * np.pi * 25 * times)  # Beta (awake)
            + 10 * np.random.randn(len(times))  # Noise
        ) * 1e-6  # Convert to volts

        # Create MNE raw object
        info = mne.create_info(["Cz"], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(eeg[np.newaxis, :], info)

        # Run YASA sleep staging
        sls = yasa.SleepStaging(raw, eeg_name="Cz")
        hypno = sls.predict()

        print("✓ YASA prediction successful!")
        print(f"  - Predicted stages: {set(hypno)}")
        print(f"  - Number of epochs: {len(hypno)}")

    except Exception as e:
        logger.error(f"✗ Direct YASA test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Test YASA directly first
    test_yasa_direct()

    # Then test our sleep analysis pipeline
    test_sleep_analysis()

    print("\n✅ Sleep analysis testing complete!")
