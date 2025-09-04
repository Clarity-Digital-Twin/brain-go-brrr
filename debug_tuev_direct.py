#!/usr/bin/env python3
"""Debug TUEV preprocessing directly."""

import os
os.environ["BGB_ALLOW_SYNTH_TUEV"] = "1"

from pathlib import Path
import tempfile

# Get the sample path same way tests do
from tests.conftest import _create_synthetic_tuev

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    edf_path = _create_synthetic_tuev(tmp_path)

    print(f"Created synthetic TUEV at: {edf_path}")

    # Load raw and check what we have
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    print(f"\nRaw file has {len(raw.ch_names)} channels:")
    print(f"Channel names: {raw.ch_names}")
    print(f"Channel types: {raw.get_channel_types()}")

    # Apply canonicalization
    from brain_go_brrr.infra.preprocessing.channel_utils import canonicalize_channel_types
    raw = canonicalize_channel_types(raw)

    print(f"\nAfter canonicalization:")
    print(f"Channel types: {raw.get_channel_types()}")

    # Now pick just EEG channels
    raw_eeg = raw.copy().pick_types(eeg=True)
    print(f"\nEEG channels only: {len(raw_eeg.ch_names)}")
    print(f"EEG channel names: {raw_eeg.ch_names}")

    # Check which are standard
    from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor
    preprocessor = TUEVPreprocessor()
    standard = preprocessor.STANDARD_CHANNELS

    available = [ch for ch in standard if ch in raw_eeg.ch_names]
    missing = [ch for ch in standard if ch not in raw_eeg.ch_names]

    print(f"\nStandard channels available: {len(available)}")
    print(f"Missing standard channels: {missing}")

    # Check for Oz specifically
    for ch in raw.ch_names:
        if 'oz' in ch.lower():
            print(f"\nFound Oz variant: '{ch}'")
            print(f"  Type: {raw.get_channel_types([ch])[0]}")
