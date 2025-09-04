#!/usr/bin/env python3
"""Debug TUEV preprocessing step by step."""

import os
os.environ["BGB_ALLOW_SYNTH_TUEV"] = "1"

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from pathlib import Path
import tempfile

# Import the fixture creator and preprocessor
import sys
sys.path.insert(0, "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr")
from tests.conftest import _create_synthetic_tuev
from brain_go_brrr.infra.preprocessing.tuev_preprocessor import TUEVPreprocessor

# Create synthetic TUEV
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    edf_path = _create_synthetic_tuev(tmp_path)

    print(f"\n=== Created synthetic TUEV at: {edf_path} ===\n")

    # Try to preprocess it
    try:
        preprocessor = TUEVPreprocessor()
        epochs, info = preprocessor.process_raw(edf_path)

        print(f"\n=== SUCCESS! ===")
        print(f"Epochs shape: {epochs.get_data().shape}")
        print(f"Channels ({len(epochs.ch_names)}): {epochs.ch_names}")

    except Exception as e:
        print(f"\n=== FAILED with error: {e} ===")

        # Debug: check raw file directly
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        print(f"\nRaw channels before preprocessing ({len(raw.ch_names)}): {raw.ch_names}")

        # Check which are EEG
        eeg_channels = [ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == 'eeg']
        print(f"\nEEG channels ({len(eeg_channels)}): {eeg_channels}")

        # Check for Oz specifically
        has_oz = any('oz' in ch.lower() for ch in raw.ch_names)
        print(f"\nHas Oz (case-insensitive): {has_oz}")
        oz_variants = [ch for ch in raw.ch_names if 'oz' in ch.lower()]
        print(f"Oz variants found: {oz_variants}")
