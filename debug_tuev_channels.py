#!/usr/bin/env python3
"""Debug TUEV synthetic data channels."""

import os
os.environ["BGB_ALLOW_SYNTH_TUEV"] = "1"

from pathlib import Path
import tempfile
import mne

# Import the fixture creator
import sys
sys.path.insert(0, "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr")
from tests.conftest import _create_synthetic_tuev

# Create synthetic TUEV
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    edf_path = _create_synthetic_tuev(tmp_path)
    
    print(f"Created synthetic TUEV at: {edf_path}")
    
    # Read and check channels
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    
    print(f"\nTotal channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    
    # Check for Oz specifically
    ch_upper = [ch.upper() for ch in raw.ch_names]
    has_oz = "OZ" in ch_upper
    
    print(f"\nHas Oz channel: {has_oz}")
    
    # Count EEG channels
    eeg_channels = [ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == 'eeg']
    print(f"EEG channels ({len(eeg_channels)}): {eeg_channels}")
    
    # Check for required TUEV channels
    from brain_go_brrr.infra.data.channels import CHANNELS_TUEV_20
    print(f"\nExpected TUEV channels ({len(CHANNELS_TUEV_20)}): {CHANNELS_TUEV_20}")
    
    # Find missing
    eeg_upper = [ch.upper() for ch in eeg_channels]
    missing = [ch for ch in CHANNELS_TUEV_20 if ch not in eeg_upper]
    print(f"\nMissing channels: {missing}")