"""Single preparation function for EEGPT input data.

This module provides a centralized preprocessing pipeline for EEGPT,
ensuring consistent data preparation across all code paths.
"""

import logging
from typing import Any

import numpy as np
import torch

from .nan_policy import validate_no_nan

logger = logging.getLogger(__name__)

# Standard EEGPT channels (modern naming)
EEGPT_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
]

# Old to new channel mapping for TUAB compatibility
CHANNEL_MAP = {
    'T3': 'T7',
    'T4': 'T8', 
    'T5': 'P7',
    'T6': 'P8'
}


def prepare_for_eegpt(
    raw: Any,  # MNE Raw object
    target_sfreq: int = 256,
    required_channels: int = 19,
    pad_to_multiple: int = 64
) -> np.ndarray:
    """Prepare raw EEG data for EEGPT model input.
    
    Single entry point for all EEGPT preprocessing:
    1. Resample to target sampling rate
    2. Map old channel names to modern naming
    3. Validate channel count
    4. Pad temporal dimension to multiple of patch size
    5. Validate no NaN/Inf values
    
    Args:
        raw: MNE Raw object with EEG data
        target_sfreq: Target sampling rate (default 256 Hz)
        required_channels: Minimum required channels (default 19)
        pad_to_multiple: Pad temporal dimension to multiple of this (default 64)
        
    Returns:
        Preprocessed numpy array of shape (channels, samples)
        
    Raises:
        ValueError: If data contains NaN/Inf or insufficient channels
    """
    # Step 1: Resample if needed
    current_sfreq = int(raw.info['sfreq'])
    if current_sfreq != target_sfreq:
        logger.info(f"Resampling from {current_sfreq} Hz to {target_sfreq} Hz")
        raw = raw.copy().resample(target_sfreq, npad='auto')
    
    # Step 2: Map old channel names to modern naming
    current_channels = raw.ch_names
    renamed_channels = []
    for ch in current_channels:
        ch_upper = ch.upper()
        # Apply mapping if needed
        mapped = CHANNEL_MAP.get(ch_upper, ch_upper)
        renamed_channels.append(mapped)
    
    # Update channel names if any were mapped
    if renamed_channels != current_channels:
        raw.rename_channels(dict(zip(current_channels, renamed_channels)))
        logger.info(f"Mapped channel names: {current_channels[:3]} -> {renamed_channels[:3]}...")
    
    # Step 3: Validate channel count
    n_channels = len(raw.ch_names)
    if n_channels < required_channels:
        raise ValueError(
            f"Insufficient channels: got {n_channels}, need at least {required_channels}"
        )
    
    # Step 4: Get data and pad temporal dimension
    data = raw.get_data()
    n_samples = data.shape[1]
    
    if pad_to_multiple > 0:
        remainder = n_samples % pad_to_multiple
        if remainder != 0:
            pad_amount = pad_to_multiple - remainder
            logger.debug(f"Padding {n_samples} samples by {pad_amount} to reach multiple of {pad_to_multiple}")
            # Pad with edge values (repeat last samples)
            data = np.pad(data, ((0, 0), (0, pad_amount)), mode='edge')
    
    # Step 5: Validate no NaN/Inf
    validate_no_nan(data, "EEGPT input after preprocessing")
    
    # Ensure float32 for consistency
    data = data.astype(np.float32)
    
    logger.debug(f"Prepared EEGPT input: shape {data.shape}, dtype {data.dtype}")
    
    return data


def supports_eegpt(raw: Any) -> bool:
    """Check if raw data supports EEGPT analysis.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        True if data meets EEGPT requirements
    """
    # Check channel count
    n_channels = len(raw.ch_names)
    if n_channels < 19:
        logger.debug(f"EEGPT requires 19+ channels, got {n_channels}")
        return False
    
    # Check sampling rate is reasonable (can be resampled)
    sfreq = raw.info['sfreq']
    if sfreq < 100 or sfreq > 5000:
        logger.debug(f"Sampling rate {sfreq} Hz out of reasonable range")
        return False
    
    # Check duration (need at least 1 second)
    duration = len(raw.times) / sfreq
    if duration < 1.0:
        logger.debug(f"Duration {duration:.2f}s too short for EEGPT")
        return False
    
    return True