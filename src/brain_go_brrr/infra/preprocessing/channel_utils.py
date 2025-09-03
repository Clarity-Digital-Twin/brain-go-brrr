"""Channel utilities for preprocessing.

Handles channel type canonicalization and mapping.
"""

import logging
from typing import Any

import mne

logger = logging.getLogger(__name__)


def canonicalize_channel_types(raw: mne.io.Raw) -> mne.io.Raw:
    """Set channel types based on channel names.
    
    EDF format doesn't preserve channel types - everything becomes 'eeg'.
    This function sets proper types based on channel names to ensure
    correct channel selection downstream.
    
    Args:
        raw: MNE Raw object with channels to canonicalize
        
    Returns:
        The same Raw object with corrected channel types
        
    Channel type rules:
        - EOG/EYE -> eog
        - ECG/EKG -> ecg  
        - A1/A2/M1/M2/REF -> misc (reference electrodes)
        - EMG -> emg
        - Everything else -> eeg
    """
    mapping = {}
    
    for ch_name in raw.ch_names:
        ch_upper = ch_name.upper()
        
        # EOG/Eye movement channels
        if "EOG" in ch_upper or "EYE" in ch_upper:
            mapping[ch_name] = "eog"
        # ECG/Heart channels
        elif "ECG" in ch_upper or "EKG" in ch_upper:
            mapping[ch_name] = "ecg"
        # EMG/Muscle channels
        elif "EMG" in ch_upper:
            mapping[ch_name] = "emg"
        # Reference electrodes
        elif ch_upper in {"A1", "A2", "M1", "M2"} or "REF" in ch_upper:
            mapping[ch_name] = "misc"
        # Default to EEG
        else:
            mapping[ch_name] = "eeg"
    
    # Apply the mapping
    if mapping:
        # Count types for logging
        from collections import Counter
        type_counts = Counter(mapping.values())
        logger.debug(
            f"Canonicalized channel types: {dict(type_counts)} "
            f"(total: {len(mapping)} channels)"
        )
        raw.set_channel_types(mapping, verbose=False)
    
    return raw