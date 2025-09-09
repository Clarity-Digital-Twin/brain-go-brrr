"""Channel utilities for preprocessing.

Handles channel type canonicalization and mapping.
"""

import logging

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
            
        # Reference electrodes (true reference leads only)
        elif ch_upper in {"A1", "A2", "M1", "M2"}:
            mapping[ch_name] = "misc"

        # Ocular leads sometimes labeled LOC/ROC in TUEV
        elif ch_upper.startswith("LOC") or ch_upper.startswith("ROC"):
            mapping[ch_name] = "eog"
        # Default to EEG
        else:
            mapping[ch_name] = "eeg"

    # Apply the mapping
    if mapping:
        # Count types for logging
        from collections import Counter

        type_counts = Counter(mapping.values())
        logger.debug(
            f"Canonicalized channel types: {dict(type_counts)} (total: {len(mapping)} channels)"
        )
        raw.set_channel_types(mapping, verbose=False)

    return raw


def canonicalize_channel_labels(raw: mne.io.Raw) -> mne.io.Raw:
    """Canonicalize channel labels to standard 10-20 naming.

    Handles:
    - Stripping EDF prefixes/suffixes ("EEG ", "-REF")
    - Legacy to modern mapping (T3→T7, T4→T8, T5→P7, T6→P8)
    - Case normalization to mixed-case 10-20 standard (Fp1 not FP1, Cz not CZ)

    Args:
        raw: MNE Raw object with channels to canonicalize

    Returns:
        The same Raw object with standardized channel names

    Channel naming rules:
        - Strip "EEG " prefix and "-REF" suffix
        - Map T3→T7, T4→T8, T5→P7, T6→P8
        - Use mixed-case: Fp1/Fp2 (not FP1/FP2), Cz/Pz/Fz/Oz (not CZ/PZ/FZ/OZ)
    """
    import re

    rename_dict = {}

    for ch_name in raw.ch_names:
        # Strip common EDF prefixes/suffixes
        clean = ch_name
        clean = re.sub(r'^EEG\s+', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'-REF$', '', clean, flags=re.IGNORECASE)
        clean = clean.strip()

        # Convert to uppercase for comparison
        upper = clean.upper()

        # Apply legacy to modern mapping
        legacy_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        modern = legacy_map.get(upper, upper)

        # Apply mixed-case standard (Fp not FP, lowercase 'z' in Cz/Pz/Fz/Oz)
        case_map = {
            'FP1': 'Fp1',
            'FP2': 'Fp2',
            'FPZ': 'Fpz',
            'CZ': 'Cz',
            'PZ': 'Pz',
            'FZ': 'Fz',
            'OZ': 'Oz',
        }

        if modern in case_map:
            final = case_map[modern]
        else:
            # For standard channels, use title case (F7, C3, etc.)
            # but preserve already correct mixed-case
            if modern in ['Fp1', 'Fp2', 'Fpz', 'Cz', 'Pz', 'Fz', 'Oz']:
                final = modern
            else:
                # Standard channels like F7, C3, O1, etc.
                final = modern.title()

        # Only add to rename dict if different
        if ch_name != final:
            rename_dict[ch_name] = final

    # Apply renaming if needed
    if rename_dict:
        logger.debug(f"Renaming {len(rename_dict)} channels to standard labels")
        # Log a sample of renames for debugging
        sample = list(rename_dict.items())[:5]
        for old, new in sample:
            logger.debug(f"  {old} -> {new}")
        if len(rename_dict) > 5:
            logger.debug(f"  ... and {len(rename_dict) - 5} more")

        raw.rename_channels(rename_dict)

    return raw
