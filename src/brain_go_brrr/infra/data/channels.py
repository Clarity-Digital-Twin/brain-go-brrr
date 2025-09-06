"""Channel definitions and mapping - Single Source of Truth.

CRITICAL: These are the ONLY valid channel configurations.
- TUAB: 19 channels (no Fz)
- TUEV: 20 channels (with Fz and Fpz, no Oz)
"""

# TUAB standard: 19 channels (NO Fz) per EEGPT paper
# Using standard 10-20 mixed case naming (Fp not FP, Cz not CZ, etc.)
CHANNELS_TUAB_19 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "F4",
    "F8",  # Frontal (no Fz!)
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",  # Central/Temporal
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",  # Parietal
    "O1",
    "Oz",
    "O2",  # Occipital
]

# TUEV standard: 20 channels (WITH Fz and Fpz, NO Oz) per EEGPT Table 13
# Using standard 10-20 mixed case naming (Fp not FP, Cz not CZ, etc.)
CHANNELS_TUEV_20 = [
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",  # Frontal (with Fz and Fpz)
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",  # Central/Temporal
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",  # Parietal
    "O1",
    "O2",  # Occipital (no Oz)
]

# Full 10-20 system for reference
CHANNELS_10_20_FULL = [
    "FP1",
    "FPZ",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "OZ",
    "O2",
    "A1",
    "A2",
]

# Modern naming (T3→T7, T4→T8, T5→P7, T6→P8) and case conversions
CHANNEL_ALIASES = {
    # Legacy to modern
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    # EEG prefix variations
    "EEG T3-REF": "T7",
    "EEG T4-REF": "T8",
    "EEG T5-REF": "P7",
    "EEG T6-REF": "P8",
    # Case conversions (uppercase to mixed-case)
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "FPZ": "Fpz",
}


def validate_channels(channels: list[str], expected: list[str], dataset_name: str) -> None:
    """Validate channels match expected configuration exactly.

    Args:
        channels: Actual channel names
        expected: Expected channel configuration
        dataset_name: Name for error messages

    Raises:
        ValueError: If channels don't match exactly
    """
    if len(channels) != len(expected):
        raise ValueError(
            f"{dataset_name} requires exactly {len(expected)} channels, got {len(channels)}"
        )

    # Apply aliasing
    normalized = [CHANNEL_ALIASES.get(ch, ch) for ch in channels]

    missing = set(expected) - set(normalized)
    extra = set(normalized) - set(expected)

    if missing or extra:
        msg = f"{dataset_name} channel mismatch."
        if missing:
            msg += f" Missing: {sorted(missing)}."
        if extra:
            msg += f" Extra: {sorted(extra)}."
        raise ValueError(msg)


def map_channels_to_indices(
    source_channels: list[str], target_channels: list[str]
) -> dict[int, int]:
    """Map source channel indices to target channel indices.

    Args:
        source_channels: Channel names in source data
        target_channels: Target channel configuration

    Returns:
        Dict mapping source index -> target index

    Raises:
        ValueError: If required channels are missing
    """
    # Apply aliasing
    normalized_source = [CHANNEL_ALIASES.get(ch, ch) for ch in source_channels]

    mapping = {}
    for target_idx, target_ch in enumerate(target_channels):
        if target_ch in normalized_source:
            source_idx = normalized_source.index(target_ch)
            mapping[source_idx] = target_idx
        else:
            raise ValueError(f"Required channel {target_ch} not found in source")

    return mapping


# Export key constants
__all__ = [
    "CHANNELS_10_20_FULL",
    "CHANNELS_TUAB_19",
    "CHANNELS_TUEV_20",
    "CHANNEL_ALIASES",
    "map_channels_to_indices",
    "validate_channels",
]
