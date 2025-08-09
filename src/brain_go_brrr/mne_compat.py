"""MNE-Python compatibility layer.

This module provides a clean interface to MNE-Python functions, handling
API differences and type issues in one place rather than scattering
type: ignore comments throughout the codebase.
"""

from typing import Any, TYPE_CHECKING

import mne
import numpy as np
import numpy.typing as npt
from mne.io import BaseRaw as MNERaw

if TYPE_CHECKING:
    from typing import TypeGuard


def is_mne_raw(obj: Any) -> "TypeGuard[MNERaw]":
    """Check if object is an MNE Raw object.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is MNE Raw
    """
    return (
        hasattr(obj, "get_data") 
        and hasattr(obj, "info") 
        and hasattr(obj, "ch_names")
        and hasattr(obj, "times")
    )


def get_eeg_picks(raw: MNERaw, exclude_bads: bool = True) -> list[int]:
    """Get indices of EEG channels.

    Args:
        raw: MNE Raw object
        exclude_bads: Whether to exclude bad channels

    Returns:
        List of channel indices
    """
    exclude = "bads" if exclude_bads else []
    return list(mne.pick_types(raw.info, meg=False, eeg=True, eog=False, exclude=exclude))


def get_all_picks(raw: MNERaw, eeg: bool = True, eog: bool = False, emg: bool = False) -> list[int]:
    """Get indices of specified channel types.

    Args:
        raw: MNE Raw object
        eeg: Include EEG channels
        eog: Include EOG channels
        emg: Include EMG channels

    Returns:
        List of channel indices
    """
    return list(mne.pick_types(raw.info, meg=False, eeg=eeg, eog=eog, emg=emg, exclude="bads"))


def has_montage(raw: MNERaw) -> bool:
    """Check if raw data has channel positions.

    Args:
        raw: MNE Raw object

    Returns:
        True if positions exist
    """
    # Check for digitization points (modern way)
    if raw.info.get("dig"):
        return True
    # Fallback: check if channels have positions
    try:
        montage = raw.get_montage()
        return montage is not None
    except (AttributeError, RuntimeError):
        return False


def set_montage_safe(
    raw: MNERaw,
    montage_name: str = "standard_1020",
    match_case: bool = False,
    on_missing: str = "ignore",
) -> None:
    """Safely set montage on raw data.

    Args:
        raw: MNE Raw object
        montage_name: Name of standard montage
        match_case: Whether to match channel names case-sensitively
        on_missing: What to do with channels not in montage
    """
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=match_case, on_missing=on_missing)
    except Exception:
        # Silently ignore if montage can't be set
        pass


def get_channel_types(raw: MNERaw) -> list[str]:
    """Get channel types for all channels.

    Args:
        raw: MNE Raw object

    Returns:
        List of channel types
    """
    types = raw.get_channel_types()
    return list(types)


def get_channel_type(raw: MNERaw, channel: str | int) -> str:
    """Get type of a specific channel.

    Args:
        raw: MNE Raw object
        channel: Channel name or index

    Returns:
        Channel type string
    """
    if isinstance(channel, str):
        if channel not in raw.ch_names:
            raise ValueError(f"Channel {channel} not found")
        idx = raw.ch_names.index(channel)
    else:
        idx = channel

    types = raw.get_channel_types()
    return str(types[idx])


def update_data_inplace(raw: MNERaw, data: npt.NDArray[np.float64]) -> MNERaw:
    """Update raw data array safely using public API.

    This replaces direct writes to raw._data which is private API.

    Args:
        raw: MNE Raw object to update
        data: New data array (n_channels, n_samples)

    Returns:
        New Raw object with updated data
    """
    # Create new raw with updated data
    info = raw.info.copy()

    # Get first_samp value if it exists (don't try to set it)
    first_samp = getattr(raw, 'first_samp', 0)

    # Create new RawArray with proper first_samp
    raw_new = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)

    # Preserve annotations if present
    if hasattr(raw, "annotations") and raw.annotations is not None:
        raw_new.set_annotations(raw.annotations.copy())

    return raw_new


def has_annotations(raw: MNERaw) -> bool:
    """Check if raw has annotations.

    Args:
        raw: MNE Raw object

    Returns:
        True if annotations exist
    """
    return hasattr(raw, "annotations") and raw.annotations is not None


def copy_annotations(source: MNERaw, target: MNERaw) -> None:
    """Copy annotations from source to target.

    Args:
        source: Source Raw object
        target: Target Raw object
    """
    if has_annotations(source):
        target.set_annotations(source.annotations.copy())


def info_has_field(info: Any, field: str) -> bool:
    """Check if info dict has a field.

    Args:
        info: MNE Info object
        field: Field name to check

    Returns:
        True if field exists
    """
    try:
        return field in info
    except (TypeError, KeyError):
        return False


def filter_raw(
    raw: MNERaw,
    l_freq: float | None = None,
    h_freq: float | None = None,
    picks: str | list[int] | None = None,
    method: str = "fir",
    **kwargs: Any,
) -> None:
    """Apply frequency filter with proper picks handling.

    Args:
        raw: MNE Raw object
        l_freq: Low frequency cutoff
        h_freq: High frequency cutoff
        picks: Channels to filter ('eeg', 'all', or list of indices)
        method: Filter method
        **kwargs: Additional filter arguments
    """
    if picks == "eeg":
        picks_idx = get_eeg_picks(raw)
    elif picks == "all":
        picks_idx = None
    elif isinstance(picks, list):
        picks_idx = picks
    else:
        picks_idx = None

    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks_idx, method=method, **kwargs)


def notch_filter_raw(
    raw: MNERaw,
    freqs: float | list[float],
    picks: str | list[int] | None = None,
    **kwargs: Any,
) -> None:
    """Apply notch filter with proper picks handling.

    Args:
        raw: MNE Raw object
        freqs: Frequencies to notch
        picks: Channels to filter ('eeg', 'all', or list of indices)
        **kwargs: Additional filter arguments
    """
    if picks == "eeg":
        picks_idx = get_eeg_picks(raw)
    elif picks == "all":
        picks_idx = None
    elif isinstance(picks, list):
        picks_idx = picks
    else:
        picks_idx = None

    raw.notch_filter(freqs=freqs, picks=picks_idx, **kwargs)


def pick_channels(raw: MNERaw, picks: str | list[str] | list[int]) -> None:
    """Pick channels from raw data.

    Args:
        raw: MNE Raw object
        picks: Channel selection ('eeg', channel names, or indices)
    """
    if picks == "eeg":
        picks_idx = get_eeg_picks(raw)
        raw.pick(picks_idx)
    elif isinstance(picks, list):
        if picks and isinstance(picks[0], str):
            raw.pick_channels(picks)
        else:
            raw.pick(picks)
