# Channel Positions Solution for Sleep-EDF Data

## Problem Summary

The Sleep-EDF dataset doesn't include 3D channel positions, which causes AutoReject to fail with:

```
Valid channel positions are needed for autoreject to work
```

## Root Cause

1. Sleep-EDF files use non-standard channel names like "EEG Fpz-Cz"
2. No montage/position information is included in the files
3. AutoReject requires valid 3D positions for its data augmentation feature

## Solution Strategy

### Option 1: Add Montage to Sleep-EDF Data (Recommended)

Before preprocessing, add a standard montage to Sleep-EDF data:

```python
def add_sleep_edf_montage(raw):
    """Add standard montage to Sleep-EDF data."""
    # Map Sleep-EDF channels to standard names
    channel_mapping = {
        "EEG Fpz-Cz": "Fpz",
        "EEG Pz-Oz": "Pz",
        "EOG horizontal": "EOG",
        "EMG submental": "EMG",
        "EEG Fpz-A2": "Fpz",
        "EEG C3-A2": "C3",
        "EEG C4-A1": "C4",
        "EEG O1-A2": "O1",
        "EEG O2-A1": "O2"
    }

    # Rename channels
    raw.rename_channels(lambda x: channel_mapping.get(x, x))

    # Set channel types
    for ch in raw.ch_names:
        if ch.startswith('EOG'):
            raw.set_channel_types({ch: 'eog'})
        elif ch.startswith('EMG'):
            raw.set_channel_types({ch: 'emg'})

    # Apply standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw
```

### Option 2: Modify EEGPreprocessor to Check Positions

Update `_apply_autoreject` to check for positions first:

```python
def _apply_autoreject(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Apply Autoreject with position check."""
    if not AUTOREJECT_AVAILABLE:
        return raw

    # Check if we have valid channel positions
    montage = raw.get_montage()
    if montage is None:
        logger.warning("No channel positions found - skipping AutoReject")
        return raw

    # Check if positions are valid (not all zeros)
    positions = np.array([ch['loc'][:3] for ch in raw.info['chs']])
    if np.allclose(positions, 0):
        logger.warning("Channel positions are invalid - skipping AutoReject")
        return raw

    # Proceed with AutoReject as normal
    epochs = mne.make_fixed_length_epochs(
        raw, duration=4.0, preload=True, proj=False, verbose=False
    )

    ar = AutoReject(n_jobs=-1, verbose=False)
    ar.fit(epochs)
    epochs_clean = ar.transform(epochs)

    # Convert back to raw
    raw_clean = mne.concatenate_raws(
        [mne.io.RawArray(epoch, epochs.info, verbose=False)
         for epoch in epochs_clean],
        verbose=False
    )

    return raw_clean
```

### Option 3: Use Amplitude-Based Rejection Fallback

When positions aren't available, fall back to simple amplitude rejection:

```python
def _amplitude_based_rejection(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Simple amplitude-based artifact rejection."""
    # Get EEG data
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)

    # Detect high amplitude artifacts (>150 µV)
    threshold = 150e-6
    bad_mask = np.abs(data) > threshold

    # Mark bad channels
    bad_channels = []
    for i, ch_idx in enumerate(picks):
        if np.sum(bad_mask[i]) > 0.1 * data.shape[1]:  # >10% bad
            bad_channels.append(raw.ch_names[ch_idx])

    raw.info['bads'] = bad_channels

    # Interpolate bad segments
    # ... implementation ...

    return raw
```

## Recommended Implementation

1. **For test_full_pipeline.py**: Add montage setting before preprocessing
2. **For EEGPreprocessor**: Add position check with graceful fallback
3. **For FlexibleEEGPreprocessor**: Already handles this correctly with `has_positions` check

## Testing Strategy

1. Test with Sleep-EDF data (no positions)
2. Test with data that has positions
3. Verify both paths work correctly
4. Ensure no regression in processing quality

## References

- AutoReject documentation on `augment` parameter
- MNE-Python montage documentation
- Sleep-EDF dataset description
