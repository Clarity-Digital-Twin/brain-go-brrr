# EEGPT Linear Probe Channel Mapping Explained

## The Channel Naming Challenge

The TUAB dataset uses older EEG channel naming conventions (T3, T4, T5, T6) while EEGPT expects modern naming (T7, T8, P7, P8). This document explains how we resolved this mismatch.

## Historical Context

- **Old 10-20 System**: Used T3, T4, T5, T6 for temporal electrodes
- **Modern 10-20 System**: Renamed these to T7, T8, P7, P8 for clarity
- **TUAB Dataset**: Uses old naming in raw EDF files
- **EEGPT Model**: Trained with modern 20-channel layout

## Our Solution

### 1. Channel Mapping (in `tuab_dataset.py`)
```python
CHANNEL_MAPPING = {
    "T3": "T7",  # Left temporal
    "T4": "T8",  # Right temporal
    "T5": "P7",  # Left posterior temporal
    "T6": "P8",  # Right posterior temporal
    # ... other mappings
}
```

### 2. Standard Channels (20 channels for EEGPT)
```python
STANDARD_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8",
    "O1", "O2", "OZ"
]
```

### 3. Missing Channels
- **FPZ**: Often missing in TUAB recordings
- **OZ**: Sometimes missing in TUAB recordings
- These are handled gracefully with zero-padding

## Processing Flow

1. **Load EDF**: Raw file has old naming (T3, T4, T5, T6)
2. **Rename Channels**: Map to modern names (T7, T8, P7, P8)
3. **Select Available**: Pick channels that exist and are in STANDARD_CHANNELS
4. **Zero-pad Missing**: Fill missing channels with zeros
5. **Output**: Consistent 20-channel array for EEGPT

## Why This Matters

- **Data Integrity**: Ensures correct electrode signals reach the model
- **Compatibility**: Aligns TUAB data with EEGPT's expected format
- **Performance**: Prevents the model from seeing zero-filled channels where data exists

## Verification

After implementing the fix:
- ✅ Channel warnings only show legitimately missing channels (FPZ, OZ)
- ✅ No more warnings about T3/T4/T5/T6 being missing
- ✅ Model receives actual EEG data, not zeros
