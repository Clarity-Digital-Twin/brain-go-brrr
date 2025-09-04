# Channel Configuration Policy

## Single Source of Truth (SSOT)

The channel configuration for this project is defined in:
- **`src/brain_go_brrr/infra/data/channels.py`** - Channel definitions
- **`src/brain_go_brrr/infra/preprocessing/`** - Preprocessors that enforce these definitions

## Dataset-Specific Channel Requirements

### TUAB (TUH Abnormal EEG)
- **Expected channels**: 19 (defined in `CHANNELS_TUAB_19`)
- **Accepted in practice**: 18-19 channels
- **Missing channel**: Oz is commonly absent in real TUAB files
- **Key difference**: NO Fz channel
- **Channel order**: FP1, FP2, F7, F3, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, OZ, O2

### TUEV (TUH EEG Events)
- **Expected channels**: 20 (defined in `CHANNELS_TUEV_20`)
- **Accepted in practice**: Exactly 20 channels (strict)
- **Key differences**: HAS Fz, NO Fpz, HAS Oz
- **Channel order**: FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2, OZ

## Channel Name Mapping

The system automatically maps old naming conventions to modern standards:
- T3 → T7
- T4 → T8
- T5 → P7
- T6 → P8

This mapping is applied during preprocessing to ensure consistency.

## Processing Pipeline

```
Raw EDF → Preprocessor (SSOT) → Normalized Output → Model/Analysis
             ↑
             └── Single place that handles:
                 - Channel mapping (T3→T7, etc.)
                 - Channel selection (18-20 channels)
                 - Resampling (256Hz)
                 - Filtering
                 - Autoreject (optional)
```

## Testing Policy

All tests MUST:
1. Use the SSOT preprocessor for normalization
2. Assert on normalized output, NOT raw EDF headers
3. Check strict invariants:
   - Channel count (18-19 for TUAB, 20 for TUEV)
   - Modern naming only (T7/T8/P7/P8, not T3/T4/T5/T6)
   - Sampling rate exactly 256Hz
   - Voltage in microvolts range (1e-7 to 5e-3 V)

## Implementation Guidelines

### When Processing New Data

1. **Always use the preprocessor** from `brain_go_brrr.infra.preprocessing`
2. **Never test raw EDF headers** - test the normalized output
3. **Log channel configurations** in provenance for reproducibility

### When Adding New Datasets

1. Define the channel set in `channels.py`
2. Create a preprocessor in `infra/preprocessing/`
3. Write tests that assert the normalized contract
4. Document any dataset-specific quirks here

## Known Issues and Resolutions

### TUAB Missing Oz
- **Issue**: Many real TUAB files are missing the Oz channel
- **Resolution**: Preprocessor accepts 18 or 19 channels
- **Rationale**: Prevents pipeline failures on real data

### Channel Name Variations
- **Issue**: Files may have prefixes (EEG) or suffixes (-REF)
- **Resolution**: Preprocessor strips these and maps to standard names
- **Example**: "EEG T3-REF" → "T7"

## EEGPT Model Compatibility

The EEGPT model was pretrained on 58 channels but our downstream tasks use subsets:
- **TUAB tasks**: 18-19 channels
- **TUEV tasks**: 20 channels
- **Channel adapter**: Available when needed (1x1 conv layer)

## Provenance Tracking

All preprocessing steps log:
- Final channel count
- Channel names in order
- Any missing channels that were expected
- Sampling rate after resampling
- Dataset name
- Processing timestamp
