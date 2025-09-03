# Unit Conventions for EEG Data

## Single Source of Truth: Volts (SI Units)

This codebase follows a strict convention for EEG data units to ensure consistency and prevent scaling errors.

## Internal Representation

- **All internal tensors and caches**: Volts (V) - SI base unit
- **MNE convention**: Always returns data in Volts
- **Cache format**: Data stored in Volts with metadata indicating unit="V"
- **Model input**: Normalized from Volts using std=50e-6 V (50 μV typical EEG scale)

## Data Flow

```
EDF File (variable units) 
    → MNE Loader (converts to V) 
    → Cache (stores in V)
    → Dataset (returns V)
    → EEGPT Wrapper (normalizes with std=50μV)
    → Model (expects N(0,1))
```

## Unit Conversion Table

| Format | Typical Unit | Conversion to V |
|--------|--------------|-----------------|
| EDF import | μV or mV | Automatic via MNE |
| Cache storage | V | No conversion |
| Model input | Normalized | x / 50e-6 |
| EDF export | μV | x * 1e6 |

## Validation

The system includes runtime validation to catch unit errors:

1. **First batch check**: Verifies q99 of data is in expected range (1e-6 to 5e-2 V)
2. **Cache metadata**: Validates unit field is "V"
3. **Normalization check**: Logs wrapper std to ensure proper scaling

## Common Pitfalls

- **DO NOT** manually scale data from cache (it's already in Volts)
- **DO NOT** assume EDF files are in any specific unit (MNE handles conversion)
- **DO NOT** trust legacy cache metadata without verification (use data statistics)

## Legacy Cache Migration

Some older caches may have incorrect metadata (e.g., unit="mV" when data is actually in Volts).
To verify and fix:

```python
# Check actual data scale
import numpy as np
data = load_cache_file()
q99 = np.quantile(np.abs(data), 0.99)
print(f"q99: {q99:.2e} V")

# Expected ranges:
# - Volts: 1e-6 to 1e-4 (1μV to 100μV)
# - Millivolts (wrong): 1e-3 to 1e-1
# - Microvolts (wrong): 1.0 to 100.0

# Fix metadata if needed
meta['unit'] = 'V'
meta['source_unit'] = 'mV'  # Document original label
meta['scale_applied'] = 1.0  # No scaling needed if already in V
```

## References

- MNE documentation on units: https://mne.tools/stable/auto_tutorials/intro/20_events_from_raw.html
- EEGPT paper: Expects data normalized with typical EEG std of 50μV
- EDF specification: Physical dimension field specifies units