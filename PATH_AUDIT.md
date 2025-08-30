# Dataset Path Audit - August 30, 2025

## Current ACTUAL Dataset Structure
```
/data/datasets/
├── tuab/                                     # TUH Abnormal dataset
│   ├── edf/                                  # EDF files organized by split
│   └── AAREADME.txt
├── tuev/                                     # TUH Events dataset  
│   ├── edf/                                  # EDF files organized by split
│   └── AAREADME.txt
└── sleep-edf/                                # Sleep-EDF dataset
    └── sleep-edf-database-expanded-1.0.0/   # Extracted folder
        ├── sleep-cassette/                   # 101 PSG + 88 Hypnogram files
        ├── sleep-telemetry/                  # 44 PSG + 44 Hypnogram files
        └── metadata files
```

## Incorrect Path References Found

### 1. Sleep-EDF paths (19 references found)
**OLD (wrong):** `data/datasets/external/sleep-edf/sleep-cassette/`
**NEW (correct):** `data/datasets/sleep-edf/sleep-edf-database-expanded-1.0.0/sleep-cassette/`

Files with wrong paths:
- `/src/brain_go_brrr/application/pipeline/parallel.py`
- `/tests/conftest.py`
- `/tests/fixtures/benchmark_data.py`
- `/tests/integration/api/test_api_sleep_edf.py` (3 occurrences)
- `/tests/integration/test_eegpt_integration.py`
- `/tests/integration/test_parallel_pipeline.py`
- `/tests/integration/test_sleep_enhanced.py`
- `/tests/integration/test_yasa_channel_aliasing.py`
- `/CLAUDE.md` (lines 602, 653, 293-294)

### 2. TUAB/TUEV paths
These use environment variable `BGB_DATA_ROOT` and relative paths.
- Training scripts expect: `$BGB_DATA_ROOT/datasets/tuab/`
- Need to verify this matches actual structure

### 3. CLAUDE.md outdated structure (lines 290-295)
```
OLD:
│   └── datasets/          # EEG datasets
│       └── external/
│           └── sleep-edf/ # 197 PSG recordings (✅ downloaded)

SHOULD BE:
│   └── datasets/          # EEG datasets
│       ├── tuab/          # TUH Abnormal dataset
│       ├── tuev/          # TUH Events dataset
│       └── sleep-edf/     # Sleep-EDF dataset (197 recordings)
```

## Action Plan
1. Update all Sleep-EDF paths (19 references)
2. Fix CLAUDE.md project structure documentation
3. Verify TUAB/TUEV paths with environment variable
4. Test that all datasets can be loaded
5. Run integration tests to verify nothing broke