# Current Alignment Status - Full Repository Analysis

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



**Date**: 2025-08-31
**Status**: Sleep-EDF Complete, TUAB/TUEV Pending

## Executive Summary

After analyzing the repository from first principles, here's the exact state:

1. **Sleep-EDF**: ✅ FULLY ALIGNED
   - All paths centralized through DataConfig
   - SC4001E0-PSG references are MOCKS ONLY (verified)
   - Deterministic file selection working
   - Synthetic fallback implemented

2. **TUAB/TUEV**: 🔴 NOT ALIGNED
   - Roots exist in DataConfig but no deterministic selection
   - Hardcoded paths still present in test_accuracy.py
   - No synthetic fallbacks
   - No proper fixtures

3. **Technical Debt**: 🟡 MINOR ISSUES
   - MNE export API needs update (1 line fix)
   - Channel names in synthetic data (1 line fix)
   - Missing @pytest.mark.data on some tests

## Verification Results

### Sleep-EDF Hardcoded Paths
```bash
# Found in: tests/integration/test_train_sleep_probe.py
# Lines 135, 241: SC4001E0-PSG references
# STATUS: ✅ VERIFIED AS MOCKS - These create fake test files, not real paths
```

### TUAB Hardcoded Paths
```bash
# Found in multiple files:
scripts/data/verify_tuab_dataset.py:24: "version": "v3.0.1"
scripts/data/download_datasets.py:41: 'TUAB': {'version': 'v3.0.1'...
tests/unit/domain/abnormal/test_accuracy.py:24: config.tuab_root / "v3.0.1/edf/train"
tests/unit/domain/abnormal/test_accuracy.py:31-32: / "01_tcp_ar").glob("*.edf")
# STATUS: ❌ REAL HARDCODED PATHS - Need fixing
```

## What We Have vs What We Need

### DataConfig Status

| Feature                      | Sleep-EDF | TUAB | TUEV |
|------------------------------|-----------|------|------|
| Root property                | ✅        | ✅   | ✅   |
| Version property             | ✅        | ❌   | ❌   |
| Deterministic file picker    | ✅        | ❌   | ❌   |
| Environment override         | ✅        | ✅   | ✅   |

### Test Infrastructure Status

| Component                    | Sleep-EDF | TUAB | TUEV |
|------------------------------|-----------|------|------|
| Real-data fixture            | ✅        | ❌   | ❌   |
| Synthetic fallback function  | ✅        | ❌   | ❌   |
| @pytest.mark.data markers    | ✅        | ❌   | ❌   |
| Integration tests            | ✅        | ❌   | ❌   |

## Critical Path to Full Alignment

### Phase 1: Technical Debt (Quick Fixes)
1. **Fix MNE Export API** (conftest.py:335)
   - Change: `raw.export()` → `mne.export.export_raw()`

2. **Fix Channel Names** (conftest.py:329)
   - Change: `["EEG Fpz-Cz", "EEG Pz-Oz"]` → `["Fpz-Cz", "Pz-Oz"]`

3. **Add Missing Markers**
   - `test_yasa_channel_aliasing.py`: Add `@pytest.mark.data`
   - `test_accuracy.py`: Add `@pytest.mark.data`

### Phase 2: TUAB/TUEV DataConfig Methods
Need to add to `src/brain_go_brrr/application/config/base.py`:
- `tuab_version` property
- `tuev_version` property
- `get_tuab_sample_file()` method
- `get_tuev_sample_file()` method

### Phase 3: Test Infrastructure
Need to add to `tests/conftest.py`:
- `_create_synthetic_tuab()` function
- `_create_synthetic_tuev()` function
- `tuab_sample_path` fixture
- `tuev_sample_path` fixture

### Phase 4: Fix Existing Tests
- Replace hardcoded paths in `test_accuracy.py` with fixtures
- Update any TUAB/TUEV scripts to use DataConfig

### Phase 5: Pre-commit Protection
Expand patterns in `.pre-commit-hooks/check_hardcoded_paths.py`:
- Add TUAB version patterns
- Add TUEV patterns
- Add protocol patterns (01_tcp_ar)

## Key Design Principles

### SOLID/DRY Compliance
- **Single Responsibility**: DataConfig owns ALL dataset paths
- **Open/Closed**: Easy to add new datasets, hard to break existing
- **DRY**: No duplicate path logic anywhere

### Test Isolation
- **Unit Tests**: Synthetic data only (no I/O)
- **Integration Tests**: Real data, marked with @pytest.mark.data
- **CI/CD**: Skip real-data tests when data not mounted

### Determinism
- **Sorted Globs**: All file listings must be sorted
- **First File**: Always select first file from sorted list
- **Reproducible**: Same file selected every run

## Acceptance Criteria

### For Sleep-EDF (✅ MET)
- [x] No hardcoded filenames outside mocks
- [x] No version strings outside config
- [x] All globs sorted
- [x] Fixtures use DataConfig
- [x] Tests marked appropriately

### For TUAB/TUEV (❌ NOT MET)
- [ ] No hardcoded versions
- [ ] No hardcoded protocols (01_tcp_ar)
- [ ] All file access through DataConfig
- [ ] Fixtures available
- [ ] Tests marked with @pytest.mark.data

## Commands to Verify Alignment

```bash
# Check Sleep-EDF alignment (should return 0)
rg 'SC4001E0-PSG' src/ tests/ scripts/ --type py | grep -v mock | wc -l

# Check TUAB alignment (should return 0 when complete)
rg 'v3\.0\.1|01_tcp_ar' src/ tests/ scripts/ --type py | grep -v DataConfig | wc -l

# Check for unsorted globs (should return 0)
rg '\.glob\(' src/ tests/ | grep -v sorted | wc -l

# Check test markers (should show all marked)
rg -l 'read_raw_edf|sleep_edf_path|tuab' tests/ | xargs -I{} sh -c 'rg -q "@pytest.mark.data" {} || echo "NEEDS @data: {}"'
```

## Conclusion

The Sleep-EDF alignment is COMPLETE and working. The patterns are established and proven. We now need to apply the exact same patterns to TUAB/TUEV to achieve full alignment.

The work is well-defined, low-risk, and follows established patterns. No new architecture needed - just consistent application of what's already working for Sleep-EDF.
