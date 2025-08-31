# Dataset Path Alignment Status

**LAST UPDATED**: 2025-08-30 19:45 PST
**OVERALL STATUS**: 60% Complete

## Summary

We successfully centralized Sleep-EDF paths. TUAB/TUEV need the same treatment. Documentation and technical debt items remain.

## Component Status

### Sleep-EDF ✅ COMPLETE
- **DataConfig methods**: ✅ `sleep_edf_version`, `sleep_edf_root`, `get_sleep_edf_psg_file()`
- **Fixtures**: ✅ `sleep_edf_path`, `sleep_edf_dir`
- **Synthetic fallback**: ✅ `_create_synthetic_sleep_edf()`
- **Tests marked**: ✅ 9 files with `@pytest.mark.data`
- **Hardcoded paths**: ✅ Eliminated (only 1 in mock)
- **Version strings**: ✅ Eliminated outside config

### TUAB 🟡 PARTIAL
- **DataConfig root**: ✅ `tuab_root` exists
- **Dataset class**: ✅ `TUABDataset` exists, takes `root_dir`
- **Deterministic picker**: ❌ No `get_tuab_sample_file()` yet
- **Fixtures**: ❌ Only tiny synthetic fixtures exist
- **Synthetic fallback**: ❌ No `_create_synthetic_tuab()` yet
- **Integration tests**: ❌ Not marked with `@pytest.mark.data`

### TUEV 🟡 PARTIAL  
- **DataConfig root**: ✅ `tuev_root` exists
- **Dataset class**: ✅ `TUEVDataset` exists, takes `root_dir`
- **Deterministic picker**: ❌ No `get_tuev_sample_file()` yet
- **Fixtures**: ❌ None found
- **Synthetic fallback**: ❌ No `_create_synthetic_tuev()` yet
- **Integration tests**: ❌ None found

## Technical Debt 🔴

1. **MNE Export API** (conftest.py:335)
   - Using deprecated `raw.export()` instead of `mne.export.export_raw()`
   
2. **Channel Names** (conftest.py:329)
   - Using `"EEG Fpz-Cz"` instead of `"Fpz-Cz"`
   
3. **Missing Test Marker**
   - `tests/integration/test_yasa_channel_aliasing.py` needs `@pytest.mark.data`
   
4. **Pre-commit Patterns**
   - Missing TUAB/TUEV patterns (v3.0.1, 01_tcp_ar, etc.)

## Documentation Drift 🟠

- **AGENTS.md**: Contains `datasets/external/sleep-edf` references
- **Archived docs**: Old paths (acceptable as historical record)
- **Policy docs**: Need `DATA_PATHS_POLICY.md` and `TEST_DATA_POLICY.md`

## Action Plans Created

1. **TUAB_TUEV_ALIGNMENT_PLAN.md** - How to achieve parity
2. **FINAL_POLISH_CHECKLIST.md** - All remaining items
3. **FINAL_ALIGNMENT_ACTION_PLAN.md** - Prioritized implementation

## Verification Status

```bash
# Sleep-EDF hardcoded paths: 0 ✅
rg 'SC4001E0-PSG' src/ tests/ --type py | grep -v mock | wc -l
# Result: 0

# TUAB hardcoded paths: UNKNOWN ❓
rg 'v3\.0\.1/edf' src/ tests/ --type py | wc -l
# Need to check

# Unsorted globs: 0 ✅
rg '\.glob\(' src/ tests/ | grep -v sorted | wc -l
# Result: 0

# Tests properly marked: PARTIAL 🟡
# 9 files have @pytest.mark.data
# Some integration tests still need marking
```

## Next Steps (Priority Order)

### HIGH Priority (Blocking)
1. Fix MNE export API
2. Fix channel names
3. Add missing @pytest.mark.data to test_yasa_channel_aliasing.py
4. Expand pre-commit patterns

### MEDIUM Priority (Important)
5. Implement TUAB/TUEV DataConfig methods
6. Create synthetic fallback functions
7. Add real-data fixtures
8. Create integration tests

### LOW Priority (Nice to Have)
9. Update AGENTS.md
10. Create policy documentation
11. Remove unused files

## Why This Matters

- **NO PARALLEL UNIVERSES**: We're using existing infrastructure
- **DETERMINISTIC**: All file selection is reproducible
- **CI-FRIENDLY**: Tests work without 100GB downloads
- **MAINTAINABLE**: Clear patterns for future datasets
- **PRODUCTION-SAFE**: Test code never leaks to production