# Final Polish Checklist - Brain-Go-Brrr Dataset Alignment

**STATUS**: Sleep-EDF COMPLETE, TUAB/TUEV PENDING
**LAST UPDATED**: 2025-08-30

## Critical Issues from External Audit

### 1. ✅ Sleep-EDF Path Centralization (COMPLETED)
- [x] DataConfig owns all Sleep-EDF paths
- [x] No hardcoded `SC4001E0-PSG.edf` outside mocks
- [x] No `sleep-edf-database-expanded-1.0.0` strings outside config
- [x] All globs sorted for determinism
- [x] Synthetic fallback for CI

### 2. 🟡 TUAB/TUEV Parity (IN PROGRESS - DOCUMENTED)
**Current State:**
- [x] `DataConfig.tuab_root` and `tuev_root` exist
- [x] TUABDataset and TUEVDataset classes exist (take root_dir param)
- [ ] No real-data fixtures yet (only tiny synthetic)
- [ ] No `get_tuab_sample_file()` methods
- [ ] No synthetic fallback functions
- [ ] Integration tests not marked with @pytest.mark.data

**Plan Created:** See `TUAB_TUEV_ALIGNMENT_PLAN.md`

### 3. 🔴 Technical Debt Items

#### A. Synthetic EDF Export API Issue
**Problem:** Using deprecated `raw.export()` instead of stable API
```python
# Current (may break):
raw.export(str(edf_path), fmt="edf")

# Should be:
mne.export.export_raw(raw, str(edf_path), fmt="edf", physical_range=(None, None))
```

**Also:** Channel names should be `["Fpz-Cz", "Pz-Oz"]` not `["EEG Fpz-Cz", "EEG Pz-Oz"]`

#### B. TEST-ONLY EEG Re-tag Heuristic
**Location:** `tests/unit/domain/sleep/test_analysis.py`
**Issue:** Re-tags channels as EEG based on name patterns
**Fix:** Should skip test if no EEG channels instead of guessing

#### C. Missing @pytest.mark.data Coverage
Files still needing markers:
- `tests/unit/domain/sleep/test_montage_detection.py` ✅ (done)
- `tests/integration/test_yasa_integration.py` ✅ (done)
- Any future TUAB/TUEV integration tests

#### D. Pre-commit Hook Expansion Needed
Current patterns don't catch:
- TUAB patterns: `v3.0.1`, `01_tcp_ar`, `tuh_eeg_abnormal`
- TUEV patterns: `v2.0.0`, `tuev_v2`
- Generic version patterns: `v\d+\.\d+\.\d+`

#### E. Documentation Drift
Files with outdated paths:
- `AGENTS.md` - may reference old paths
- Various planning docs in root (OK to leave as historical record)

### 4. 🟢 What's Already Clean

- [x] Sleep-EDF fixtures use DataConfig
- [x] `parallel.py` uses DataConfig
- [x] Pre-commit hook exists (needs expansion)
- [x] Documentation in CLAUDE.md, QUICK_START.md, TRAINING.md updated
- [x] CI can run without datasets (synthetic fallback)

## Priority Action Items

### HIGH Priority (Blocking)
1. [ ] Fix synthetic EDF export to use `mne.export.export_raw()` with `physical_range`
2. [ ] Change Sleep-EDF synthetic channels to `["Fpz-Cz", "Pz-Oz"]` (no "EEG " prefix)
3. [ ] Expand pre-commit hook patterns for TUAB/TUEV

### MEDIUM Priority (Important)
4. [ ] Implement DataConfig methods: `get_tuab_sample_file()`, `get_tuev_sample_file()`
5. [ ] Create synthetic fallback functions: `_create_synthetic_tuab()`, `_create_synthetic_tuev()`
6. [ ] Add fixtures: `tuab_sample_path`, `tuev_sample_path`
7. [ ] Create one integration test per dataset with @pytest.mark.data

### LOW Priority (Nice to Have)
8. [ ] Add pytest_runtest_setup guard to enforce @pytest.mark.data
9. [ ] Clean up AGENTS.md documentation
10. [ ] Add --outdir and summary.json to pipeline

## Verification Commands

```bash
# 1. Check Sleep-EDF literals (should be 0)
rg 'SC4001E0-PSG' --type py | grep -v mock | wc -l

# 2. Check TUAB literals (should be 0 after fix)
rg 'v3\.0\.1/edf|01_tcp_ar' --type py | grep -v config | wc -l

# 3. Check all globs are sorted
rg '\.glob\(' --type py | grep -v sorted

# 4. Verify @pytest.mark.data coverage
rg -l 'sleep_edf_path|tuab_sample_path' tests | while read f; do
  rg -q '@pytest.mark.data' "$f" || echo "Missing @data: $f"
done

# 5. Test without data (should skip/pass)
pytest -m "not data" --tb=short

# 6. Test with synthetic
BGB_ALLOW_SYNTH_SLEEP_EDF=1 pytest -k sleep --tb=short
```

## Definition of "DONE"

The codebase is considered fully aligned when:

1. **Zero hardcoded paths** outside of DataConfig and test mocks
2. **All datasets** (Sleep-EDF, TUAB, TUEV) have:
   - DataConfig methods for path resolution
   - Real-data fixtures with synthetic fallbacks
   - At least one @pytest.mark.data integration test
3. **Pre-commit hook** catches all dataset path patterns
4. **Tests pass** in three modes:
   - No data mounted (skip cleanly)
   - Synthetic fallback (pass with env vars)
   - Real data mounted (pass with --run-data)
5. **Documentation** shows DataConfig usage, not literal paths

## Why This Matters

- **No Parallel Universes**: Single implementation, single source of truth
- **CI Stability**: Tests don't randomly fail when data missing
- **Maintainability**: New devs can run tests without 100GB downloads
- **Production Safety**: Real code never has test-only fallbacks
- **DeepMind/Google Standard**: Synthetic units + golden real-data checks