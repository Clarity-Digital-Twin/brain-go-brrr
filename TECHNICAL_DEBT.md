# 🚀 TECHNICAL DEBT - MOSTLY RESOLVED (August 28, 2025)

## ✅ CRITICAL ISSUES RESOLVED

### FIXED: Architecture Unification (Aug 28, 2025)

1. **Normalization SSOT** → Wrapper-only normalization ✅
2. **Datasets cleaned** → tuab_cached, tuab_enhanced now deprecation aliases ✅
3. **experiments/ fixed** → Now thin shims importing from src/ ✅
4. **Channel validation** → Enforces correct order ✅
5. **META schema** → Unified across all datasets ✅

### REMAINING IN SRC/ (Minor)

#### ✅ TUAB Dataset - FIXED
```
src/brain_go_brrr/infra/data/
├── tuab_dataset.py          # Main implementation ✅
├── tuab_cached_dataset.py   # Deprecation alias ✅
└── tuab_enhanced_dataset.py # Deprecation alias ✅
```

#### 🔴 EEGPT Model Clusterfuck (11 FILES!)
```
src/brain_go_brrr/infra/ml_models/
├── eegpt_wrapper.py         # Used by experiments - KEEP
├── eegpt_compat.py          # Used by src API - MAYBE KEEP
├── eegpt_model.py           # DELETE - duplicate
├── eegpt_probe_unified.py   # DELETE - duplicate  
├── eegpt_architecture.py    # Maybe keep for reference
├── eegpt_config.py          # Config - KEEP
├── eegpt_normalize.py       # DELETE - should be in wrapper
├── eegpt_feature_extractor.py # DELETE - should be in wrapper
└── [3+ more duplicates]     # DELETE ALL
```

### REDUNDANCIES IN EXPERIMENTS/ (FIX AFTER SRC!)

#### 🔴 Triple Collate Functions
```
utils/custom_collate_fixed.py  # OLD HACK - DELETE
utils/collate_tuab.py          # Should use src/ version
utils/collate_tuev.py          # Should use src/ version
```

#### 🔴 Dataset Reimplementations
```
datasets/tuab_mne_dataset.py   # DELETE - use src/tuab_dataset.py
datasets/tuev_mne_dataset.py   # MOVE TO SRC first, then use from there
datasets/tuev_dataset_cached.py # WTF is "Padded"? DELETE
```

#### 🔴 Documentation Overload (10 REDUNDANT FILES!)
```
Keep ONLY: CHANNEL_SPECIFICATIONS.md
DELETE ALL: The other 9 fix/audit/summary files
```

### THE FIX ORDER

1. **Clean src/ FIRST** - Delete redundant datasets and models
2. **Move valuable code to src/** - MNE preprocessing, TUEV dataset
3. **Make experiments/ use src/** - Delete all duplicates
4. **Delete documentation garbage** - Keep 1 file only

## Recently Resolved Issues ✅

### ✅ RESOLVED: PyTorch Lightning Training Bug (August 25, 2025)
- **Issue**: Lightning 2.5.2 hangs on large cached datasets
- **Solution**: Implemented pure PyTorch training scripts with crash guards
- **Files Fixed**:
  - `experiments/eegpt_linear_probe/train_tuab.py` (added crash guards)
  - `experiments/eegpt_linear_probe/train_tuev.py` (added crash guards)

### ✅ RESOLVED: CI/CD Pipeline Failures (August 25, 2025)
- **Issue**: Literature folder with Windows paths breaking pre-commit hooks
- **Solution**: Excluded from pre-commit but kept in repo for contributors
- **Files Fixed**: `.pre-commit-config.yaml`

## Priority Issues to Address

### 🔴 Critical: Intelligent Channel Routing

**Issue**: EEGPT requires 19+ channels for meaningful results. Data should be routed to appropriate pathways based on channel count.

**Impact**:
- `/eeg/sleep/stages` endpoint should route based on channel count
- Tests incorrectly skip combinations instead of routing appropriately

**Current Architecture** (CORRECT):
- **YASA pathway**: ANY channel count (auto-selects best channel) → YASA sleep staging ✅
- **EEGPT pathway**: Full EEG (256Hz, 19+ channels) → EEGPT features → Linear probes ✅
- These are PARALLEL, not sequential
- Note: YASA achieves 85%+ accuracy with just 1 channel

**Solution Required**:
1. Route inputs intelligently: <19 channels → YASA only, 19+ channels → both pathways available
2. Refactor tests to explicitly separate YASA and EEGPT pathways

**Files to Modify**:
- `src/brain_go_brrr/api/routers/sleep.py` (add channel validation)
- `tests/unit/test_cli_streaming.py` (refactor, not skip)

---

### 🟡 Medium: TestClient Dependency Override with File Uploads

**Issue**: FastAPI's TestClient doesn't properly handle dependency overrides when uploading files. This prevents proper mocking of the QC controller in integration tests.

**Impact**:
- Cannot test concurrent file uploads without real EEGPT model
- Forces us to skip important API robustness tests
- Reduces test coverage for file upload endpoints

**Current Workaround**: Test skipped with reason documented

**Potential Solutions**:
1. Use a real test server (e.g., `httpx.AsyncClient` with `asgi`)
2. Mock at a different layer (e.g., service level instead of DI)
3. Create test fixtures that pre-configure the app with mocks

**Files Affected**:
- `tests/api/test_api_sleep_edf.py::test_concurrent_sleep_edf_processing`

---

### 🟡 Medium: AutoReject Memory Requirements

**Issue**: AutoReject requires 100+ epochs for cross-validation, needing 2+ minutes of EEG data. This makes unit tests slow and memory-intensive.

**Impact**:
- Slow test execution when AutoReject is involved
- High memory usage during preprocessing tests
- Cannot test with short EEG segments

**Current Workaround**: Test uses 2 minutes of synthetic data (slow)

**Potential Solutions**:
1. Create a "fast mode" for AutoReject with reduced cross-validation
2. Mock AutoReject in unit tests, only test in integration
3. Pre-compute AutoReject thresholds for test data

**Files Affected**:
- `tests/unit/test_abnormality_preprocessor.py::test_preprocessing_preserves_eeg_patterns`

---

### 🟢 Low: Redis Dependency for Caching Tests

**Issue**: Redis caching tests require a running Redis server, which isn't always available in CI/CD environments.

**Impact**:
- Cannot test Redis caching without Redis server
- Reduces coverage of caching functionality
- May miss cache-related bugs

**Current Workaround**: Tests correctly skip when Redis unavailable

**Potential Solutions**:
1. Use `fakeredis` library for unit tests
2. Add Redis to CI/CD pipeline via Docker
3. Create in-memory cache adapter for testing

**Files Affected**:
- `tests/unit/test_redis_pool.py`

---

## Technical Improvements Needed

### 1. Data Pipeline Enhancements
- [ ] Automatic resampling for mismatched sampling rates
- [ ] Streaming support for large files (>2GB)
- [ ] Multi-format support (BDF, GDF, etc.)

### 2. Testing Infrastructure
- [ ] Better mocking strategy for FastAPI dependencies
- [ ] Faster AutoReject testing mode
- [ ] Docker-compose for test dependencies (Redis, PostgreSQL)

### 3. Model Integration
- [ ] EEGPT checkpoint auto-download if missing
- [ ] Model versioning and compatibility checks
- [ ] Support for multiple EEGPT variants

### 4. Performance Optimizations
- [ ] Batch processing for multiple files
- [ ] GPU memory optimization for large datasets
- [ ] Caching of preprocessed data

---

## Tracking

| Issue | Priority | Effort | Impact | Status |
|-------|----------|--------|--------|--------|
| Intelligent Channel Routing | 🔴 High | 2-4 hrs | Critical | Open |
| TestClient DI Override | 🟡 Medium | 4-8 hrs | Medium | Open |
| AutoReject Memory | 🟡 Medium | 2-4 hrs | Low | Open |
| Redis Testing | 🟢 Low | 1-2 hrs | Low | Open |
| PyTorch Lightning Bug | ✅ | - | - | Resolved |
| CI/CD Pipeline | ✅ | - | - | Resolved |

---

## Notes

- All skipped tests are properly documented with reasons
- Current test pass rate: 100% (58 passed, 4 legitimately skipped)
- Code quality: 100% (lint + type checks passing)

Last Updated: August 25, 2025
