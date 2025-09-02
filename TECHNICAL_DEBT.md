# 🚀 TECHNICAL DEBT STATUS (September 2, 2025)

## ✅ MAJOR DEBT PAID DOWN (Aug 28 - Sep 2, 2025)

### ✅ RESOLVED: Sleep-EDF/TUAB/TUEV Dataset Alignment
**Completed**: September 1, 2025
- Unified all datasets through DataConfig
- Eliminated ALL hardcoded paths
- Created parallel test paths (synthetic + real data)
- All 12 real data tests passing (5 TUAB, 7 TUEV)
- Coverage improved from 66% → 86%

### ✅ RESOLVED: Training Crash & Resume Issues
**Completed**: September 1, 2025
- Fixed deterministic resume with epoch_indices tracking
- Added sample-level checkpoint precision
- Implemented intra-epoch checkpointing (every 500 batches)
- Fixed DataLoader optimization (4 workers, pin_memory, persistent)
- Training now running stably at 76% AUROC and climbing

### ✅ RESOLVED: Architecture Unification
**Completed**: August 28, 2025
- Normalization SSOT in wrapper
- Datasets emit raw mV, wrapper normalizes
- experiments/ now uses src/ components
- Channel validation enforces correct order
- META schema unified across all datasets

### ✅ RESOLVED: PyTorch Lightning Bug
**Completed**: August 25, 2025
- Replaced with pure PyTorch training
- Added crash recovery and auto-resume
- No more hanging on large datasets

## 🔴 REMAINING CRITICAL DEBT

### 1. EEGPT Model File Explosion (11 DUPLICATE FILES!)
**Status**: Not addressed yet
**Impact**: Confusing, maintenance nightmare
```
src/brain_go_brrr/infra/ml_models/
├── eegpt_wrapper.py         # Used by experiments - KEEP
├── eegpt_compat.py          # Used by src API - MAYBE KEEP
├── eegpt_config.py          # Config - KEEP
├── eegpt_architecture.py    # Reference - KEEP
├── eegpt_model.py           # DELETE - duplicate
├── eegpt_probe_unified.py   # DELETE - duplicate
├── eegpt_normalize.py       # DELETE - in wrapper now
├── eegpt_feature_extractor.py # DELETE - in wrapper now
└── [3+ more duplicates]     # DELETE ALL
```
**Action**: Consolidate to 3-4 files max

### 2. Experiments Folder Redundancies
**Status**: Partially addressed
**Files to clean**:
```
experiments/eegpt_linear_probe/
├── utils/custom_collate_fixed.py  # DELETE - old hack
├── utils/collate_tuab.py          # Should use src/
├── utils/collate_tuev.py          # Should use src/
├── datasets/tuab_mne_dataset.py   # DELETE - use src/
├── datasets/tuev_mne_dataset.py   # Move to src/ first
└── docs/ [9 redundant files]      # Keep only CHANNEL_SPECIFICATIONS.md
```

## 🟡 MEDIUM PRIORITY DEBT

### 1. Intelligent Channel Routing
**Issue**: API doesn't route based on channel count
- <19 channels should go to YASA only
- 19+ channels can use both EEGPT and YASA
**Files**: `src/brain_go_brrr/api/routers/sleep.py`

### 2. TestClient Dependency Override Bug
**Issue**: FastAPI TestClient breaks with file uploads + DI
**Impact**: Can't properly test concurrent uploads
**Workaround**: Tests skipped with documentation

### 3. AutoReject Memory Requirements
**Issue**: Needs 100+ epochs (2+ minutes of data)
**Impact**: Slow tests, high memory usage
**Solution**: Create "fast mode" for testing

## 🟢 LOW PRIORITY IMPROVEMENTS

### 1. Performance Optimizations
- [ ] Batch processing for multiple files
- [ ] GPU memory optimization
- [ ] Cached preprocessing results

### 2. Testing Infrastructure
- [ ] Docker-compose for Redis/PostgreSQL
- [ ] Better mocking strategies
- [ ] Faster test modes

### 3. Model Management
- [ ] Auto-download EEGPT if missing
- [ ] Model versioning system
- [ ] Support multiple EEGPT variants

## 📊 DEBT METRICS

| Category | Before (Aug 28) | After (Sep 2) | Improvement |
|----------|----------------|---------------|-------------|
| Test Coverage | 66% | 86% | +20% ✅ |
| Test Count | 751 | 899 | +148 ✅ |
| Hardcoded Paths | 50+ | 0 | -100% ✅ |
| Training Stability | Crashes at 58h | Running stable | ✅ |
| Dataset Alignment | 3 divergent | Fully unified | ✅ |
| EEGPT Files | 11 duplicates | 11 (unchanged) | ❌ TODO |
| Experiments Cruft | High | Medium | 🟡 Partial |

## 🎯 NEXT PRIORITIES

1. **URGENT**: Clean up EEGPT model files (11 → 4 files)
2. **HIGH**: Remove experiments/ redundancies
3. **MEDIUM**: Implement channel-based routing in API
4. **LOW**: Performance optimizations

## 📝 NOTES

- All CI/CD pipelines green
- Production training running with auto-recovery
- Test suite split: unit (fast) vs integration (real data)
- Pre-commit hooks enforce no hardcoded paths

**Last Updated**: September 2, 2025
**Next Review**: After EEGPT cleanup
