# 🚀 TECHNICAL DEBT STATUS (September 2, 2025)

## ✅ MASSIVE DEBT PAID DOWN (Aug 28 - Sep 2, 2025)

### ✅ COMPLETE: Sleep-EDF/TUAB/TUEV Dataset Alignment
**Status**: FULLY RESOLVED
- ✅ All datasets unified through DataConfig
- ✅ No hardcoded dataset paths in application/tests (verified by pre-commit; tooling scripts may contain constants by design)
- ✅ Parallel test paths working (synthetic + real data)
- ✅ 12 real data tests passing (5 TUAB, 7 TUEV)
- ✅ Coverage improved 66% → 86%

### ✅ COMPLETE: Training Crash & Resume Issues
**Status**: FULLY RESOLVED
- ✅ Deterministic resume with epoch_indices tracking
- ✅ Sample-level checkpoint precision implemented
- ✅ Intra-epoch checkpointing (every 500 batches)
- ✅ DataLoader optimized (4 workers, pin_memory, persistent)
- ✅ Training stable; AUROC improving toward target (see run logs for current value)

### ✅ COMPLETE: Architecture Unification
**Status**: FULLY RESOLVED
- ✅ Normalization SSOT in EEGPTWrapper
- ✅ Datasets emit raw mV, wrapper normalizes
- ✅ experiments/ uses src/ components (verified imports)
- ✅ Channel validation enforces correct order
- ✅ META schema unified across all datasets

### ✅ COMPLETE: Experiments Folder Cleanup
**Status**: FULLY RESOLVED
- ✅ NO duplicate collate functions (verified - none exist)
- ✅ NO dataset reimplementations (datasets/ dir doesn't exist)
- ✅ Training imports from src/brain_go_brrr (verified)
- ✅ Old files only in archive/ folders

### ✅ COMPLETE: Dataset Deprecation Aliases
**Status**: FULLY RESOLVED
- ✅ tuab_cached_dataset.py → Deprecation alias to TUABDataset
- ✅ tuab_enhanced_dataset.py → Deprecation alias to TUABDataset
- ✅ Proper deprecation warnings in place

## 🟡 REMAINING MINOR ISSUES

### 1. EEGPT Model Files (NOT 11 - Only 6!)
**Reality Check**: Only 6 files exist (not 11 as previously claimed):
```
src/brain_go_brrr/infra/ml_models/
├── __init__.py
├── eegpt_architecture.py    # Model architecture - KEEP
├── eegpt_compat.py          # API compatibility - KEEP
├── eegpt_probe_unified.py   # Unified probe head - REVIEW
├── eegpt_wrapper.py         # Main wrapper - KEEP
└── linear_probe.py          # Two-layer probe - KEEP
```

**Action**: Only potential cleanup is deciding between `eegpt_probe_unified.py` and `linear_probe.py`
- Current: experiments use `linear_probe.py::TwoLayerProbe`
- Decision needed: Keep both or consolidate?

### 2. Channel Routing (Still Open)
**Current**: API rejects <19 channels with 400 error
**Needed**: Intelligent routing
- <19 channels → Route to YASA automatically
- ≥19 channels → Both EEGPT and YASA available
**File**: `src/brain_go_brrr/api/routers/sleep.py`

### 3. Documentation in experiments/
**Current**: 3 docs remain
```
experiments/eegpt_linear_probe/docs/
├── README.md
├── MNE_INTEGRATION_README.md
└── CHANNEL_SPECIFICATIONS.md
```
**Decision**: Keep as reference or remove?

## 🟢 LOW PRIORITY (Nice to Have)

### TestClient File Upload Issue
- FastAPI TestClient doesn't handle dependency overrides with file uploads
- Test skipped with documentation
- Alternatives: httpx/ASGI or service-level mocks

### AutoReject Performance
- ChunkedAutoRejectProcessor implemented
- FakeAutoReject used in unit tests
- "Fast mode" for real AutoReject not implemented

### Redis Testing
- Unit tests use mocking (not fakeredis)
- CI doesn't require Redis service
- Could add Docker-compose for integration tests

### CI Hardcoded Paths Check (Optional)
- Pre-commit hook exists locally
- CI runs other guards but not this specific hook
- Could add explicit step to CI pipeline

## 📊 ACHIEVEMENTS SUMMARY

| Category | Before (Aug 28) | After (Sep 2) | Status |
|----------|----------------|---------------|--------|
| Test Coverage | 66% | ~86% | ✅ +20% |
| Test Count | 751 | 899 | ✅ +148 |
| Hardcoded Paths | 50+ | 0 | ✅ ELIMINATED |
| Training Stability | Crashes | Stable w/ auto-recovery | ✅ FIXED |
| Dataset Alignment | Divergent | Fully unified | ✅ COMPLETE |
| Experiments Cleanup | Duplicates | Clean, uses src/ | ✅ COMPLETE |
| EEGPT Files | Claimed 11 | Actually 6 (clean) | ✅ CLEAN |
| CI/CD | Failing | Configured; local checks pass | ✅ FIXED |

## 🎯 ACTUAL REMAINING WORK

1. **Channel Routing** - Add intelligent routing in API (2-4 hours)
2. **Probe Consolidation** - Decide on eegpt_probe_unified.py vs linear_probe.py (1 hour)
3. **Docs Cleanup** - Remove or mark as reference the 3 experiment docs (30 min)

## 📝 NOTES

**THE CODEBASE IS ACTUALLY VERY CLEAN!**
- Most "debt" in the old document was already resolved
- The "11 duplicate files" never existed
- experiments/ is properly using src/ components
- Training is stable and progressing well

**Last Updated**: September 2, 2025
**Next Review**: After channel routing implementation
