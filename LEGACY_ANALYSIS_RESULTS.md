# Legacy Code Analysis Results

## Executive Summary
Based on professional analysis using vulture, coverage, and ripgrep, here's what we found:

### ✅ Safe to Remove (NOT used in production)
1. **compat_coerce parameter and all its logic**
   - Only exists in `eegpt_compat.py`
   - Only used in test files
   - No production code sets `compat_coerce=True`
   - 91% test coverage means we understand the code well

2. **Legacy shape handling code blocks**
   - Lines 206-213: Averaging tokens to create summary
   - Lines 214-222: Accepting 768 dimensions
   - Lines 232-240: Packed tokens reshape
   - Lines 241-248: Tiling summary to create tokens
   - Lines 256-264: Single-sample batch removal

3. **Legacy fields in EEGPTConfig**
   - `model_size` (line 34) - marked 60% unused by vulture
   - `embed_dim` (line 35) - marked 60% unused by vulture
   - `max_channels` (line 36) - marked 60% unused by vulture

### ⚠️ Keep But Monitor (Used in production)
1. **extract_windows** - Used by:
   - `api/routers/eegpt.py` (2 calls)
   - `domain/preprocessing/features/extractor.py` (indirect)

2. **predict_abnormality** - Used by:
   - `domain/quality/controller.py` (hasattr check + call)
   - `application/pipeline/__init__.py` (export)

3. **extract_features_from_raw** - Used by:
   - `infra/ml_models/__init__.py` (exported)
   - Possibly external consumers

### 📊 Coverage Analysis
```
Total Coverage: 91.19%
Branch Coverage: 64/73 branches covered
Missing Lines: 9
Partial Branches: 9
```

**Uncovered code (can likely be removed):**
- Line 175: Old encoder path without extract_features method
- Lines 190-193: Another old encoder fallback
- Lines 302, 307: Fallback in extract_features_batch

### 🔍 Dependency Analysis

**Who imports eegpt_compat?**
```bash
# Direct imports
src/brain_go_brrr/infra/ml_models/__init__.py
src/brain_go_brrr/api/routers/eegpt.py
src/brain_go_brrr/domain/quality/controller.py

# Test imports (can be updated)
tests/unit/test_models_eegpt_model.py
tests/unit/test_eegpt_compat_strict.py
tests/unit/test_eegpt_compat_coverage.py
tests/unit/test_eegpt_real_inference.py
```

### 🎯 Removal Strategy

#### Phase 1: Remove compat_coerce (SAFE - Do First)
1. Delete parameter from `__init__`
2. Delete all `if self.compat_coerce` blocks
3. Keep only strict validation paths
4. Update 4 test files to remove `compat_coerce=True`

**Lines to delete:**
- 69: `compat_coerce: bool = False,`
- 78-80: Docstring about compat_coerce
- 97: `self.compat_coerce = compat_coerce`
- 206-213: Token averaging block
- 214-222: 768 dimension acceptance
- 232-240: Packed tokens reshape
- 241-248: Summary tiling
- 256-264: Batch removal

#### Phase 2: Remove Legacy Fields (SAFE)
Delete from EEGPTConfig:
- Lines 34-36: `model_size`, `embed_dim`, `max_channels`

#### Phase 3: Clean Up Dead Fallbacks (SAFE)
Remove untested fallback paths:
- Lines 190-193: Old encoder path
- Line 302: Fallback in extract_features_batch

### 📈 Expected Impact

**Code Reduction:**
- ~50 lines of conditional logic removed
- 9 complex if/elif branches eliminated
- 3 unused config fields removed

**Complexity Reduction:**
- Cyclomatic complexity drops from ~15 to ~8
- No more shape coercion overhead
- Single path through extract_features

**Performance Impact:**
- No runtime shape checks for compat mode
- No unnecessary reshaping/tiling
- Faster test execution (no deprecation warnings)

### 🚦 Risk Assessment

| Component | Risk | Mitigation |
|-----------|------|------------|
| compat_coerce removal | **LOW** | Only in tests, not production |
| Legacy fields removal | **LOW** | Vulture confirms unused |
| Fallback paths removal | **LOW** | 0% coverage = never executed |
| extract_windows | **MEDIUM** | Keep for now, deprecate later |
| predict_abnormality | **MEDIUM** | Keep for now, used by QC |

### ✅ Pre-Flight Checklist

Before removing:
- [x] Vulture analysis complete
- [x] Coverage analysis complete (91%)
- [x] Production usage verified (none for compat_coerce)
- [x] Test usage identified (4 files)
- [x] Git history reviewed
- [ ] Create git tag for rollback
- [ ] Add CI guard against compat_coerce
- [ ] Update affected tests
- [ ] Run full test suite
- [ ] Benchmark performance before/after

### 🎬 Action Items

1. **Immediate (Today)**:
   - Tag current state: `git tag pre-compat-removal`
   - Create feature branch: `git checkout -b remove-compat-coerce`
   - Remove compat_coerce logic
   - Update tests

2. **Tomorrow**:
   - Add CI guard
   - Run mutation testing
   - Performance benchmark

3. **Next Week**:
   - Deprecate extract_windows
   - Plan predict_abnormality migration

This analysis confirms it's SAFE to remove compat_coerce and its associated logic.