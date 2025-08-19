# Deprecation & Refactoring Cleanup Plan

## Overview
After major refactoring of EEGPT embeddings, TUAB/TUEV training, and DI cache control, we need to clean up technical debt.

## Deprecation Warnings to Address (31 total)

### 1. EEGPTModel Deprecations (High Priority)
**Files using deprecated `EEGPTModel`:**
- `tests/unit/test_eegpt_pipeline.py` (8 occurrences)
- `tests/unit/test_encoder_raw_output.py` (1 occurrence)
- `tests/unit/test_models_eegpt_model.py` (2 occurrences)
- `src/brain_go_brrr/domain/quality/controller.py:112` (1 occurrence)
- `src/brain_go_brrr/infra/adapters/model_adapter.py:28` (5 occurrences in smoke tests)

**Action:** Replace with `create_normalized_eegpt()` from `brain_go_brrr.infra.ml_models.eegpt_wrapper`

### 2. Probe Model Deprecations (Medium Priority)
**Files using deprecated probes:**
- `EEGPTLinearProbe` → `EEGPTProbe` 
  - `tests/unit/test_coverage_boost_refactored.py:137`
- `EEGPTTwoLayerProbe` → `EEGPTProbe(architecture='two_layer')`
  - `tests/unit/test_coverage_boost_refactored.py:341`
  - `src/brain_go_brrr/application/use_cases/tasks/enhanced_abnormality_detection.py:91` (10 warnings)

### 3. Redundant/Dead Code to Remove
- `src/brain_go_brrr/infra/ml_models/eegpt_linear_probe_robust.py` (deleted but check references)
- `src/brain_go_brrr/infra/cache_port.py` (already deleted)
- Old experiment files in `experiments/eegpt_linear_probe/archive/`

## Code Duplication to Consolidate

### EEGPT Architecture Files
- `eegpt_model.py` (deprecated)
- `eegpt_wrapper.py` (current)
- `eegpt_architecture.py` (core)
- `eegpt_probe_unified.py` (new unified probe)

### Cache Implementation
- Multiple cache interfaces that could be unified
- Settings vs deps cache control

## Testing Improvements

### Coverage Gaps (Currently 65.25%)
**Low coverage modules to improve:**
- `eegpt_preprocessing.py` (8.40%)
- `eegpt_orchestration.py` (14.67%)
- `hierarchical_pipeline.py` (27.89%)

### Test Organization
- Remove redundant test files
- Consolidate coverage boost tests
- Update deprecated model usage in tests

## Experiments Cleanup

### TUAB/TUEV Training Scripts
**Keep (working):**
- `train_tuab.py`
- `train_tuev.py`
- `tuab_dataset.py`
- `tuev_dataset.py`
- `configs/tuab.yaml`
- `configs/tuev.yaml`

**Archive/Remove:**
- Old bulletproof versions
- WSL-safe variants
- Paper-aligned naming (now just tuab/tuev)

## Priority Order

1. **Phase 1: Update deprecated model usage** (prevents future breaks)
   - Update tests to use new APIs
   - Update production code in domain/quality/controller.py
   - Update model_adapter.py

2. **Phase 2: Remove dead code** (reduces confusion)
   - Clean experiments/archive
   - Remove deprecated model files (after updating references)
   - Remove redundant cache implementations

3. **Phase 3: Consolidate & simplify** (improves maintainability)
   - Unify probe implementations
   - Consolidate preprocessing pipelines
   - Simplify cache control to single pattern

4. **Phase 4: Documentation** (ensures knowledge transfer)
   - Update CLAUDE.md with new patterns
   - Document the unified EEGPT API
   - Create migration guide for deprecated classes

## Success Metrics
- ✅ 0 deprecation warnings
- ✅ Coverage maintained >65%
- ✅ All tests green
- ✅ Reduced file count in experiments/
- ✅ Clear, single way to use EEGPT models

## Next Steps
1. Start with Phase 1: Update deprecated model usage
2. Run tests after each change
3. Commit frequently with clear messages
4. Create PR when Phase 1-2 complete