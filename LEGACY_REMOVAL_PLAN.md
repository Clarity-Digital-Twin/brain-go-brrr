# Legacy Code Removal Plan

## Phase 1: Remove compat_coerce Logic (HIGHEST PRIORITY)
This is the biggest source of complexity and technical debt.

### Files to modify:
1. **src/brain_go_brrr/infra/ml_models/eegpt_compat.py**
   - Remove `compat_coerce` parameter from `__init__`
   - Remove all conditional logic based on `compat_coerce`
   - Enforce strict shape contracts only
   - Remove legacy single-sample batch removal
   - Remove support for 768-dimension features
   - Remove support for packed tokens (2048)

### Tests to update:
1. **tests/unit/test_models_eegpt_model.py**
   - Remove `compat_coerce=True` usage
   - Update DummyEncoder to return correct 512 dimensions

2. **tests/unit/test_eegpt_compat_strict.py**
   - Remove tests for compat_coerce behavior
   - Keep only strict validation tests

3. **tests/unit/test_eegpt_compat_coverage.py**
   - Remove deprecation warning tests
   - Remove compat_coerce edge case tests

## Phase 2: Remove Deprecated Modules
These are entire modules marked as deprecated for v2.0.0 removal.

### Modules to DELETE entirely:
1. **src/brain_go_brrr/models/** - Deprecated redirect to infra.ml_models
2. **src/brain_go_brrr/preprocessing/** - Deprecated redirect to domain/infra.preprocessing  
3. **src/brain_go_brrr/visualization/** - Deprecated redirect to presentation.visualization
4. **src/brain_go_brrr/services/yasa_adapter.py** - Moved to infra.external
5. **src/brain_go_brrr/utils/deprecated_redirect.py** - No longer needed

## Phase 3: Remove Backward Compatibility Aliases

### Files with aliases to clean:
1. **src/brain_go_brrr/services/__init__.py**
   - Remove `HierarchicalEEGAnalyzer` legacy alias

2. **src/brain_go_brrr/services/hierarchical_pipeline.py**
   - Remove `HierarchicalPipeline` backward compatible alias

3. **src/brain_go_brrr/infra/preprocessing/chunked_autoreject.py**
   - Remove compatibility alias

4. **src/brain_go_brrr/infra/preprocessing/autoreject_adapter.py**
   - Remove compatibility alias

5. **src/brain_go_brrr/infra/ml_models/eegpt_probe_unified.py**
   - Remove compatibility methods and aliases

6. **src/brain_go_brrr/domain/abnormal/__init__.py**
   - Remove `AbnormalityDetector` legacy export

7. **src/brain_go_brrr/domain/abnormal/detector.py**
   - Remove backward compatibility methods
   - Remove legacy parameters

## Phase 4: Clean Up EEGPTConfig Legacy Fields

### In src/brain_go_brrr/infra/ml_models/eegpt_compat.py:
- Remove `model_size`, `embed_dim`, `max_channels` from EEGPTConfig
- These are marked as "Legacy fields for test compatibility"

## Phase 5: Remove Unused/Dead Code

### Check and remove:
1. Any TODO comments about removing legacy code
2. Functions that are only used by deprecated code paths
3. Test fixtures that only support legacy behavior

## Testing Strategy

After EACH phase:
1. Run full test suite: `make test`
2. Run type checking: `make type-check`
3. Run linting: `make lint`
4. Run integration tests if available
5. Check test coverage doesn't drop

## Rollback Plan

1. Keep this branch separate from main/development
2. Create tags before each phase:
   - `pre-phase-1-compat-removal`
   - `pre-phase-2-module-removal`
   - etc.
3. If anything breaks unexpectedly, can cherry-pick working commits

## Expected Benefits

1. **Code simplification**: Remove ~100+ lines of conditional logic
2. **Better type safety**: No more Union types for multiple shapes
3. **Clearer API**: Single way to do things
4. **Easier maintenance**: Less code paths to test
5. **Performance**: No runtime shape coercion overhead

## Risk Assessment

- **LOW RISK**: Phase 1 (compat_coerce) - Only affects test compatibility
- **MEDIUM RISK**: Phase 2 (deprecated modules) - May break external imports
- **LOW RISK**: Phase 3 (aliases) - Simple renames
- **LOW RISK**: Phase 4 (legacy fields) - Unused fields
- **LOW RISK**: Phase 5 (dead code) - No functional impact

## Recommended Order

1. Start with Phase 1 (compat_coerce) - Biggest win, lowest risk
2. Then Phase 2 (deprecated modules) - Clean major structure
3. Then Phases 3-5 in any order - Minor cleanups