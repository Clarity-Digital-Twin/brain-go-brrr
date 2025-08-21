# 🔍 REFACTORING INVESTIGATION REPORT

## Executive Summary
The codebase underwent TWO major refactorings:
1. **Clean Architecture Refactoring** (Aug 13-15): Domain-Driven Design with backward compatibility shims
2. **EEGPT Model Refactoring** (Aug 16-19): Unified probe architecture with deprecation warnings

Both refactorings created technical debt through compatibility layers that now need cleanup.

## 📊 Current State Analysis

### Deprecation Statistics
- **31 deprecation warnings** in test runs
- **49 non-deprecated usages** of old EEGPT classes in production code
- **42 redirect modules** in `core/` providing backward compatibility
- **19 files** still using old EEGPT models

### Key Patterns Found

#### 1. Clean Architecture Shims (Phase 1)
All modules in `src/brain_go_brrr/core/` are now just redirects:
```python
# Every file in core/ follows this pattern:
from brain_go_brrr.utils.deprecated_redirect import redirect
redirect("brain_go_brrr.core.X", "brain_go_brrr.domain.X", globals())
```

**Affected modules:**
- `core/exceptions.py` → `domain/exceptions.py`
- `core/config.py` → `application/config/`
- `core/channels.py` → `domain/channels.py`
- `core/abnormality_config.py` → `application/config/abnormality_config.py`
- `core/edf_loader.py` → `infra/data/edf_loader.py`
- `core/edf_validator.py` → `domain/validation/edf_validator.py`
- `core/features.py` → `domain/preprocessing/features/`
- `core/logger.py` → `infra/logging/logger.py`
- `core/window_extractor.py` → `domain/preprocessing/window_extractor.py`
- `core/quality/` → `domain/quality/`
- `core/sleep/` → `domain/sleep/`
- `core/abnormal/` → `domain/abnormal/`
- `core/pipeline/` → `application/pipeline/`
- `core/jobs/` → `application/jobs/`

#### 2. EEGPT Model Deprecations (Phase 2)
Old classes with deprecation warnings:
```python
# Old (deprecated)
EEGPTModel → create_normalized_eegpt()
EEGPTLinearProbe → EEGPTProbe(architecture='linear')
EEGPTTwoLayerProbe → EEGPTProbe(architecture='two_layer')
```

**Files still using old models:**
1. `domain/quality/controller.py:112` - Uses EEGPTModel
2. `infra/adapters/model_adapter.py:28` - Uses EEGPTModel
3. `application/use_cases/tasks/enhanced_abnormality_detection.py:91` - Uses EEGPTTwoLayerProbe
4. Tests throughout - Multiple old model usages

## 🔴 Critical Issues to Fix

### Priority 1: Production Code Using Deprecated APIs
These files use deprecated models in production:
1. **domain/quality/controller.py** - Critical, used by API
2. **infra/adapters/model_adapter.py** - Critical, adapter layer
3. **application/use_cases/tasks/enhanced_abnormality_detection.py** - Important use case

### Priority 2: Test Suite Deprecations
Tests generating warnings:
- `test_eegpt_pipeline.py` (8 warnings)
- `test_models_eegpt_model.py` (2 warnings)
- `test_coverage_boost_refactored.py` (2 warnings)
- `test_encoder_raw_output.py` (1 warning)
- Smoke tests (5 warnings via model_adapter.py)

### Priority 3: Redundant Redirect Modules
The entire `core/` directory is now just redirects and can be removed after updating imports.

## 📁 File Structure Issues

### Duplicate/Redundant Files
```
src/brain_go_brrr/infra/ml_models/
├── eegpt_model.py (deprecated, 293 lines)
├── eegpt_linear_probe.py (deprecated, 63 lines)
├── eegpt_two_layer_probe.py (deprecated, 90 lines)
├── eegpt_probe_unified.py (NEW unified, 98 lines)
├── eegpt_wrapper.py (current wrapper, 63 lines)
└── eegpt_architecture.py (core architecture, 196 lines)
```

### Cache Implementation Duplication
```
├── core/cache_port.py (deprecated redirect)
├── domain/ports/cache.py (interface)
├── infra/cache.py (implementation)
├── infra/cache_factory.py (factory)
└── api/cache.py (API layer)
```

## 🎯 Cleanup Action Plan

### Phase 1: Update Production Code (CRITICAL)
1. Replace `EEGPTModel` with `create_normalized_eegpt()` in:
   - `domain/quality/controller.py`
   - `infra/adapters/model_adapter.py`

2. Replace `EEGPTTwoLayerProbe` with `EEGPTProbe(architecture='two_layer')` in:
   - `application/use_cases/tasks/enhanced_abnormality_detection.py`

### Phase 2: Update Test Suite
1. Update all test files to use new APIs
2. Remove test files for deprecated models
3. Consolidate coverage boost tests

### Phase 3: Remove Backward Compatibility
1. Delete entire `core/` directory (all redirects)
2. Update all imports from `core.X` to proper locations
3. Remove `utils/deprecated_redirect.py` (no longer needed)

### Phase 4: Remove Deprecated Models
1. Delete `infra/ml_models/eegpt_model.py`
2. Delete `infra/ml_models/eegpt_linear_probe.py`
3. Delete `infra/ml_models/eegpt_two_layer_probe.py`
4. Update `__init__.py` exports

### Phase 5: Consolidate Cache Implementation
1. Unify cache interfaces
2. Remove duplicate cache_port files
3. Single cache control pattern

## 📈 Expected Outcomes

### Before Cleanup
- 31 deprecation warnings
- 49 deprecated usages
- 42 redirect modules
- ~800 lines of compatibility code

### After Cleanup
- 0 deprecation warnings
- 0 deprecated usages
- 0 redirect modules
- Clean, single way to use each component

## 🚨 Risk Assessment

### Low Risk
- Test updates (can't break production)
- Documentation updates

### Medium Risk
- Production code updates (well-tested paths)
- Import path changes (automated find/replace)

### High Risk
- None identified (all changes are to deprecated code)

## ✅ Success Criteria
1. Zero deprecation warnings in test suite
2. No `core/` directory (all redirects removed)
3. Single implementation for each component
4. All tests still passing
5. Coverage maintained >65%

## 🔧 Tools for Cleanup
```bash
# Find all imports from core
grep -r "from brain_go_brrr\.core" --include="*.py" .

# Find all deprecated model usages
grep -r "EEGPTModel\|EEGPTLinearProbe\|EEGPTTwoLayerProbe" --include="*.py" .

# Run tests with deprecation warnings as errors
python -W error::DeprecationWarning -m pytest

# Check for circular imports
import-linter --config pyproject.toml
```

## 📝 Next Steps
1. **Commit this investigation** for reference
2. **Start with Phase 1** - Fix production code (highest priority)
3. **Test after each phase** to ensure nothing breaks
4. **Document changes** in commit messages

---

*Investigation Complete: Ready to systematically eliminate all technical debt from refactoring.*
