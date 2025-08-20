# 🔧 COMPREHENSIVE REFACTORING FIX PLAN

## Executive Summary
**Total Files Requiring Changes:** 110+ files
- **38 files** with `core.*` imports
- **53 files** with deprecated EEGPT models
- **19 files** with redirect shims
- **Multiple overlapping issues** in same files

## 📊 DETAILED FINDINGS

### Search Results Summary
```
38 files - importing from brain_go_brrr.core.*
53 files - using deprecated EEGPT models (171 total occurrences)
19 files - contain deprecated_redirect shims
621 files - found in various core directories (includes test files)
```

## 🔴 SECTION 1: CORE IMPORT MIGRATIONS

### Files Using `brain_go_brrr.core.*` Imports (38 files)

#### Production Code (2 files)
```
src/brain_go_brrr/core/preprocessing.py
src/brain_go_brrr/core/preprocessing_utils.py
```
**Action:** These ARE the redirect files themselves - DELETE after migration

#### Test Files (27 files)
```
tests/unit/test_core_exceptions.py
tests/unit/test_config.py
tests/unit/test_config_defaults.py
tests/unit/test_channel_mapper.py
tests/unit/test_enhanced_sleep_analyzer.py
tests/unit/test_exceptions_hierarchy.py
tests/unit/test_improved_mocking.py
tests/unit/test_logger_singleton.py
tests/unit/test_models_eegpt_model.py
tests/unit/test_quality_controller.py
tests/unit/test_sleep_analysis.py
tests/unit/test_sleep_montage_detection.py
tests/unit/test_yasa_compliance.py
tests/unit/test_yasa_smoothing.py
tests/unit/test_api_routers_eegpt.py
tests/unit/test_coverage_boost_refactored.py
tests/smoke/test_imports.py
tests/integration/test_data_pipeline_integration.py
tests/integration/test_eegpt_integration.py
tests/integration/test_parallel_pipeline.py
tests/integration/test_sleep_enhanced.py
tests/benchmarks/test_eegpt_performance.py
tests/benchmarks/test_performance.py
tests/conftest.py
tests/fixtures/mock_eegpt.py
tests/api/test_api_endpoints.py
```

#### Scripts & Examples (9 files)
```
examples/end_to_end_pipeline.py
scripts/test_sleep_analysis.py
scripts/benchmark_end_to_end.py
scripts/archive/debugging/verify_all_fixes.py
scripts/archive/run_test_directly.py
scripts/archive/testing/test_eegpt_qc_integration.py
scripts/archive/testing/test_eegpt_real_inference.py
scripts/archive/testing/test_eegpt_updated.py
scripts/archive/testing/test_flexible_pipeline.py
scripts/archive/testing/test_full_pipeline.py
```

### Import Mapping Required
```python
# OLD → NEW MAPPINGS
brain_go_brrr.core.exceptions → brain_go_brrr.domain.exceptions
brain_go_brrr.core.config → brain_go_brrr.application.config
brain_go_brrr.core.channels → brain_go_brrr.domain.channels
brain_go_brrr.core.abnormality_config → brain_go_brrr.application.config.abnormality_config
brain_go_brrr.core.edf_loader → brain_go_brrr.infra.data.edf_loader
brain_go_brrr.core.edf_validator → brain_go_brrr.domain.validation.edf_validator
brain_go_brrr.core.features → brain_go_brrr.domain.preprocessing.features
brain_go_brrr.core.logger → brain_go_brrr.infra.logging.logger
brain_go_brrr.core.window_extractor → brain_go_brrr.domain.preprocessing.window_extractor
brain_go_brrr.core.quality → brain_go_brrr.domain.quality
brain_go_brrr.core.sleep → brain_go_brrr.domain.sleep
brain_go_brrr.core.abnormal → brain_go_brrr.domain.abnormal
brain_go_brrr.core.pipeline → brain_go_brrr.application.pipeline
brain_go_brrr.core.jobs → brain_go_brrr.application.jobs
brain_go_brrr.core.preprocessing → brain_go_brrr.domain.preprocessing
```

## 🔴 SECTION 2: EEGPT MODEL DEPRECATIONS

### Critical Production Files Using Deprecated Models (19 files)

#### HIGH PRIORITY - API/Domain Layer (3 files)
```python
# src/brain_go_brrr/domain/quality/controller.py - Lines 109, 112
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
model = EEGPTModel(eegpt_model_path)
# FIX: Replace with
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
model = create_normalized_eegpt(eegpt_model_path)

# src/brain_go_brrr/infra/adapters/model_adapter.py - Lines 13, 28
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
self.model = EEGPTModel(checkpoint_path=model_path, device=device)
# FIX: Replace with
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
self.model = create_normalized_eegpt(checkpoint_path=model_path, device=device)

# src/brain_go_brrr/application/use_cases/tasks/enhanced_abnormality_detection.py - Lines 28, 91
from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
probe = EEGPTTwoLayerProbe(...)
# FIX: Replace with
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
probe = EEGPTProbe(architecture='two_layer', ...)
```

#### MEDIUM PRIORITY - Application Layer (10 files)
```
src/brain_go_brrr/api/routers/eegpt.py
src/brain_go_brrr/api/routers/sleep.py
src/brain_go_brrr/application/factories.py
src/brain_go_brrr/application/factories/detector.py
src/brain_go_brrr/application/factories/extractor.py
src/brain_go_brrr/application/factories/qc.py
src/brain_go_brrr/application/pipeline/eegpt_orchestration.py
src/brain_go_brrr/application/training/sleep_probe_trainer.py
src/brain_go_brrr/application/use_cases/tasks/abnormality_detection.py
src/brain_go_brrr/cli.py
```

#### LOW PRIORITY - Infrastructure Layer (6 files)
```
src/brain_go_brrr/infra/adapters/eegpt_feature_extractor.py
src/brain_go_brrr/infra/ml_models/__init__.py
src/brain_go_brrr/infra/ml_models/eegpt_linear_probe.py (DEPRECATED FILE - DELETE)
src/brain_go_brrr/infra/ml_models/eegpt_model.py (DEPRECATED FILE - DELETE)
src/brain_go_brrr/infra/ml_models/eegpt_probe_unified.py (NEW FILE - KEEP)
src/brain_go_brrr/infra/ml_models/eegpt_two_layer_probe.py (DEPRECATED FILE - DELETE)
```

### Test Files Using Deprecated Models (34 files)
```
tests/_mocks.py
tests/benchmarks/test_eegpt_performance.py
tests/benchmarks/test_performance.py
tests/fixtures/mock_eegpt.py
tests/integration/test_eegpt_integration.py
tests/integration/test_eegpt_streaming_integration.py
tests/unit/test_abnormality_accuracy.py
tests/unit/test_api_routers_eegpt.py
tests/unit/test_classifier_compatibility.py
tests/unit/test_coverage_boost_refactored.py
tests/unit/test_eegpt_extreme_discrimination.py
tests/unit/test_eegpt_linear_probe.py
tests/unit/test_eegpt_model_loading.py
tests/unit/test_eegpt_pipeline.py
tests/unit/test_eegpt_summary_tokens.py
tests/unit/test_encoder_raw_output.py
tests/unit/test_improved_mocking.py
tests/unit/test_models_eegpt_model.py
tests/unit/test_models_eegpt_wrapper.py
tests/unit/test_quality_controller.py
tests/unit/test_robust_eegpt_probe.py
```

### EEGPT Model Migration Pattern
```python
# OLD PATTERNS → NEW PATTERNS

# Pattern 1: EEGPTModel
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
model = EEGPTModel(checkpoint_path="path.ckpt")
# REPLACE WITH:
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
model = create_normalized_eegpt(checkpoint_path="path.ckpt")

# Pattern 2: EEGPTLinearProbe
from brain_go_brrr.infra.ml_models.eegpt_linear_probe import EEGPTLinearProbe
probe = EEGPTLinearProbe(n_classes=2)
# REPLACE WITH:
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
probe = EEGPTProbe(architecture='linear', n_classes=2)

# Pattern 3: EEGPTTwoLayerProbe
from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
probe = EEGPTTwoLayerProbe(n_classes=2, hidden_dim=512)
# REPLACE WITH:
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
probe = EEGPTProbe(architecture='two_layer', n_classes=2, hidden_dim=512)
```

## 🔴 SECTION 3: REDIRECT/SHIM FILES TO DELETE

### Core Redirect Files (15 files)
```
src/brain_go_brrr/core/__init__.py
src/brain_go_brrr/core/abnormal/__init__.py
src/brain_go_brrr/core/abnormality_config.py
src/brain_go_brrr/core/cache_port.py
src/brain_go_brrr/core/channels.py
src/brain_go_brrr/core/config.py
src/brain_go_brrr/core/edf_loader.py
src/brain_go_brrr/core/edf_validator.py
src/brain_go_brrr/core/exceptions.py
src/brain_go_brrr/core/features.py
src/brain_go_brrr/core/jobs/__init__.py
src/brain_go_brrr/core/logger.py
src/brain_go_brrr/core/pipeline/__init__.py
src/brain_go_brrr/core/preprocessing.py
src/brain_go_brrr/core/preprocessing_utils.py
src/brain_go_brrr/core/quality/__init__.py
src/brain_go_brrr/core/sleep/__init__.py
src/brain_go_brrr/core/window_extractor.py
```

### Other Redirect Files (4 files)
```
src/brain_go_brrr/config/__init__.py (redirect to application.config)
src/brain_go_brrr/models/__init__.py (redirect to infra.ml_models)
src/brain_go_brrr/services/__init__.py (redirect to application.services)
src/brain_go_brrr/services/yasa_adapter.py (redirect to infra.external.yasa_adapter)
src/brain_go_brrr/visualization/__init__.py (redirect to presentation.visualization)
```

## 📋 EXECUTION PLAN

### Phase 1: Fix Critical Production Code (IMMEDIATE)
**Files: 3** | **Risk: HIGH** | **Time: 1 hour**

1. `domain/quality/controller.py` - Replace EEGPTModel
2. `infra/adapters/model_adapter.py` - Replace EEGPTModel
3. `application/use_cases/tasks/enhanced_abnormality_detection.py` - Replace EEGPTTwoLayerProbe

**Test Command:**
```bash
pytest tests/unit/test_quality_controller.py tests/unit/test_models_eegpt_wrapper.py -xvs
```

### Phase 2: Fix Application Layer (TODAY)
**Files: 10** | **Risk: MEDIUM** | **Time: 2 hours**

Update all factories and routers to use new EEGPT APIs.

**Test Command:**
```bash
pytest tests/api/ tests/unit/test_api_routers_eegpt.py -xvs
```

### Phase 3: Update Test Suite (TODAY)
**Files: 61** | **Risk: LOW** | **Time: 3 hours**

Update all test files to use new imports and APIs.

**Test Command:**
```bash
pytest tests/ -xvs --tb=short
```

### Phase 4: Remove Redirect Files (TOMORROW)
**Files: 19** | **Risk: MEDIUM** | **Time: 1 hour**

Delete all redirect shim files after confirming no usage.

**Verification Command:**
```bash
grep -r "from brain_go_brrr\.core" --include="*.py" . | wc -l  # Should be 0
```

### Phase 5: Delete Deprecated Models (TOMORROW)
**Files: 3** | **Risk: LOW** | **Time: 30 min**

Delete deprecated model files:
- `eegpt_model.py`
- `eegpt_linear_probe.py`
- `eegpt_two_layer_probe.py`

### Phase 6: Clean Up Archives (OPTIONAL)
**Files: 20+** | **Risk: ZERO** | **Time: 15 min**

Archive scripts can be ignored or bulk updated.

## 🛠️ AUTOMATION SCRIPTS

### Script 1: Update Core Imports
```bash
#!/bin/bash
# update_core_imports.sh

# Map old to new
declare -A IMPORT_MAP=(
    ["brain_go_brrr.core.exceptions"]="brain_go_brrr.domain.exceptions"
    ["brain_go_brrr.core.config"]="brain_go_brrr.application.config"
    ["brain_go_brrr.core.channels"]="brain_go_brrr.domain.channels"
    # ... add all mappings
)

for old in "${!IMPORT_MAP[@]}"; do
    new="${IMPORT_MAP[$old]}"
    find . -name "*.py" -type f -exec sed -i "s|$old|$new|g" {} \;
done
```

### Script 2: Update EEGPT Models
```bash
#!/bin/bash
# update_eegpt_models.sh

# Replace EEGPTModel
find . -name "*.py" -exec sed -i \
    -e 's/from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel/from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt/g' \
    -e 's/EEGPTModel(/create_normalized_eegpt(/g' {} \;

# Replace EEGPTLinearProbe
find . -name "*.py" -exec sed -i \
    -e 's/from brain_go_brrr.infra.ml_models.eegpt_linear_probe import EEGPTLinearProbe/from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe/g' \
    -e 's/EEGPTLinearProbe(/EEGPTProbe(architecture="linear", /g' {} \;

# Replace EEGPTTwoLayerProbe
find . -name "*.py" -exec sed -i \
    -e 's/from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe/from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe/g' \
    -e 's/EEGPTTwoLayerProbe(/EEGPTProbe(architecture="two_layer", /g' {} \;
```

## 📊 SUCCESS METRICS

### Before Cleanup
- ❌ 38 files with core imports
- ❌ 171 deprecated EEGPT usages
- ❌ 31 deprecation warnings
- ❌ 19 redirect files
- ❌ 3 deprecated model files

### After Cleanup Target
- ✅ 0 files with core imports
- ✅ 0 deprecated EEGPT usages
- ✅ 0 deprecation warnings
- ✅ 0 redirect files
- ✅ 0 deprecated model files
- ✅ Single unified EEGPT API
- ✅ Clean architecture maintained
- ✅ All tests passing
- ✅ Coverage >65%

## ⚠️ RISK MITIGATION

### Backup Strategy
```bash
# Create backup branch before changes
git checkout -b backup/pre-cleanup-$(date +%Y%m%d)
git push origin backup/pre-cleanup-$(date +%Y%m%d)
```

### Rollback Plan
```bash
# If something breaks
git reset --hard HEAD
git checkout development
```

### Testing Strategy
1. Run tests after EACH file change
2. Commit after each successful phase
3. Push to remote after each phase completion
4. Monitor CI/CD pipeline

## 📝 MANUAL REVIEW REQUIRED

These files need manual inspection due to complex usage:

1. **src/brain_go_brrr/cli.py** - May have complex initialization
2. **src/brain_go_brrr/application/training/sleep_probe_trainer.py** - Training logic
3. **src/brain_go_brrr/application/pipeline/eegpt_orchestration.py** - Pipeline orchestration
4. **tests/fixtures/mock_eegpt.py** - Mock objects may need restructuring
5. **tests/_mocks.py** - Central mock definitions

## 🏁 COMPLETION CHECKLIST

### Phase 1 Complete ☐
- [ ] Fixed domain/quality/controller.py
- [ ] Fixed infra/adapters/model_adapter.py
- [ ] Fixed enhanced_abnormality_detection.py
- [ ] All tests passing
- [ ] Committed changes

### Phase 2 Complete ☐
- [ ] Updated all factories
- [ ] Updated all routers
- [ ] Updated CLI
- [ ] All tests passing
- [ ] Committed changes

### Phase 3 Complete ☐
- [ ] Updated all test files
- [ ] No deprecation warnings
- [ ] Coverage maintained >65%
- [ ] Committed changes

### Phase 4 Complete ☐
- [ ] Deleted all core redirect files
- [ ] No broken imports
- [ ] All tests passing
- [ ] Committed changes

### Phase 5 Complete ☐
- [ ] Deleted deprecated model files
- [ ] Updated __init__ exports
- [ ] All tests passing
- [ ] Committed changes

### Final Verification ☐
- [ ] Zero deprecation warnings
- [ ] Zero core imports
- [ ] Single EEGPT API pattern
- [ ] Documentation updated
- [ ] PR created and reviewed

---

**Total Estimated Time:** 7-8 hours
**Recommended Execution:** Over 2 days to minimize risk
**Priority:** Fix production code FIRST (Phase 1)

*This plan ensures systematic elimination of ALL technical debt from both refactorings.*
