# 🚨 CRITICAL DIVE SAFETY REPORT

## Executive Summary
**GOOD NEWS:** The refactoring is mostly safe to proceed, but we found 2 critical mapping errors that need correction.
**BAD NEWS:** Cache files contain old references (but these can be safely deleted).

## ✅ VERIFIED SAFE

### 1. New EEGPT APIs Exist and Work
```python
✅ create_normalized_eegpt() exists at infra/ml_models/eegpt_wrapper.py:160
✅ EEGPTProbe class exists at infra/ml_models/eegpt_probe_unified.py:21
```

### 2. Core Redirects Are Correctly Mapped
All redirect files correctly point to existing destinations:
```python
✅ core.exceptions → domain.exceptions (BrainGoBrrrError, not BrainError)
✅ core.config → application.config
✅ core.channels → domain.channels
✅ core.edf_loader → infra.data.edf_loader
✅ core.edf_validator → infra.data.edf_validator (NOT domain.validation!)
✅ core.logger → infra.logger (NOT infra.logging!)
✅ core.features → domain.preprocessing.features
✅ core.window_extractor → domain.preprocessing.window_extractor
```

### 3. No Circular Dependencies Found
- All redirects are one-way (old → new)
- No new modules import from core
- Clean layered architecture maintained

### 4. Dynamic Imports Are Safe
Only found in:
- `utils/deprecated_redirect.py` (the redirect mechanism itself)
- `__init__.py` (module aliasing for backward compat)
- No string-based imports that would break

## ⚠️ CORRECTIONS NEEDED TO ORIGINAL PLAN

### 1. Fixed Module Mappings
```python
# WRONG in original plan:
core.edf_validator → domain.validation.edf_validator  ❌
core.logger → infra.logging.logger  ❌

# CORRECT mappings:
core.edf_validator → infra.data.edf_validator  ✅
core.logger → infra.logger  ✅
```

### 2. Cache Files to Clean
```bash
# MyPy caches contain old references - DELETE THESE:
rm -rf .mypy_cache/
rm -rf .mypy_cache_fast/
rm -rf .mypy_cache_strict/
rm -rf .import_linter_cache/
```

## 🔐 SAFETY CHECKLIST

### Pre-Flight Checks ✅
- [x] All new module destinations exist
- [x] New EEGPT APIs are available
- [x] No circular dependency risks
- [x] Redirect mappings are correct
- [x] Tests exist for affected modules
- [x] Backup branch can be created

### Phase 1 Safety (Critical Production)
```bash
# Before starting Phase 1, verify these files exist:
✅ src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py
✅ src/brain_go_brrr/infra/ml_models/eegpt_probe_unified.py
✅ Function: create_normalized_eegpt()
✅ Class: EEGPTProbe

# Test imports work:
uv run python -c "
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
print('Imports work!')
"
```

### Files That Are SAFE to Change (Phase 1)
1. **domain/quality/controller.py:112**
   ```python
   # OLD: from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
   # NEW: from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
   # OLD: model = EEGPTModel(eegpt_model_path)
   # NEW: model = create_normalized_eegpt(checkpoint_path=eegpt_model_path)
   ```

2. **infra/adapters/model_adapter.py:28**
   ```python
   # Same pattern as above
   ```

3. **application/use_cases/tasks/enhanced_abnormality_detection.py:91**
   ```python
   # OLD: from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
   # NEW: from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
   # OLD: probe = EEGPTTwoLayerProbe(n_classes=2, hidden_dim=512)
   # NEW: probe = EEGPTProbe(architecture='two_layer', n_classes=2, hidden_dim=512)
   ```

## 🎯 GO/NO-GO DECISION

### GO ✅ - Safe to Proceed Because:
1. All new APIs exist and are tested
2. Redirect mappings are verified correct
3. No circular dependencies
4. Tests exist for validation
5. Clear rollback path available

### Recommended Approach:
1. **START WITH PHASE 1 ONLY** (3 critical files)
2. Run full test suite after Phase 1
3. If tests pass, proceed to Phase 2
4. Take breaks between phases
5. Commit after each successful phase

## 📝 REVISED AUTOMATION SCRIPT

```bash
#!/bin/bash
# CORRECTED update_core_imports.sh

declare -A IMPORT_MAP=(
    ["brain_go_brrr.core.exceptions"]="brain_go_brrr.domain.exceptions"
    ["brain_go_brrr.core.config"]="brain_go_brrr.application.config"
    ["brain_go_brrr.core.channels"]="brain_go_brrr.domain.channels"
    ["brain_go_brrr.core.abnormality_config"]="brain_go_brrr.application.config.abnormality_config"
    ["brain_go_brrr.core.edf_loader"]="brain_go_brrr.infra.data.edf_loader"
    ["brain_go_brrr.core.edf_validator"]="brain_go_brrr.infra.data.edf_validator"  # FIXED!
    ["brain_go_brrr.core.features"]="brain_go_brrr.domain.preprocessing.features"
    ["brain_go_brrr.core.logger"]="brain_go_brrr.infra.logger"  # FIXED!
    ["brain_go_brrr.core.window_extractor"]="brain_go_brrr.domain.preprocessing.window_extractor"
    ["brain_go_brrr.core.quality"]="brain_go_brrr.domain.quality"
    ["brain_go_brrr.core.sleep"]="brain_go_brrr.domain.sleep"
    ["brain_go_brrr.core.abnormal"]="brain_go_brrr.domain.abnormal"
    ["brain_go_brrr.core.pipeline"]="brain_go_brrr.application.pipeline"
    ["brain_go_brrr.core.jobs"]="brain_go_brrr.application.jobs"
    ["brain_go_brrr.core.preprocessing"]="brain_go_brrr.domain.preprocessing"
)
```

## 🚦 FINAL RECOMMENDATION

**YES, PROCEED WITH PHASE 1** - The refactoring is safe with the corrections noted above.

The risk is manageable because:
1. We're only changing 3 files initially
2. All new APIs are verified to exist
3. We have comprehensive tests
4. Clear rollback strategy

**Start with Phase 1 NOW while the investigation is fresh in your mind.**

---

*Investigation Complete: Safe to proceed with corrections noted.*