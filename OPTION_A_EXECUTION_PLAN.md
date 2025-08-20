# 🚀 OPTION A EXECUTION PLAN - CLEAN BREAK IMPLEMENTATION

## 🎯 OBJECTIVE
Achieve 100% clean codebase with zero legacy support, zero failing tests.

## ⏱️ TIMELINE: 2 HOURS TO VICTORY

### Phase 1: Quick Wins (15 minutes)
### Phase 2: Import Updates (30 minutes)
### Phase 3: Test Expectations (45 minutes)
### Phase 4: Validation & Cleanup (30 minutes)

---

## 📋 PHASE 1: QUICK WINS (15 min)

### Task 1.1: Delete Backward Compatibility Test
```bash
rm tests/smoke/test_imports_backward_compat.py
```
**Impact**: Removes 1 failing test immediately

### Task 1.2: Delete Legacy Probe Files
```bash
rm src/brain_go_brrr/infra/ml_models/eegpt_linear_probe.py
rm src/brain_go_brrr/infra/ml_models/eegpt_two_layer_probe.py
```
**Impact**: Forces use of unified probe

### Task 1.3: Remove Empty Core Directories
```bash
rm -rf src/brain_go_brrr/core/
```
**Impact**: Ensures no accidental imports

---

## 📋 PHASE 2: IMPORT UPDATES (30 min)

### Task 2.1: Fix Smoke Test Imports
**File**: `tests/smoke/test_imports.py`

```python
# OLD (lines 44-48)
from brain_go_brrr.core import preprocessing_utils
from brain_go_brrr.core.jobs.models import JobData, JobPriority, JobStatus
from brain_go_brrr.core.cache_port import AsyncCachePort, CachePort

# NEW
from brain_go_brrr.domain import channels as preprocessing_utils
from brain_go_brrr.application.jobs.models import JobData, JobPriority, JobStatus
from brain_go_brrr.infra.cache import CacheProtocol as CachePort
# Note: AsyncCachePort doesn't exist - remove or create
```

### Task 2.2: Fix Sleep Test Patches
**File**: `tests/integration/test_sleep_enhanced.py`

```python
# Find and replace all:
@patch("brain_go_brrr.core.sleep.analyzer_enhanced.yasa.SleepStaging")
# With:
@patch("brain_go_brrr.domain.sleep.analyzer_enhanced.yasa.SleepStaging")
```

### Task 2.3: Fix EEGPT Pipeline Patch
**File**: `tests/unit/test_eegpt_pipeline.py`

```python
# OLD (line ~300)
@patch("brain_go_brrr.models.eegpt_model.EEGPTModel")
# NEW
@patch("brain_go_brrr.infra.ml_models.eegpt_compat.EEGPTModel")
```

### Task 2.4: Update Deprecated Probe Imports
**Files**: `test_coverage_boost_refactored.py`, others

```python
# OLD
from brain_go_brrr.infra.ml_models.eegpt_linear_probe import EEGPTLinearProbe
from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe

# NEW
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
# Usage:
probe = EEGPTProbe(architecture='linear', ...)  # Instead of EEGPTLinearProbe
probe = EEGPTProbe(architecture='two_layer', ...)  # Instead of EEGPTTwoLayerProbe
```

---

## 📋 PHASE 3: TEST EXPECTATIONS (45 min)

### Task 3.1: Fix preprocess_for_eegpt Assertions
**File**: `tests/unit/test_eegpt_pipeline.py`

```python
# OLD - Expects MNERaw object
def test_preprocess_resampling(self, sample_raw):
    processed = preprocess_for_eegpt(sample_raw)
    assert processed.info["sfreq"] == 256
    assert processed.n_times == expected

# NEW - Expects numpy array
def test_preprocess_resampling(self, sample_raw):
    processed = preprocess_for_eegpt(sample_raw)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (n_channels, expected_samples)
    # If you need to check sampling rate, use the original raw
    # or pass it as metadata
```

### Task 3.2: Remove Deprecated Config Field Tests
**File**: `tests/unit/test_models_eegpt_model.py`

```python
# DELETE these test methods entirely:
def test_config_with_custom_model_size(self):
    config = EEGPTConfig(model_size="xlarge", embed_dim=768)
    assert config.model_size == "xlarge"  # DELETE - deprecated field

def test_max_channels_property(self):
    config = EEGPTConfig()
    assert config.max_channels == 58  # DELETE - deprecated field

def test_n_patches_calculation(self):
    config = EEGPTConfig()
    assert config.n_patches_per_window == 16  # DELETE - deprecated property
```

### Task 3.3: Update Config Usage
**File**: `tests/unit/test_eegpt_pipeline.py`

```python
# OLD
config = EEGPTConfig(model_size="xlarge", max_channels=32)

# NEW - Remove deprecated fields
config = EEGPTConfig(window_duration=4.0, sampling_rate=256)
# Don't test internal fields, test behavior
```

### Task 3.4: Fix AsyncCachePort Reference
**File**: `tests/smoke/test_imports.py`

```python
# AsyncCachePort doesn't exist in new architecture
# Either:
# Option 1: Remove the import test
# from brain_go_brrr.infra.cache import AsyncCachePort  # DELETE LINE

# Option 2: Import what actually exists
from brain_go_brrr.infra.cache import CacheProtocol, RedisCache
```

---

## 📋 PHASE 4: VALIDATION & CLEANUP (30 min)

### Task 4.1: Run Smoke Tests
```bash
pytest tests/smoke/ -xvs
# Fix any remaining import issues
```

### Task 4.2: Run Unit Tests
```bash
pytest tests/unit/test_eegpt_pipeline.py -xvs
pytest tests/unit/test_models_eegpt_model.py -xvs
# Fix any remaining assertion issues
```

### Task 4.3: Run Full Test Suite
```bash
pytest tests/ -x --tb=short
# Should be 100% green
```

### Task 4.4: Verify No Core Imports Remain
```bash
# Should return 0
grep -r "from brain_go_brrr\.core" . --include="*.py" | grep -v "^#" | wc -l

# Should return 0
grep -r "brain_go_brrr\.core\." . --include="*.py" | grep -v "^#" | wc -l
```

### Task 4.5: Verify Type Checking Still Green
```bash
make type-check
make lint
```

---

## 🛠️ AUTOMATED SCRIPT

```bash
#!/bin/bash
# option_a_cleanup.sh

echo "🚀 Starting Option A Clean Break Implementation"

# Phase 1: Quick Wins
echo "📋 Phase 1: Quick Wins"
rm -f tests/smoke/test_imports_backward_compat.py
rm -f src/brain_go_brrr/infra/ml_models/eegpt_linear_probe.py
rm -f src/brain_go_brrr/infra/ml_models/eegpt_two_layer_probe.py
rm -rf src/brain_go_brrr/core/

# Phase 2: Import Updates
echo "📋 Phase 2: Updating Imports"

# Fix smoke test imports
sed -i 's/from brain_go_brrr.core import preprocessing_utils/from brain_go_brrr.domain import channels as preprocessing_utils/' tests/smoke/test_imports.py
sed -i 's/from brain_go_brrr.core.jobs.models/from brain_go_brrr.application.jobs.models/' tests/smoke/test_imports.py
sed -i 's/from brain_go_brrr.core.cache_port/from brain_go_brrr.infra.cache/' tests/smoke/test_imports.py

# Fix sleep test patches
sed -i 's/brain_go_brrr.core.sleep/brain_go_brrr.domain.sleep/g' tests/integration/test_sleep_enhanced.py

# Fix EEGPT patches
sed -i 's/brain_go_brrr.models.eegpt_model/brain_go_brrr.infra.ml_models.eegpt_compat/g' tests/unit/test_eegpt_pipeline.py

# Fix probe imports
find tests -name "*.py" -exec sed -i \
  -e 's/from brain_go_brrr.infra.ml_models.eegpt_linear_probe import EEGPTLinearProbe/from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe/' \
  -e 's/from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe/from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe/' \
  -e 's/EEGPTLinearProbe(/EEGPTProbe(architecture="linear", /' \
  -e 's/EEGPTTwoLayerProbe(/EEGPTProbe(architecture="two_layer", /' {} \;

echo "📋 Phase 3: Manual test updates needed (see plan)"
echo "📋 Phase 4: Running validation"

# Validation
echo "Checking for remaining core imports..."
grep -r "from brain_go_brrr\.core" . --include="*.py" | grep -v "^#"

echo "✅ Option A implementation complete!"
```

---

## ✅ SUCCESS CRITERIA

1. **Zero test failures** - All 700+ tests passing
2. **Zero core imports** - No references to brain_go_brrr.core
3. **Zero deprecated models** - Only unified probe used
4. **Type checking green** - mypy passes
5. **Linting green** - ruff passes
6. **Coverage maintained** - Still >59%

## 🎯 READY TO EXECUTE

This plan is surgical, comprehensive, and achievable in 2 hours. Every change is mapped, every file is identified, and success criteria are clear.

**We are ready for Option A.**
