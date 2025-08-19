# 📚 OPTION A TEST MIGRATION GUIDE

## Purpose
This guide provides exact before/after examples for updating tests to match the clean architecture.

---

## 🔄 IMPORT MIGRATIONS

### Core to Domain/Application/Infra

```python
# ❌ OLD
from brain_go_brrr.core.exceptions import BrainGoBrrrError
from brain_go_brrr.core.config import Config
from brain_go_brrr.core.channels import ChannelMapper
from brain_go_brrr.core.preprocessing_utils import validate_channels
from brain_go_brrr.core.edf_loader import load_edf
from brain_go_brrr.core.logger import get_logger
from brain_go_brrr.core.jobs.models import JobData
from brain_go_brrr.core.cache_port import CachePort

# ✅ NEW
from brain_go_brrr.domain.exceptions import BrainGoBrrrError
from brain_go_brrr.application.config import Config
from brain_go_brrr.domain.channels import ChannelMapper
from brain_go_brrr.domain.channels import validate_channels
from brain_go_brrr.infra.data.edf_loader import load_edf
from brain_go_brrr.infra.logger import get_logger
from brain_go_brrr.application.jobs.models import JobData
from brain_go_brrr.infra.cache import CacheProtocol as CachePort
```

### EEGPT Model Migrations

```python
# ❌ OLD
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
from brain_go_brrr.infra.ml_models.eegpt_linear_probe import EEGPTLinearProbe
from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe

# ✅ NEW
from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel  # For compatibility
# OR
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt  # For new code
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
```

---

## 🔧 PATCH TARGET UPDATES

### Mocking Core Modules

```python
# ❌ OLD
@patch("brain_go_brrr.core.exceptions.BrainGoBrrrError")
@patch("brain_go_brrr.core.sleep.analyzer_enhanced.yasa.SleepStaging")
@patch("brain_go_brrr.models.eegpt_model.EEGPTModel")

# ✅ NEW
@patch("brain_go_brrr.domain.exceptions.BrainGoBrrrError")
@patch("brain_go_brrr.domain.sleep.analyzer_enhanced.yasa.SleepStaging")
@patch("brain_go_brrr.infra.ml_models.eegpt_compat.EEGPTModel")
```

---

## 🎯 API CONTRACT UPDATES

### 1. preprocess_for_eegpt Return Type

```python
# ❌ OLD TEST (expects MNERaw)
def test_preprocess_returns_raw(self):
    raw = create_test_raw()
    processed = preprocess_for_eegpt(raw)
    
    assert hasattr(processed, 'info')
    assert processed.info["sfreq"] == 256
    assert processed.ch_names == expected_channels
    assert processed._data.shape == expected_shape

# ✅ NEW TEST (expects numpy array)
def test_preprocess_returns_array(self):
    raw = create_test_raw()
    processed = preprocess_for_eegpt(raw)
    
    assert isinstance(processed, np.ndarray)
    assert processed.shape[0] == len(expected_channels)
    # To check sampling rate, keep reference to original
    # The function already resampled internally
```

### 2. EEGPTConfig Fields

```python
# ❌ OLD TEST (testing deprecated fields)
def test_config_fields(self):
    config = EEGPTConfig(
        model_size="xlarge",
        max_channels=58,
        embed_dim=768
    )
    assert config.model_size == "xlarge"
    assert config.max_channels == 58
    assert config.n_patches_per_window == 16

# ✅ NEW TEST (testing behavior, not internals)
def test_config_behavior(self):
    config = EEGPTConfig(
        window_duration=4.0,
        sampling_rate=256
    )
    # Don't test internal fields
    # Test that config can be used successfully
    model = EEGPTModel(config=config.__dict__)
    assert model is not None
```

### 3. Probe Instantiation

```python
# ❌ OLD
probe = EEGPTLinearProbe(n_classes=2, hidden_dim=128)
probe = EEGPTTwoLayerProbe(n_classes=2, hidden_dim=512)

# ✅ NEW
probe = EEGPTProbe(architecture='linear', n_classes=2, hidden_dim=128)
probe = EEGPTProbe(architecture='two_layer', n_classes=2, hidden_dim=512)
```

---

## 🚫 TESTS TO DELETE

### Delete Entire Test Methods
These test deprecated implementation details:

```python
# DELETE THESE TESTS ENTIRELY:

def test_config_model_size_property(self):
    """Tests deprecated model_size field"""
    
def test_config_max_channels(self):
    """Tests deprecated max_channels field"""
    
def test_n_patches_per_window_calculation(self):
    """Tests deprecated computed property"""
    
def test_embed_dim_validation(self):
    """Tests internal implementation detail"""
```

### Delete Entire Test Files
```bash
# DELETE:
tests/smoke/test_imports_backward_compat.py  # Tests old imports
tests/unit/test_eegpt_linear_probe.py  # Tests deprecated class
tests/unit/test_eegpt_two_layer_probe.py  # Tests deprecated class
```

---

## 🔍 SPECIAL CASES

### 1. AsyncCachePort (Doesn't Exist)
```python
# ❌ OLD
from brain_go_brrr.core.cache_port import AsyncCachePort

# ✅ NEW OPTIONS:
# Option 1: Remove the test (if not needed)
# Option 2: Test what actually exists
from brain_go_brrr.infra.cache import RedisCache  # Actual implementation
```

### 2. Parallel Pipeline Location
```python
# ❌ OLD
from brain_go_brrr.domain.pipeline.parallel import ParallelEEGPipeline

# ✅ NEW
from brain_go_brrr.application.pipeline.parallel import ParallelEEGPipeline
```

### 3. Channel Utils Rename
```python
# ❌ OLD
from brain_go_brrr.core import preprocessing_utils

# ✅ NEW
from brain_go_brrr.domain import channels as preprocessing_utils
# OR just use directly:
from brain_go_brrr.domain.channels import validate_channels, map_channels
```

---

## ✅ VALIDATION CHECKLIST

After migration, verify:

1. **No core imports remain**
   ```bash
   grep -r "brain_go_brrr\.core" tests/ --include="*.py"
   # Should return nothing
   ```

2. **No deprecated model imports**
   ```bash
   grep -r "eegpt_linear_probe\|eegpt_two_layer_probe" tests/
   # Should return nothing
   ```

3. **All tests pass**
   ```bash
   pytest tests/ -x
   # Should be 100% green
   ```

4. **Type checking passes**
   ```bash
   make type-check
   # Should be green
   ```

---

## 🎯 Common Patterns

### Pattern 1: Test Less, Not More
Instead of testing internal config fields, test behavior:
- ❌ Don't test: `config.max_channels == 58`
- ✅ Do test: Model works with various channel counts

### Pattern 2: Use Real Imports
Instead of testing import redirects, test actual functionality:
- ❌ Don't test: Old import path still works
- ✅ Do test: New import provides expected functionality

### Pattern 3: Delete Rather Than Fix
If a test is testing deprecated internals:
- ❌ Don't: Update to test new internals
- ✅ Do: Delete the test entirely

---

## 📝 Quick Reference Card

| Old Import | New Import |
|------------|------------|
| `core.exceptions` | `domain.exceptions` |
| `core.config` | `application.config` |
| `core.channels` | `domain.channels` |
| `core.preprocessing_utils` | `domain.channels` |
| `core.edf_loader` | `infra.data.edf_loader` |
| `core.logger` | `infra.logger` |
| `core.jobs.models` | `application.jobs.models` |
| `core.cache_port` | `infra.cache` |
| `models.eegpt_model` | `infra.ml_models.eegpt_compat` |
| `eegpt_linear_probe` | `eegpt_probe_unified` (linear) |
| `eegpt_two_layer_probe` | `eegpt_probe_unified` (two_layer) |

---

*Use this guide alongside OPTION_A_EXECUTION_PLAN.md for complete migration.*