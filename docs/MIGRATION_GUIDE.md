# Migration Guide

## v1.2.0 → v1.3.0

### EEGPT Dimension Constants

**Old Code (hardcoded values):**
```python
# In your code
if features.shape[-1] == 512:
    # Single summary token
    pass
elif features.shape[-1] == 2048:
    # All 4 tokens flattened
    pass
```

**New Code (use constants):**
```python
from brain_go_brrr.domain.constants import (
    EEGPT_TOKEN_DIM,         # 512
    EEGPT_PROBE_INPUT_DIM,   # 2048
    EEGPT_SUMMARY_TOKENS,    # 4
)

if features.shape[-1] == EEGPT_TOKEN_DIM:
    # Single summary token
    pass
elif features.shape[-1] == EEGPT_PROBE_INPUT_DIM:
    # All tokens flattened
    pass
```

### Benefits
- Single source of truth for dimensions
- Better documentation through named constants
- Easier to update if model architecture changes

---

## v1.1.0 → v1.2.0

### Architecture Compliance

All domain→infra imports have been removed. If you were importing from domain layer and relying on transitive imports from infra, you'll need to update:

**Old Code:**
```python
from brain_go_brrr.domain.abnormal.detector import CleanAbnormalityDetector
# This might have worked due to internal imports
```

**New Code:**
```python
from brain_go_brrr.domain.abnormal.detector import CleanAbnormalityDetector
from brain_go_brrr.infra.safe_load import safe_load  # Import directly if needed
```

### Extract Features Parameter Fix

The `extract_features` method now correctly accepts `chan_ids` parameter:

**Old Code (broken):**
```python
features = model.extract_features(x, channel_names=channels)  # Wrong!
```

**New Code:**
```python
features = model.extract_features(x, chan_ids=None)  # Correct
```

---

## v1.0.0 → v1.1.0

### Removed Deprecated Modules

The following deprecated modules have been removed:

**Old Import:**
```python
from brain_go_brrr.core.quality import QualityController
from brain_go_brrr.core.sleep import SleepAnalyzer
from brain_go_brrr.core.config import Config
```

**New Import:**
```python
from brain_go_brrr.domain.quality.controller import EEGQualityController
from brain_go_brrr.domain.sleep import SleepAnalyzer
from brain_go_brrr.application.config import DataConfig
```

### Services Migration

The `services` package has been restructured:

**Old Import:**
```python
from brain_go_brrr.services.yasa_adapter import YASASleepStager
```

**New Import:**
```python
from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager
```

---

## Breaking Changes Summary

### v1.3.0
- No breaking changes, only additions

### v1.2.0
- Domain layer no longer imports from infra (clean architecture)
- `extract_features` parameter name fixed

### v1.1.0
- Removed all `core.*` modules
- Removed `services.yasa_adapter`
- Removed `compat_coerce` parameter from EEGPTModel

---

## Deprecation Timeline

### Currently Deprecated (will be removed in next major version)
- `brain_go_brrr.infra.ml_models.eegpt_compat` - Use `eegpt_wrapper` instead
- `brain_go_brrr.models` package - Use domain/infra equivalents

### Removed in v1.2.0
- All `core.*` redirect modules
- `services` package redirects

### Removed in v1.1.0
- `compat_coerce` parameter from EEGPTModel
- Legacy 768-dimension tolerance
