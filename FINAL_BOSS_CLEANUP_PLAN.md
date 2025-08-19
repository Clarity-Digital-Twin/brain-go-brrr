# 🎮 FINAL BOSS CLEANUP PLAN - ZERO FAILURES, COMPLETE CLEAN

## 🔴 CURRENT REALITY CHECK

We have **TWO COLLIDING REALITIES**:
1. **Clean hard deletion** - Removed `core/*` and `eegpt_model.py` (good for long-term)
2. **Legacy test expectations** - Tests still assert old world exists (bad for now)

### The Brutal Truth
- ✅ **Type checking**: 100% green (zero errors)
- ✅ **Linting**: 100% green (zero issues) 
- ❌ **Tests**: 32+ failures from missing legacy imports
- ❌ **Compatibility**: Broken for anyone using old APIs

## 🎯 THE DECISION: OPTION A vs OPTION B

### Option A: "100% Clean, v2 Semantics" 🔥
**Break compatibility NOW, ship as major version**
- Remove ALL legacy tests
- Update ALL patches to new paths
- Zero shims, zero redirects
- Clean tests, clean code

### Option B: "Transitional, Tests Green Now" 🌉
**Keep code clean AND satisfy legacy tests**
- Add minimal surgical redirects
- Expand compat shim for test expectations
- Ship now, remove later
- All tests green TODAY

## 📊 FAILURE ANALYSIS

### Import Failures (32 tests)
```python
# FAILING: tests expect these to exist
brain_go_brrr.core.preprocessing_utils
brain_go_brrr.core.edf_loader
brain_go_brrr.core.exceptions
brain_go_brrr.core.config
brain_go_brrr.core.channels
brain_go_brrr.core.jobs.models
brain_go_brrr.core.cache_port
brain_go_brrr.infra.ml_models.eegpt_model.EEGPTModel
brain_go_brrr.domain.pipeline.parallel.ParallelEEGPipeline
```

### API Contract Failures
```python
# Tests expect preprocess_for_eegpt to:
1. Accept target_sfreq parameter
2. Return MNERaw (not ndarray)
3. Have .info, .ch_names, ._data attributes

# Tests expect EEGPTConfig to have:
- model_size: str = "large"
- embed_dim: int
- max_channels: int = 58
- n_patches_per_window: property
- window_samples: property

# Tests expect extract_features to:
- Return np.float32 (not float64)
```

## 🚀 OPTION A: CLEAN BREAK IMPLEMENTATION

### Phase 1: Remove Legacy Test Files (30 min)
```bash
# Delete backward compat tests
rm tests/smoke/test_imports_backward_compat.py

# Update smoke tests to use NEW imports
sed -i 's/brain_go_brrr.core/brain_go_brrr.domain/g' tests/smoke/test_imports.py
```

### Phase 2: Update All Test Patches (1 hour)
```python
# OLD patches in tests
@patch("brain_go_brrr.infra.ml_models.eegpt_model.EEGPTModel")
@patch("brain_go_brrr.core.exceptions.BrainGoBrrrError")

# NEW patches
@patch("brain_go_brrr.infra.ml_models.eegpt_compat.EEGPTModel")
@patch("brain_go_brrr.domain.exceptions.BrainGoBrrrError")
```

### Phase 3: Update Test Expectations (2 hours)
```python
# Tests expecting old config
config = EEGPTConfig(model_size="large", max_channels=58)
assert config.n_patches_per_window == 16

# Update to new API
config = {"window_duration": 4.0, "sampling_rate": 256}
# Remove assertions on deprecated fields
```

### Phase 4: Fix Pipeline Tests (1 hour)
```python
# Tests expecting Raw object
processed = preprocess_for_eegpt(raw)
assert processed.info["sfreq"] == 256  # FAILS - returns ndarray

# Either:
# A) Update test to expect ndarray
assert processed.shape[1] == expected_samples
# B) Update function to return Raw (breaks other code)
```

## 🔧 OPTION B: TRANSITIONAL SHIM IMPLEMENTATION

### Phase 1: Minimal Core Redirects (15 min)
```python
# src/brain_go_brrr/core/__init__.py
"""Minimal redirects for test compatibility - TO BE REMOVED v2.0"""
import sys
from ..domain import channels as preprocessing_utils
from ..domain.exceptions import *
from ..application.config import *
from ..infra.data.edf_loader import *
from ..infra.cache import RedisCacheProtocol as CachePort

# Module aliases for patch() compatibility
sys.modules[__name__ + ".preprocessing_utils"] = preprocessing_utils
sys.modules[__name__ + ".exceptions"] = sys.modules["brain_go_brrr.domain.exceptions"]
sys.modules[__name__ + ".edf_loader"] = sys.modules["brain_go_brrr.infra.data.edf_loader"]

# Jobs models redirect
from ..application.jobs import models as jobs_models
class _Jobs:
    models = jobs_models
jobs = _Jobs()
sys.modules[__name__ + ".jobs"] = jobs
sys.modules[__name__ + ".jobs.models"] = jobs_models
```

### Phase 2: EEGPT Model Visibility (10 min)
```python
# src/brain_go_brrr/infra/ml_models/__init__.py
"""Make old patch paths work without resurrecting files"""
import sys
from . import eegpt_compat as _eegpt_model

# Allow patch("...eegpt_model.EEGPTModel") to work
sys.modules[__name__ + ".eegpt_model"] = _eegpt_model
```

### Phase 3: Expand Compat Shim (30 min)
```python
# src/brain_go_brrr/infra/ml_models/eegpt_compat.py

@dataclass
class EEGPTConfig:
    """Extended compat config matching test expectations"""
    # New fields tests expect
    model_size: str = "large"
    embed_dim: int = 512
    max_channels: int = 58
    
    # Original fields
    sampling_rate: int = 256
    window_duration: float = 4.0
    window_samples: int = 1024
    patch_size: int = 64
    n_channels: int = 20
    device: str = "auto"
    batch_size: int = 32
    
    @property
    def window_samples(self) -> int:
        return int(self.window_duration * self.sampling_rate)
    
    @property
    def n_patches_per_window(self) -> int:
        return self.window_samples // self.patch_size

def preprocess_for_eegpt(
    raw: MNERaw,
    sampling_rate: int = 256,
    target_sfreq: int | None = None,  # Accept both parameter names
    window_duration: float = 4.0,
    bandpass: tuple[float, float] = (0.5, 50.0),
    notch: float = 50.0,
) -> MNERaw:  # Return Raw, not ndarray!
    """Compatibility function that returns MNERaw."""
    sfreq = target_sfreq or sampling_rate
    raw = raw.copy()
    
    if raw.info["sfreq"] != sfreq:
        raw = raw.resample(sfreq)
    
    raw = raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
    raw = raw.notch_filter(freqs=notch)
    
    return raw  # Return Raw object for test compatibility

class EEGPTModel:
    # ... existing code ...
    
    def extract_features(self, data, channel_names=None):
        # ... existing code ...
        # Ensure float32 return
        return features.astype(np.float32)  # Not float64!
```

### Phase 4: Pipeline Re-exports (5 min)
```python
# src/brain_go_brrr/domain/pipeline/__init__.py
"""Re-export for test compatibility"""
from ...application.pipeline.parallel import ParallelEEGPipeline
__all__ = ["ParallelEEGPipeline"]
```

## 📊 IMPACT ANALYSIS

### Option A Impact
- **Pros**: Clean codebase, no tech debt, clear v2 semantics
- **Cons**: 4-5 hours work, must update 60+ test files
- **Risk**: Medium - could break unknown external dependencies

### Option B Impact  
- **Pros**: All tests green in 1 hour, ship today
- **Cons**: Carries minimal tech debt forward
- **Risk**: Low - purely additive, removes nothing

## 🎯 MY RECOMMENDATION: OPTION B NOW, A LATER

### Why Option B First?
1. **Ship TODAY** with zero red tests
2. **Minimal risk** - only adds thin redirects
3. **Clear upgrade path** - mark everything deprecated
4. **Professional** - no broken tests in main branch

### Implementation Order (Option B)
1. ✅ Add minimal core redirects (15 min)
2. ✅ Fix EEGPT model visibility (10 min)
3. ✅ Expand compat shim (30 min)
4. ✅ Add pipeline re-exports (5 min)
5. ✅ Run full test suite (10 min)
6. ✅ Commit & push (5 min)

**Total: 75 minutes to GREEN**

### Then Schedule Option A for v2.0
```python
# Add deprecation warnings everywhere
import warnings

def __getattr__(name):
    warnings.warn(
        f"Importing from brain_go_brrr.core is deprecated. "
        f"This will be removed in v2.0. Use brain_go_brrr.domain instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return globals()[name]
```

## 🚦 SUCCESS CRITERIA

### Immediate (Option B)
- ✅ ALL tests passing (700+)
- ✅ Zero import errors
- ✅ Type checking still green
- ✅ Linting still green
- ✅ Coverage maintained >59%

### Future (Option A - v2.0)
- ✅ Zero redirect modules
- ✅ Zero deprecated imports
- ✅ Single clean API
- ✅ Updated documentation
- ✅ Migration guide published

## 🎮 FINAL BOSS DEFEATED

With Option B, we achieve:
- **Professional outcome**: No red tests
- **Clean architecture**: Core code stays clean
- **Clear path forward**: Deprecations guide to v2
- **Ship today**: Unblocked for external audit

The key insight: **Don't let perfect be the enemy of good.** Ship green tests now, clean up in v2.

---

**Ready to execute Option B? It's 75 minutes to victory.**