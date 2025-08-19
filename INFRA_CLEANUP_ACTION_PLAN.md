# Infrastructure Cleanup Action Plan

## 🎯 Goal: Clean up infra/ml_models WITHOUT breaking production

## Phase 1: Immediate Fixes (Safe, No Breaking Changes)

### 1.1 Delete Never-Used Files ✅ SAFE
```bash
# These are NEVER imported according to grep
rm src/brain_go_brrr/infra/ml_models/eegpt_linear_probe_robust.py
rm src/brain_go_brrr/infra/cache_port.py
```

### 1.2 Add Deprecation Warnings
```python
# In eegpt_model.py - add to __init__:
warnings.warn(
    "EEGPTModel is deprecated. Use eegpt_wrapper.EEGPTWrapper instead. "
    "Will be removed in v2.0.0",
    DeprecationWarning,
    stacklevel=2
)
```

## Phase 2: Create Unified Probe (New File, No Breaking)

### 2.1 Create New Unified Probe
```python
# src/brain_go_brrr/infra/ml_models/eegpt_probe_unified.py
class EEGPTProbe(nn.Module):
    """Unified EEGPT probe - replaces all variants."""
    
    def __init__(
        self,
        checkpoint_path: Path,
        n_classes: int,
        architecture: str = "linear",  # "linear" or "two_layer"
        robust_mode: bool = False,     # NaN handling
        channel_adapter: bool = False, # Channel adaptation
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        # Load EEGPT backbone
        self.backbone = create_normalized_eegpt(checkpoint_path)
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Optional channel adapter
        if channel_adapter:
            self.channel_adapter = nn.Conv1d(n_input_channels, 20, 1)
        else:
            self.channel_adapter = None
        
        # Probe architecture
        if architecture == "linear":
            self.probe = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
            )
        elif architecture == "two_layer":
            self.probe = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_classes)
            )
        
        self.robust_mode = robust_mode
    
    def forward(self, x, return_all_temporal=False):
        # Robust mode checks
        if self.robust_mode:
            x = torch.clamp(x, -50, 50)
            assert not torch.isnan(x).any(), "NaN in input"
        
        # Channel adaptation
        if self.channel_adapter:
            x = self.channel_adapter(x)
        
        # Extract features
        features = self.backbone.extract_features(x, return_all_temporal=return_all_temporal)
        
        # Flatten if temporal
        if return_all_temporal:
            features = features.flatten(1)
        else:
            features = features.mean(dim=1) if features.dim() == 3 else features
        
        # Probe
        return self.probe(features)
```

### 2.2 Create Migration Shim
```python
# Add to existing probe files:
def __init__(self, *args, **kwargs):
    warnings.warn(
        f"{self.__class__.__name__} is deprecated. "
        "Use EEGPTProbe from eegpt_probe_unified instead.",
        DeprecationWarning
    )
    super().__init__(*args, **kwargs)
```

## Phase 3: Extract Functions from eegpt_model.py

### 3.1 Move Preprocessing Functions
```python
# Move to: src/brain_go_brrr/domain/preprocessing/eegpt_preprocessing.py
def preprocess_for_eegpt(raw: MNERaw, ...) -> np.ndarray:
    """Preprocess raw EEG for EEGPT input."""
    # Move existing function

def extract_windows(data: np.ndarray, ...) -> list[np.ndarray]:
    """Extract fixed-size windows from continuous data."""
    # Move existing function
```

### 3.2 Move Pipeline Functions
```python
# Move to: src/brain_go_brrr/application/pipelines/eegpt_pipeline.py
def predict_abnormality(model, raw: MNERaw) -> dict:
    """Full abnormality detection pipeline."""
    # Move from EEGPTModel.predict_abnormality()

def analyze(model, raw: MNERaw, analysis_type: str) -> dict:
    """Orchestrate different analysis types."""
    # Move from EEGPTModel.analyze()
```

### 3.3 Keep Only Core Model Wrapper
```python
# eegpt_model.py becomes thin compatibility layer:
class EEGPTModel:
    """Deprecated - use EEGPTWrapper directly."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn("Use EEGPTWrapper", DeprecationWarning)
        self.encoder = create_normalized_eegpt(...)
    
    # Delegate methods for compatibility
    def extract_features(self, x):
        return self.encoder.extract_features(x)
```

## Phase 4: Update Imports (Gradual Migration)

### 4.1 Update CLI
```python
# OLD: from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
# NEW: from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

# OLD: model = EEGPTModel(checkpoint_path=path)
# NEW: model = create_normalized_eegpt(checkpoint_path=path)
```

### 4.2 Update API Routes
```python
# In api/routers/eegpt.py
# OLD: from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
# NEW: from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
```

### 4.3 Update Tasks
```python
# abnormality_detection.py
# OLD: from brain_go_brrr.infra.ml_models.eegpt_linear_probe import EEGPTLinearProbe
# NEW: from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe

# enhanced_abnormality_detection.py
# OLD: from brain_go_brrr.infra.ml_models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
# NEW: from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
```

## Phase 5: Simplify Cache

### 5.1 Merge Cache Files
```python
# Keep only cache.py, add factory method:
def create_cache(cache_type="memory", **kwargs):
    if cache_type == "memory":
        return InMemoryCache(**kwargs)
    raise ValueError(f"Unknown cache type: {cache_type}")

# Delete cache_factory.py and cache_port.py
```

## Phase 6: Final Cleanup (After Verification)

### 6.1 Delete Deprecated Files
```bash
# After all imports updated and tested:
rm src/brain_go_brrr/infra/ml_models/eegpt_model.py
rm src/brain_go_brrr/infra/ml_models/eegpt_linear_probe.py
rm src/brain_go_brrr/infra/ml_models/eegpt_two_layer_probe.py
rm src/brain_go_brrr/infra/cache_factory.py
```

### 6.2 Update __init__.py
```python
# src/brain_go_brrr/infra/ml_models/__init__.py
from .eegpt_architecture import EEGTransformer, create_eegpt_model
from .eegpt_wrapper import EEGPTWrapper, create_normalized_eegpt
from .eegpt_probe_unified import EEGPTProbe
from .linear_probe import LinearProbeHead  # Keep generic version

__all__ = [
    "EEGTransformer",
    "EEGPTWrapper", 
    "EEGPTProbe",
    "LinearProbeHead",
    "create_eegpt_model",
    "create_normalized_eegpt",
]
```

## Testing Strategy

### Before Each Phase:
```bash
# Run existing tests
pytest tests/unit/test_models_*.py -xvs

# Check imports still work
python -c "from brain_go_brrr.infra.ml_models import *"

# Test API endpoints
curl http://localhost:8000/api/v1/health
```

### Integration Test:
```python
# test_migration.py
def test_old_imports_still_work():
    """Ensure deprecated imports show warnings but work."""
    with warnings.catch_warnings(record=True) as w:
        from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()

def test_new_unified_probe():
    """Test unified probe handles all cases."""
    probe = EEGPTProbe(
        checkpoint_path=path,
        n_classes=2,
        architecture="linear",
        robust_mode=True
    )
    # Test it works
```

## Timeline

### Week 1: Phase 1-2
- Delete unused files
- Add deprecation warnings
- Create unified probe

### Week 2: Phase 3-4
- Extract functions to proper modules
- Start updating imports
- Test each change

### Week 3: Phase 5-6
- Simplify cache
- Final cleanup
- Update documentation

## Risk Mitigation

1. **Keep old files during migration** - Only delete after verification
2. **Use deprecation warnings** - Give time for updates
3. **Test after each phase** - Catch breaks early
4. **Create migration guide** - Help others update code
5. **Version properly** - Breaking changes in major version

## Success Metrics

- ✅ Reduce 6 EEGPT files to 3 (architecture, wrapper, probe)
- ✅ Reduce 4 probe variants to 1 unified
- ✅ Move preprocessing/pipeline functions to proper layers
- ✅ Simplify cache from 3 files to 1
- ✅ All tests passing
- ✅ No production breakage

## DO NOT TOUCH

These files are working and were just fixed:
- `eegpt_architecture.py` - Core implementation with temporal fix
- `eegpt_wrapper.py` - Clean wrapper with temporal support

## Priority Order

1. **HIGH**: Delete never-used files (immediate win)
2. **HIGH**: Create unified probe (stops proliferation)
3. **MEDIUM**: Extract functions from eegpt_model.py
4. **LOW**: Simplify cache (not urgent)

This plan provides a safe, gradual migration path to clean up the infrastructure technical debt without breaking production systems.