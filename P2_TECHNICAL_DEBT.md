# 🟡 P2 TECHNICAL DEBT - Lower Priority Cleanup & Optimization

**Created**: September 8, 2025  
**Owner**: ___________________  
**Time Required**: ~8 hours total  
**Status**: 🔄 NOT STARTED  
**Approach**: Incremental cleanup and optimization

---

## 📋 EXECUTIVE SUMMARY

**P2 items are non-critical improvements that enhance code quality but don't block functionality:**
1. **Incomplete probe migration** - One class still extends deprecated EEGPTProbe
2. **Documentation issues** - Unsafe torch.load examples
3. **PyTorch Lightning** - Still in dependencies despite critical bug
4. **TUEV channel synthesis** - Could improve accuracy by ~1%
5. **Test cleanup** - Redis alias hacks and mock patterns
6. **Residual tech debt** - Various small improvements

**Business Impact**: Code maintainability, developer experience, potential 1% accuracy gain
**Fix Strategy**: Incremental improvements during downtime

---

## 🎯 P2 ISSUES - PRIORITIZED LIST

### 1. AbnormalityDetectionProbe Migration (2 hours)

**Problem**: Still extends deprecated EEGPTProbe instead of using ProbeFactory pattern

**Current State**:
```python
# src/brain_go_brrr/application/use_cases/tasks/abnormality_detection.py
class AbnormalityDetectionProbe(EEGPTProbe):  # P1 TODO: Deferred to P2
```

**Desired State**:
```python
class AbnormalityDetectionProbe:
    def __init__(self, checkpoint_path: Path, n_input_channels: int = 20):
        self.backbone = EEGPTModel(checkpoint_path)
        self.head = ProbeFactory.create_for_task("abnormality", n_classes=2)
```

**Fix Plan**:
1. Refactor to composition over inheritance
2. Use ProbeFactory for head creation
3. Update forward() to extract features with summary=False
4. Ensure backward compatibility with existing checkpoints
5. Update tests to verify functionality

**Files to Modify**:
- `src/brain_go_brrr/application/use_cases/tasks/abnormality_detection.py`
- Related test files

---

### 2. Documentation Unsafe torch.load (15 minutes)

**Problem**: Documentation shows unsafe torch.load without weights_only parameter

**Location**: `docs/TRAINING.md:246`

**Current**:
```python
checkpoint = torch.load("output/tuab_*/best_model.pt")
```

**Fix**:
```python
# For pure tensor data
checkpoint = torch.load("output/tuab_*/best_model.pt", weights_only=True)

# OR for complex objects with justification
checkpoint = torch.load("output/tuab_*/best_model.pt", weights_only=False)  # nosec:weights_only - contains optimizer state

# OR use the safe wrapper
from brain_go_brrr.infra.safe_load import safe_load
checkpoint = safe_load("output/tuab_*/best_model.pt")
```

---

### 3. PyTorch Lightning Dependency Removal (15 minutes)

**Problem**: Lightning still in dependencies despite critical training hang bug

**Current**: pyproject.toml includes `lightning>=2.1.0`

**Fix Options**:

**Option A - Complete Removal**:
```bash
# Remove from pyproject.toml
sed -i '/lightning>=2.1.0/d' pyproject.toml
uv sync
```

**Option B - Move to Optional with Warning**:
```toml
[project.optional-dependencies]
deprecated = [
    "lightning>=2.1.0",  # DO NOT USE - has critical training hang bug
]
```

**Add Guard**:
```python
# src/brain_go_brrr/__init__.py
try:
    import lightning
    import warnings
    warnings.warn(
        "PyTorch Lightning detected! DO NOT USE - has critical training hang bug. "
        "Use pure PyTorch training instead.",
        RuntimeWarning,
        stacklevel=2
    )
except ImportError:
    pass
```

---

### 4. TUEV Channel Synthesis Optimization (4 hours)

**Problem**: Currently using zero-fill for missing Fpz channel, could use learnable mapping

**Current Approach**: Zero-fill missing channels
**EEGPT Authors' Approach**: Learnable Conv2d(23→20) mapping

**Potential Improvement**: ~1% accuracy gain

**Implementation**:
```python
# src/brain_go_brrr/infra/ml_models/channel_mapper.py
class TUEVChannelMapper(nn.Module):
    """Learnable channel mapping from TUEV 23 channels to EEGPT 20 channels."""
    
    def __init__(self):
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv1d(23, 20, kernel_size=1),
            nn.BatchNorm1d(20),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map 23 TUEV channels to 20 EEGPT channels.
        
        Args:
            x: Input tensor (batch, 23, time)
            
        Returns:
            Mapped tensor (batch, 20, time)
        """
        return self.channel_conv(x)
```

**Integration Points**:
- `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
- Training scripts using TUEV data

---

### 5. Test Redis Alias Cleanup (30 minutes)

**Problem**: Tests alias APIRedisCache to RedisCache for mocking convenience

**Locations**:
- `tests/unit/api/test_dependencies.py:9`
- `tests/unit/api/test_cache.py:11`

**Current**:
```python
from brain_go_brrr.api.cache import APIRedisCache as RedisCache  # Convenience alias
```

**Fix**:
```python
from brain_go_brrr.api.cache import APIRedisCache
# Use APIRedisCache directly in tests
```

---

### 6. Services Layer Cleanup (1 hour)

**Problem**: Thin redirect file at services/yasa_adapter.py

**Current**: Deprecated redirect marked for v2.0.0 removal
**Action**: Remove file and update any remaining imports

**Files to Check**:
```bash
rg "from brain_go_brrr.services.yasa_adapter import" --type py
```

---

### 7. Import Linter Rules (30 minutes)

**Add Architecture Enforcement**:

```yaml
# .importlinter
[importlinter:contract:protocols-single-source]
name = Protocols must come from domain.protocols only
type = forbidden
source_modules =
    brain_go_brrr.api
    brain_go_brrr.application
    brain_go_brrr.infra
forbidden_modules =
    brain_go_brrr.domain.ports
    brain_go_brrr.domain.abnormal.ports
message = Use domain.protocols for all protocol imports

[importlinter:contract:no-eegpt-probe]
name = Use ProbeFactory instead of EEGPTProbe
type = forbidden
source_modules =
    brain_go_brrr.application
    brain_go_brrr.api
forbidden_modules =
    brain_go_brrr.infra.ml_models.eegpt_probe_unified
message = Use ProbeFactory.create_for_task() instead of EEGPTProbe
```

---

### 8. Duplicate Class Detection Script (30 minutes)

**Add Pre-commit Hook**:

```python
#!/usr/bin/env python3
# .pre-commit-hooks/check-duplicate-classes.py
"""Prevent duplicate class definitions."""
import sys
import re
from pathlib import Path
from collections import defaultdict

def find_duplicate_classes():
    classes = defaultdict(list)
    
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        for match in re.finditer(r'^class (\w+)', content, re.MULTILINE):
            class_name = match.group(1)
            if not class_name.startswith('_'):  # Skip private classes
                classes[class_name].append(str(py_file))
    
    duplicates = {k: v for k, v in classes.items() if len(v) > 1}
    
    # Allowed duplicates (document these)
    allowed = {
        "JobData",  # API DTO vs Domain Entity pattern
    }
    
    real_duplicates = {k: v for k, v in duplicates.items() if k not in allowed}
    
    if real_duplicates:
        print("ERROR: Duplicate class definitions found!")
        for class_name, files in real_duplicates.items():
            print(f"  {class_name}:")
            for file in files:
                print(f"    - {file}")
        return 1
    
    print(f"✓ Checked {sum(len(v) for v in classes.values())} class definitions")
    print(f"  {len(duplicates)} allowed duplicates: {', '.join(allowed & set(duplicates.keys()))}")
    return 0

if __name__ == "__main__":
    sys.exit(find_duplicate_classes())
```

**Add to .pre-commit-config.yaml**:
```yaml
  - repo: local
    hooks:
      - id: no-duplicate-classes
        name: Check for duplicate class names
        entry: python .pre-commit-hooks/check-duplicate-classes.py
        language: python
        pass_filenames: false
        always_run: true
```

---

### 9. EEGPT Feature Dimension Documentation (1 hour)

**Create canonical documentation for feature dimensions**:

```python
# src/brain_go_brrr/domain/constants.py
"""EEGPT feature dimension constants and documentation."""

# EEGPT Large Model Feature Dimensions
EEGPT_SUMMARY_TOKENS = 4
EEGPT_TOKEN_DIM = 512
EEGPT_PROBE_INPUT_DIM = EEGPT_SUMMARY_TOKENS * EEGPT_TOKEN_DIM  # 2048

def get_eegpt_features_for_probe(model, data, channels):
    """Extract EEGPT features in the correct format for linear probes.
    
    Per EEGPT paper, probes expect 2048 dimensions (4 tokens × 512 dims).
    
    Args:
        model: EEGPT model instance
        data: Input EEG data
        channels: Channel names
        
    Returns:
        Tensor of shape (batch, 2048) for probe input
    """
    features = model.extract_features(data, channels, summary=False)  # (B, 4, 512)
    return features.flatten(1)  # (B, 2048)

def get_eegpt_features_averaged(model, data, channels):
    """Extract averaged EEGPT features for non-probe usage.
    
    Returns 512-dimensional averaged features, suitable for heuristics
    or statistics but NOT for feeding to trained probes.
    
    Args:
        model: EEGPT model instance
        data: Input EEG data
        channels: Channel names
        
    Returns:
        Tensor of shape (batch, 512) averaged features
    """
    return model.extract_features(data, channels, summary=True)  # (B, 512)
```

---

### 10. Code Coverage Improvements (2 hours)

**Current**: 86% unit test coverage
**Target**: 95% coverage

**Focus Areas**:
1. Untested edge cases in preprocessing
2. Error handling paths
3. Configuration validation
4. Cache expiration logic

**Commands**:
```bash
# Generate coverage report
make coverage

# Find uncovered lines
coverage report --show-missing

# Focus on critical paths
coverage run -m pytest tests/unit/ --cov-report=html
```

---

## 📊 IMPLEMENTATION PLAN

### Quick Wins (< 30 minutes each)
1. Documentation torch.load fix - 15 minutes
2. PyTorch Lightning removal - 15 minutes  
3. Test Redis alias cleanup - 30 minutes

### Medium Tasks (1-2 hours each)
1. Services layer cleanup - 1 hour
2. Import linter rules - 30 minutes
3. Duplicate class detection - 30 minutes
4. EEGPT dimension documentation - 1 hour

### Larger Tasks (2+ hours)
1. AbnormalityDetectionProbe migration - 2 hours
2. TUEV channel synthesis - 4 hours
3. Code coverage improvements - 2 hours

---

## ✅ DEFINITION OF DONE

Each P2 item is complete when:

### Code Changes
- [ ] Implementation complete and tested
- [ ] All existing tests still pass
- [ ] New tests added for new functionality
- [ ] Type checking passes (`make typecheck`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation updated if needed

### Quality Gates
- [ ] No performance regression
- [ ] No new warnings introduced
- [ ] Code coverage maintained or improved
- [ ] Pre-commit hooks pass

### Documentation
- [ ] Code comments explain why (not what)
- [ ] CHANGELOG.md updated if user-facing
- [ ] README updated if new features

---

## 🚀 RECOMMENDED EXECUTION ORDER

### Sprint 1 (Quick Wins - 1 hour total)
1. Fix torch.load in docs
2. Remove/quarantine PyTorch Lightning
3. Clean up test Redis aliases

### Sprint 2 (Architecture - 3 hours)
1. Complete AbnormalityDetectionProbe migration
2. Add import linter rules
3. Add duplicate class detection

### Sprint 3 (Optimization - 4 hours)
1. Implement TUEV channel synthesis
2. Create EEGPT dimension documentation
3. Improve code coverage to 95%

### Sprint 4 (Final Cleanup - 1 hour)
1. Remove deprecated services redirects
2. Final verification and documentation

---

## 📈 SUCCESS METRICS

### Immediate Benefits
- Zero deprecated class usage
- No unsafe torch.load in docs
- No accidental Lightning usage
- Cleaner test code

### Long-term Benefits
- +1% accuracy on TUEV dataset
- 95% code coverage
- Enforced architecture boundaries
- No duplicate classes

### Developer Experience
- Clear documentation on feature dimensions
- Import rules prevent mistakes
- Pre-commit catches issues early
- Consistent naming patterns

---

## 🔍 VERIFICATION COMMANDS

```bash
# Check for remaining EEGPTProbe usage
rg "from.*eegpt_probe_unified import EEGPTProbe" src/brain_go_brrr/application

# Verify torch.load safety
rg "torch\.load\(" docs/ | grep -v "weights_only"

# Check for Lightning imports
rg "import lightning" src/

# Find test aliases
rg "as RedisCache" tests/

# Check coverage
make coverage

# Verify no duplicate classes (after script added)
python .pre-commit-hooks/check-duplicate-classes.py
```

---

## 📝 NOTES

### Why These Are P2 (Not P1)
- No runtime crashes
- No data corruption
- No blocking functionality
- Mostly code quality improvements

### Risk Assessment
- **Low Risk**: All changes are isolated
- **High Value**: Improves maintainability
- **Optional**: Can defer without impact

### Dependencies
- P1 fixes must be complete first
- No external dependencies
- Can be done incrementally

---

**Approved By**: _________________ **Date**: _______  
**Developer Assigned**: ____________ **Target Sprint**: _______