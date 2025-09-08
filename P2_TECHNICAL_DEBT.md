# 🟡 P2 TECHNICAL DEBT - Lower Priority Cleanup & Optimization

**Created**: September 8, 2025  
**Last Audit**: September 8, 2025 (Senior Review)  
**Owner**: ___________________  
**Time Required**: ~9 hours total  
**Status**: 🔄 NOT STARTED  
**Approach**: Incremental cleanup with concrete acceptance criteria

---

## 📋 EXECUTIVE SUMMARY

**P2 items are non-critical improvements that enhance code quality but don't block functionality:**
1. **Incomplete probe migration** - One class still extends deprecated EEGPTProbe
2. **Duplicate CachePort** - Remove infra/cache_factory.py duplicate (NEW FINDING)
3. **Documentation safety** - Add banners to archived docs (main docs already safe)
4. **PyTorch Lightning guards** - Prevent accidental re-introduction (not in deps)
5. **TUEV channel synthesis** - Optional learnable mapping for +1% accuracy
6. **Test cleanup** - Redis alias removal and services redirect cleanup
7. **Architecture enforcement** - Import linter rules and duplicate detection
8. **Code coverage** - Target 95% coverage on critical paths

**Business Impact**: Code maintainability, developer experience, potential 1% accuracy gain
**Fix Strategy**: Incremental improvements with clear acceptance gates

---

## 🎯 P2 ISSUES - VERIFIED & PRIORITIZED

### 1. AbnormalityDetectionProbe Migration (2 hours) ⭐ HIGHEST PRIORITY

**Problem**: Still extends deprecated EEGPTProbe instead of using ProbeFactory pattern

**Current State**:
```python
# src/brain_go_brrr/application/use_cases/tasks/abnormality_detection.py:35
class AbnormalityDetectionProbe(EEGPTProbe):  # P1 TODO: Deferred to P2
```

**Target Implementation**:
```python
class AbnormalityDetectionProbe:
    def __init__(self, checkpoint_path: Path, n_input_channels: int = 20):
        self.backbone = create_normalized_eegpt(checkpoint_path=checkpoint_path)
        self.head = ProbeFactory.create_for_task("abnormality", n_classes=2)
    
    def forward(self, x, channel_names):
        # Extract features with summary=False
        features = self.backbone.extract_features(x, channel_names, summary=False)
        # Prepare for probe: (B, 4, 512) → (B, 2048)
        probe_input = prepare_probe_features(features)
        # Pass to head
        return self.head(probe_input)
```

**Checkpoint Migration**:
```python
# Support both old and new checkpoint formats
if "probe_state_dict" in checkpoint:
    # Old format - use migration utility
    new_state = migrate_eegpt_probe_to_factory(checkpoint["probe_state_dict"])
    self.head.load_state_dict(new_state)
elif "model_state_dict" in checkpoint:
    # New format
    self.head.load_state_dict(checkpoint["model_state_dict"])
```

**Acceptance Criteria**:
- [ ] No `from.*eegpt_probe_unified import EEGPTProbe` in application layer
- [ ] Can load both old and new checkpoint formats
- [ ] Parity test: logits match within 1e-5 tolerance (fixed seed, synthetic input)
- [ ] Head receives (B, 2048) shaped input
- [ ] Add `PendingDeprecationWarning` for API stability

---

### 2. Remove Duplicate CachePort (30 minutes) 🆕 NEW FINDING

**Problem**: Two CachePort definitions exist - violates single source of truth

**Current Duplicates**:
```bash
src/brain_go_brrr/domain/ports/cache.py:9        # ✅ Canonical
src/brain_go_brrr/infra/cache_factory.py:25      # ❌ Duplicate to remove
```

**Fix Plan**:
1. Update `infra/cache_factory.py` to import from domain
2. Remove duplicate Protocol definition
3. Update any local references

**Acceptance Criteria**:
- [ ] `rg -n '^class\s+CachePort\b' src` returns only domain/ports/cache.py
- [ ] All tests pass after removal
- [ ] Type checking passes

---

### 3. Documentation Safety Banners (15 minutes) ✅ CORRECTED

**Problem**: Archived docs may show unsafe torch.load (main docs already safe)

**Current Status**:
- ✅ `docs/TRAINING.md:246` is ALREADY SAFE: `weights_only=False  # nosec:weights_only`
- ⚠️ Archive files may have legacy examples

**Fix Plan**:
```markdown
# Add to top of archived documentation files:
> ⚠️ **ARCHIVED DOCUMENTATION** - Examples may be outdated.  
> See [TRAINING.md](../TRAINING.md) for current safe torch.load usage.
```

**Files to Update**:
- Any files in `docs/archive/` or `docs/legacy/`
- Historical examples in `literature/`

**Acceptance Criteria**:
- [ ] Main docs pass: `rg "torch\.load\(" docs | grep -v "weights_only" | grep -v "nosec" | grep -v "archive"` → empty
- [ ] Archive files contain safety banner
- [ ] No unsafe examples in primary documentation

---

### 4. PyTorch Lightning Guardrails (30 minutes) ✅ CORRECTED

**Problem**: Prevent accidental re-introduction (NOT currently in dependencies)

**Current Status**: 
- ✅ VERIFIED: No Lightning in pyproject.toml
- Need CI guards to keep it that way

**Implementation**:
```yaml
# .github/workflows/no-lightning.yml
name: Prevent Lightning
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for Lightning imports
        run: |
          if rg "import\s+lightning|from\s+lightning" src/ experiments/; then
            echo "❌ PyTorch Lightning detected! Use pure PyTorch."
            echo "See CLAUDE.md for critical training hang bug details."
            exit 1
          fi
```

**Acceptance Criteria**:
- [ ] CI job added and passing
- [ ] `rg "import\s+lightning" src experiments` returns empty
- [ ] Warning present in CLAUDE.md and AGENTS.md

---

### 5. Services Redirect Cleanup (1 hour) ✅ VERIFIED USAGES

**Problem**: Deprecated redirect file with 3 active import sites

**Current Redirect**: `src/brain_go_brrr/services/yasa_adapter.py`

**Import Sites to Update FIRST**:
1. `src/brain_go_brrr/application/factories.py:15`
2. `tests/integration/test_yasa_integration.py:8`
3. `tests/integration/test_yasa_channel_aliasing.py:7`

**Fix Sequence**:
```python
# Step 1: Update all imports
# FROM:
from brain_go_brrr.services.yasa_adapter import YASASleepStager, YASAConfig

# TO:
from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager, YASAConfig
```

```bash
# Step 2: Remove redirect file
rm src/brain_go_brrr/services/yasa_adapter.py
```

**Acceptance Criteria**:
- [ ] All 3 import sites updated
- [ ] `rg "services.yasa_adapter" src tests` returns empty
- [ ] Redirect file deleted
- [ ] All tests green

---

### 6. Test Redis Alias Cleanup (30 minutes)

**Problem**: Tests use confusing alias pattern

**Current Aliases**:
- `tests/unit/api/test_dependencies.py:9`
- `tests/unit/api/test_cache.py:11`

**Current Pattern**:
```python
from brain_go_brrr.api.cache import APIRedisCache as RedisCache  # Confusing
```

**Fix To**:
```python
from brain_go_brrr.api.cache import APIRedisCache  # Clear and direct
```

**Acceptance Criteria**:
- [ ] `rg "as RedisCache" tests` returns empty
- [ ] All cache tests pass
- [ ] No new aliases introduced

---

### 7. TUEV Channel Synthesis (4 hours) 🔬 OPTIONAL ENHANCEMENT

**Problem**: Zero-fill approach vs learnable mapping (potential +1% accuracy)

**Implementation**:
```python
# src/brain_go_brrr/infra/ml_models/channel_mapper.py
class TUEVChannelMapper(nn.Module):
    """Learnable channel mapping from TUEV 23 → EEGPT 20 channels."""
    
    def __init__(self):
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv1d(23, 20, kernel_size=1),
            nn.BatchNorm1d(20),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, 23, T) → (B, 20, T)."""
        return self.channel_conv(x)
```

**Configuration**:
```python
# Add to config
enable_tuev_channel_mapper: bool = False  # Default OFF for safety
```

**Integration Point**: Apply BEFORE normalization in pipeline

**Acceptance Criteria**:
- [ ] Shape test: (B,23,T) → (B,20,T) with correct dtype/device
- [ ] Gradients flow properly
- [ ] Config on/off works without regression
- [ ] Document channel order dependencies
- [ ] No impact on non-TUEV paths

---

### 8. Import Linter Rules (30 minutes)

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

[importlinter:contract:no-direct-eegpt-probe]
name = Use ProbeFactory instead of EEGPTProbe
type = forbidden
source_modules =
    brain_go_brrr.application
    brain_go_brrr.api
forbidden_modules =
    brain_go_brrr.infra.ml_models.eegpt_probe_unified
message = Use ProbeFactory.create_for_task() instead
```

**Acceptance Criteria**:
- [ ] Import linter configured and passing
- [ ] CI integration added
- [ ] Violations cause build failure

---

### 9. Duplicate Class Detection (30 minutes)

**Known Duplicates**:
- `CachePort` (to be fixed in item #2)
- `JobData` (allowed - API DTO vs Domain Entity pattern)

**Pre-commit Hook**:
```python
#!/usr/bin/env python3
# .pre-commit-hooks/check-duplicate-classes.py
"""Prevent duplicate class definitions."""
import sys
import re
from pathlib import Path
from collections import defaultdict

ALLOWED_DUPLICATES = {
    "JobData",  # API DTO vs Domain Entity pattern
}

def find_duplicate_classes():
    classes = defaultdict(list)
    
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        for match in re.finditer(r'^class (\w+)', content, re.MULTILINE):
            class_name = match.group(1)
            if not class_name.startswith('_'):
                classes[class_name].append(str(py_file))
    
    duplicates = {k: v for k, v in classes.items() if len(v) > 1}
    real_duplicates = {k: v for k, v in duplicates.items() if k not in ALLOWED_DUPLICATES}
    
    if real_duplicates:
        print("❌ Duplicate class definitions found!")
        for class_name, files in real_duplicates.items():
            print(f"  {class_name}:")
            for file in files:
                print(f"    - {file}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(find_duplicate_classes())
```

**Acceptance Criteria**:
- [ ] Hook catches CachePort duplicate (before fix)
- [ ] Hook passes after CachePort fix
- [ ] Integrated into pre-commit config

---

### 10. EEGPT Feature Documentation (1 hour)

**Create Canonical Constants**:

```python
# src/brain_go_brrr/domain/constants.py
"""EEGPT feature dimension constants - SINGLE SOURCE OF TRUTH."""

# EEGPT Large Model Feature Dimensions
EEGPT_SUMMARY_TOKENS = 4
EEGPT_TOKEN_DIM = 512
EEGPT_PROBE_INPUT_DIM = EEGPT_SUMMARY_TOKENS * EEGPT_TOKEN_DIM  # 2048

# Note: Logic remains in utils/probe_utils.py:prepare_probe_features
# These constants are for documentation and testing only
```

**Acceptance Criteria**:
- [ ] Constants file created and imported in at least one test
- [ ] No logic duplication (keep in prepare_probe_features)
- [ ] Documentation references these constants

---

### 11. Code Coverage Improvements (2 hours)

**Current**: 86% coverage
**Target**: 95% coverage

**Focus Areas for Maximum Impact**:
1. **Cache TTL expiry** in `InMemoryCache.clear_pattern`
2. **Checkpoint migration** edge cases in `migrate_eegpt_probe_to_factory`
3. **Redis connection errors** in `infra/cache.py`
4. **Config validation** edge cases

**Test Additions Needed**:
```python
# Test cache TTL expiry
def test_cache_ttl_expiry():
    cache = InMemoryCache()
    cache.set("key", "value", ttl=0.01)
    time.sleep(0.02)
    assert cache.get("key") is None

# Test migration failure
def test_checkpoint_migration_missing_keys():
    with pytest.raises(KeyError):
        migrate_eegpt_probe_to_factory({"wrong_key": "data"})

# Test Redis timeout
def test_redis_connection_timeout():
    cache = APIRedisCache(host="invalid", timeout=0.001)
    with pytest.raises(ConnectionError):
        cache.get("key")
```

**Acceptance Criteria**:
- [ ] Coverage ≥ 95% for targeted modules
- [ ] All edge cases have tests
- [ ] No reduction in overall coverage

---

## 📊 IMPLEMENTATION STRATEGY

### Quick Wins (< 30 minutes each) - Do First
1. Documentation safety banners - 15 minutes
2. Test Redis alias cleanup - 30 minutes
3. Remove duplicate CachePort - 30 minutes

### Architecture Guards (1 hour total)
1. PyTorch Lightning CI guard - 30 minutes
2. Import linter rules - 30 minutes

### Medium Tasks (1-2 hours each)
1. Services redirect cleanup - 1 hour (3 imports + delete)
2. Duplicate class detection hook - 30 minutes
3. EEGPT dimension documentation - 1 hour

### Larger Tasks (2+ hours)
1. AbnormalityDetectionProbe migration - 2 hours ⭐
2. TUEV channel synthesis - 4 hours (optional)
3. Code coverage to 95% - 2 hours

---

## ✅ GLOBAL ACCEPTANCE CRITERIA

### Code Quality Gates
```bash
make typecheck              # ✅ Must pass
make lint                   # ✅ Must pass
make test                   # ✅ Must pass
make test-all-cov          # ✅ Must pass
```

### Architecture Verification
```bash
# No EEGPTProbe in application layer
rg "from.*eegpt_probe_unified import EEGPTProbe" src/brain_go_brrr/application
# → EMPTY

# No services.yasa_adapter imports
rg "services.yasa_adapter" src tests
# → EMPTY

# Single CachePort definition
rg "^class\s+CachePort\b" src
# → ONLY domain/ports/cache.py

# No unsafe torch.load in main docs
rg "torch\.load\(" docs | grep -v "weights_only" | grep -v "nosec" | grep -v "archive"
# → EMPTY

# No Lightning imports
rg "import\s+lightning" src experiments
# → EMPTY
```

### Test Coverage
```bash
# Generate report
make coverage

# Target metrics
- Overall: ≥ 86% (maintain)
- Targeted modules: ≥ 95%
- No new uncovered critical paths
```

---

## 🚀 RECOMMENDED EXECUTION ORDER

### Sprint 1: Quick Wins & Guards (2 hours)
**Goal**: Prevent regressions, clean obvious issues
1. Remove duplicate CachePort ✓
2. Add PyTorch Lightning CI guard ✓
3. Clean test Redis aliases ✓
4. Add documentation safety banners ✓

### Sprint 2: Architecture Cleanup (3 hours)
**Goal**: Complete P1 deferral, enforce boundaries
1. AbnormalityDetectionProbe migration ⭐
2. Services redirect cleanup (3 imports + delete)
3. Import linter rules
4. Duplicate class detection

### Sprint 3: Documentation & Coverage (3 hours)
**Goal**: Make codebase maintainable
1. EEGPT dimension constants
2. Code coverage to 95%
3. Update all documentation

### Sprint 4: Optional Enhancement (4 hours)
**Goal**: Performance improvement
1. TUEV channel synthesis (+1% accuracy)
2. Performance benchmarks
3. Integration tests

---

## 📈 SUCCESS METRICS

### Immediate Benefits
- Zero duplicate Protocol definitions
- No deprecated class usage in application layer
- No unsafe torch.load in documentation
- No accidental Lightning usage
- Cleaner test code

### Long-term Benefits
- +1% accuracy on TUEV dataset (if channel mapper implemented)
- 95% code coverage on critical paths
- Enforced architecture boundaries via import linter
- No duplicate classes (except allowed)
- Clear feature dimension documentation

### Developer Experience
- Single source of truth for all Protocols
- Clear import patterns enforced by CI
- Pre-commit catches issues early
- Comprehensive test coverage
- No confusing aliases or redirects

---

## 🔍 VERIFICATION COMMANDS

```bash
# Run after each sprint to verify progress
./scripts/verify_p2_progress.sh

# Manual checks
rg "class.*EEGPTProbe\(" src/brain_go_brrr/application  # Should be empty
rg "as RedisCache" tests/                                # Should be empty
rg "services.yasa_adapter" --type py                     # Should be empty
rg "^class\s+CachePort" src/                            # Single result only
make coverage | grep "TOTAL"                             # Should be ≥95%
```

---

## 📝 NOTES

### Why These Are P2 (Not P1)
- No runtime crashes or data corruption
- No blocking functionality
- Mostly code quality and maintainability
- Performance improvements are optional

### Risk Assessment
- **Low Risk**: All changes are isolated and tested
- **High Value**: Significant maintainability improvements
- **Optional**: Can defer without production impact

### Dependencies
- P1 fixes must be complete first ✅ (95% done, 1 deferral)
- No external dependencies
- Can be done incrementally

### Lessons from P1
- Test everything locally first
- Run CI checks before committing
- Verify each claim before implementing
- Keep clear acceptance criteria
- Document what was actually done vs deferred

---

**Approved By**: _________________ **Date**: _______  
**Developer Assigned**: ____________ **Target Sprint**: _______  
**Last Senior Audit**: September 8, 2025 - All findings incorporated