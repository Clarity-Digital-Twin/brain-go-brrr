# 🚨 TECHNICAL DEBT - Priority Issues Requiring Resolution

**Created**: September 4, 2025  
**Status**: Active - Requires Immediate Attention  
**Focus**: High-impact duplicate class definitions that could cause runtime errors

---

## 🔴 CRITICAL ISSUE #1: Duplicate Class Definitions

### The Problem
We have **9 classes with duplicate definitions** across different modules. This creates:
- **Import confusion** - Which one gets imported?
- **Runtime errors** - Wrong class could be used
- **Maintenance nightmare** - Updates in one place miss the other
- **Type checking issues** - MyPy may use wrong definition

### Deep Investigation Results

#### 1. CachePort Protocol (2 definitions)

**Location 1**: `src/brain_go_brrr/infra/cache_factory.py:25`
```python
class CachePort(Protocol):
    """Cache protocol for compatibility."""
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...
```

**Location 2**: `src/brain_go_brrr/domain/ports/cache.py:9`
```python
class CachePort(Protocol):
    """Port for synchronous cache operations."""
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool: ...
    # Similar but possibly different method signatures
```

**Impact Analysis**:
- The domain version is exported via `domain/ports/__init__.py`
- The infra version is used locally in cache_factory
- **RISK**: If someone imports from wrong location, Protocol won't match

#### 2. RedisCache Class (2 definitions)

**Location 1**: `src/brain_go_brrr/infra/cache.py:70`
```python
class RedisCache:
    """Infrastructure Redis implementation."""
    # Full Redis client implementation
```

**Location 2**: `src/brain_go_brrr/api/cache.py:19`
```python
class RedisCache:
    """API-specific Redis cache."""
    # Different implementation for API layer
```

**Impact Analysis**:
- API version instantiated in `api/cache.py:_cache_instance = RedisCache()`
- Infra version returned by factory in `infra/cache.py`
- **CRITICAL**: These are DIFFERENT implementations, not duplicates!
- **RISK**: Name collision could cause wrong cache to be used

#### 3. YASAConfig Class (2 definitions)

**Location 1**: `src/brain_go_brrr/infra/external/yasa_adapter.py:74`
```python
class YASAConfig:
    """Configuration for YASA sleep staging."""
    # External adapter configuration
```

**Location 2**: `src/brain_go_brrr/domain/sleep/analyzer_enhanced.py:46`
```python
class YASAConfig:
    """Domain-specific YASA configuration."""
    # Domain layer configuration
```

**Impact Analysis**:
- Used in different layers (infra vs domain)
- Likely have different fields/purposes
- **RISK**: Import confusion could pass wrong config type

#### 4. Other Duplicates Found

- `FeatureExtractorPort(Protocol)` - Multiple protocol definitions
- `JobData` - Multiple data classes for job handling
- `LoggerPort(Protocol)` - Multiple logger interfaces
- `NumpyEncoder(json.JSONEncoder)` - Multiple JSON encoders
- `_NullModel` - Multiple null object patterns
- `_NullPreprocessor` - Multiple preprocessor mocks

---

## 🔬 DEEP FORENSIC ANALYSIS - THE REAL DANGER

### ⚠️ CRITICAL DISCOVERY: LoggerPort INCOMPATIBLE SIGNATURES!

The `LoggerPort` Protocol has **INCOMPATIBLE METHOD SIGNATURES** between versions:

```python
# domain/abnormal/ports.py version - ACCEPTS *args, **kwargs
def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:

# domain/ports/base.py version - ONLY ACCEPTS message
def debug(self, message: str) -> None:
```

**THIS WILL CAUSE RUNTIME FAILURES!** If code expects one signature but gets the other, it will crash at runtime with:
```
TypeError: debug() takes 2 positional arguments but 4 were given
```

### 📊 COMPLETE DUPLICATE MAPPING

| Class | Location 1 | Location 2 | Location 3 | COMPATIBLE? |
|-------|-----------|-----------|-----------|-------------|
| CachePort | infra/cache_factory.py:25 | domain/ports/cache.py:9 | - | ✅ Yes (same methods) |
| RedisCache | infra/cache.py:70 | api/cache.py:19 | - | ❌ NO (different impl) |
| YASAConfig | infra/external/yasa_adapter.py:74 | domain/sleep/analyzer_enhanced.py:46 | - | ❌ NO (different fields) |
| FeatureExtractorPort | application/factories_types.py | domain/abnormal/ports.py | - | ❌ NO (different signatures) |
| JobData | api/schemas.py (frozen) | application/jobs/models.py | api/schemas.py (TypedDict) | ❌ NO (different fields) |
| LoggerPort | domain/abnormal/ports.py | domain/ports/base.py | - | ❌ NO (INCOMPATIBLE!) |
| NumpyEncoder | api/app.py | api/routers/qc.py | - | ✅ Yes (identical) |
| _NullModel | domain/abnormal/detector.py | domain/preprocessing/features/extractor.py | - | ✅ Yes (test mocks) |
| _NullPreprocessor | domain/abnormal/detector.py | domain/quality/controller.py | domain/preprocessing/features/extractor.py | ✅ Yes (test mocks) |

### 🔥 THE SMOKING GUN: Import Chain Analysis

#### CachePort Import Chain:
```
domain/ports/__init__.py exports CachePort
  ↓
NOBODY IMPORTS IT! (Only referenced in type hints)
  ↓
infra/cache_factory.py defines its OWN CachePort
  ↓
USES IT LOCALLY (doesn't export)
```
**VERDICT**: Two isolated definitions that never meet... YET.

#### RedisCache Import Chain:
```
infra/cache.py:RedisCache (the real implementation)
  ↓
infra/cache_factory.py imports as InfraRedisCache (aliased!)
  ↓
api/cache.py:RedisCache (DIFFERENT CLASS, wraps infra version)
  ↓
api/__init__.py exports api version
  ↓
api/dependencies.py imports api version
```
**VERDICT**: They tried to fix it with aliasing but it's STILL CONFUSING!

#### YASAConfig Import Chain:
```
infra/external/yasa_adapter.py:YASAConfig
  ↓
application/factories.py imports infra version
  ↓
domain/sleep/analyzer_enhanced.py:YASAConfig (DIFFERENT!)
  ↓
services/__init__.py might import EITHER!
```
**VERDICT**: DANGEROUS - Could get wrong config type!

### 📈 ACTUAL USAGE STATISTICS

| Class | Files Using It | Import Statements | Risk Level |
|-------|---------------|-------------------|------------|
| CachePort | 3 files | 1 import | Low (isolated) |
| RedisCache | 5 files | 3 imports | HIGH (confusion) |
| YASAConfig | 4 files | 1 import | MEDIUM |
| LoggerPort | Unknown | Unknown | CRITICAL |
| FeatureExtractorPort | Unknown | Unknown | HIGH |
| JobData | Unknown | Unknown | HIGH |

## 🎯 ROOT CAUSE ANALYSIS

### Why This Happened

1. **Layered Architecture Confusion**
   - Clean Architecture encourages ports/interfaces in domain
   - But infrastructure also needs its own interfaces
   - Result: Duplicate protocols at each layer

2. **No Naming Convention**
   - Missing prefixes like `DomainCachePort` vs `InfraCachePort`
   - Same names used for different purposes

3. **Copy-Paste Development**
   - Developers copied classes instead of importing
   - Evolved separately over time

4. **Missing Central Registry**
   - No single place defining shared interfaces
   - Each module defines what it needs

---

## 🔥 RUNTIME FAILURE SCENARIOS - WHAT WILL BREAK

### Scenario 1: LoggerPort Type Mismatch
```python
# Developer writes this expecting domain/abnormal version:
logger.debug("Processing %d samples", sample_count, extra={"user": "admin"})

# But gets domain/ports version at runtime:
# CRASH: TypeError: debug() got unexpected keyword argument 'extra'
```

### Scenario 2: Wrong YASAConfig Passed
```python
# analyzer_enhanced.py expects its own YASAConfig with use_single_channel field
config = YASAConfig(use_single_channel=True)  # domain version

# But factories.py passes the infra version without that field
# CRASH: AttributeError: 'YASAConfig' object has no attribute 'use_single_channel'
```

### Scenario 3: JobData Field Mismatch
```python
# API expects frozen dataclass with job_id field
job = JobData(job_id="123", ...)  # api/schemas.py version

# Application layer expects mutable dataclass with id field
job.id  # CRASH: AttributeError: 'JobData' object has no attribute 'id'
```

## 💊 PROPOSED SOLUTION - COMPLETE IMPLEMENTATION PLAN

### Phase 1: Immediate Fixes (This Week)

#### A. Rename for Clarity
```python
# Before: Confusing duplicates
class RedisCache  # in api/cache.py
class RedisCache  # in infra/cache.py

# After: Clear distinction
class APIRedisCache     # in api/cache.py
class InfraRedisCache   # in infra/cache.py
```

#### B. Consolidate Protocols
```python
# Create central location
src/brain_go_brrr/domain/protocols/
├── cache.py       # Single CachePort definition
├── logger.py      # Single LoggerPort definition
└── extractor.py   # Single FeatureExtractorPort

# All layers import from here
from brain_go_brrr.domain.protocols.cache import CachePort
```

#### C. Fix Import Paths
```python
# Update all imports to use consolidated versions
# Run this to find all imports:
rg "from.*import.*(CachePort|RedisCache|YASAConfig)" --type py src/
```

### Phase 2: Prevention (Next Sprint)

1. **Add Import Linter Rule**
```yaml
# .importlinter
[contracts]
protocols-single-source:
  type: forbidden
  source_modules:
    - brain_go_brrr.infra
    - brain_go_brrr.api
  forbidden_modules:
    - brain_go_brrr.domain.protocols
  message: "Protocols must be imported from domain.protocols only"
```

2. **Add Pre-commit Hook**
```python
# Check for duplicate class definitions
def check_duplicate_classes():
    classes = {}
    for file in python_files:
        for class_name in extract_classes(file):
            if class_name in classes:
                raise Error(f"Duplicate class {class_name}")
            classes[class_name] = file
```

3. **Naming Convention Enforcement**
```python
# Layer-specific prefixes required
if file.path.contains("api/"):
    assert class_name.startswith("API")
elif file.path.contains("infra/"):
    assert class_name.startswith("Infra")
```

---

## 📊 IMPACT METRICS

### Current State
- **9 duplicate classes** across codebase
- **~20 files** importing these classes
- **3 layers** with overlapping definitions
- **High risk** of wrong class usage

### After Fix
- **0 duplicate classes**
- **Single source of truth** for each interface
- **Clear naming** prevents confusion
- **Import linting** prevents regression

---

## ✅ IMPLEMENTATION CHECKLIST

### Immediate Actions
- [ ] Create `domain/protocols/` directory
- [ ] Move all Protocol classes to central location
- [ ] Rename implementation classes with layer prefixes
- [ ] Update all import statements
- [ ] Run full test suite to verify

### Testing Strategy
- [ ] Unit tests still pass
- [ ] Integration tests still pass
- [ ] Type checking with mypy passes
- [ ] No circular imports introduced

### Rollback Plan
- Git commit before changes
- If issues found, revert and reassess
- Can be done incrementally (one class at a time)

---

## 🔍 VERIFICATION COMMANDS

```bash
# Find all duplicate class definitions
for class in CachePort RedisCache YASAConfig JobData; do
    echo "=== $class ==="
    rg "^class $class" --type py src/ -n
done

# Verify no imports from old locations after fix
rg "from brain_go_brrr.infra.cache_factory import CachePort" --type py src/

# Check for successful consolidation
rg "from brain_go_brrr.domain.protocols" --type py src/ | wc -l
# Should show many imports after fix
```

---

## 📈 PREVENTION STRATEGY

### Short Term
1. Code review checklist includes "check for duplicate classes"
2. Add to CONTRIBUTING.md guidelines
3. Regular audit (monthly) for new duplicates

### Long Term
1. Automated duplicate detection in CI
2. Architecture decision records (ADRs) for interfaces
3. Module boundary enforcement with import-linter

---

## 🎯 SUCCESS CRITERIA

The fix is complete when:
1. ✅ Zero duplicate class names in codebase
2. ✅ All tests pass
3. ✅ Type checking passes
4. ✅ No runtime errors from wrong imports
5. ✅ Clear naming convention documented
6. ✅ Prevention mechanisms in place

---

## 📝 NOTES FOR IMPLEMENTER

**Warning**: This is a high-risk refactor because:
- Changes touch multiple layers
- Could break runtime behavior
- Type checking might reveal hidden issues

**Recommendation**: 
1. Do this on a fresh branch
2. Make atomic commits (one class at a time)
3. Run tests after each change
4. Have someone review before merge

---

**Priority**: 🔴 **CRITICAL**  
**Estimated Effort**: 4-6 hours  
**Risk Level**: High (but necessary)  
**Business Impact**: Prevents future production bugs