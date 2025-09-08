# 🟡 P1 PRIORITY FIXES - Type Conflicts & Technical Debt

**Created**: September 7, 2025  
**Owner**: ___________________  
**Time Required**: 3 hours total (4 issues × 30-45min each)  
**Status**: 🟡 ACTIVE - FIX THIS WEEK  
**Approach**: Systematic deduplication and standardization

---

## 📋 EXECUTIVE SUMMARY

**We have 4 dangerous duplicate types causing confusion and potential runtime errors:**
1. **LoggerPort** - Incompatible signatures between domain and infra
2. **RedisCache** - Two different classes with same name  
3. **YASAConfig** - Duplicate configs with different defaults
4. **FeatureExtractorPort** - Contract confusion across layers

**Plus 3 lower-priority cleanup items:**
- Documentation shows unsafe torch.load examples
- PyTorch Lightning still in dependencies despite ban
- Incomplete probe migration from deprecated classes

**Business Impact**: Type confusion, import errors, silent failures  
**Fix Strategy**: Consolidate to single source of truth per type

---

## 🔥 P1 ISSUES - DANGEROUS DUPLICATES

### 1. LoggerPort Protocol (DUPLICATE IN DOMAIN)

**Problem**: Two LoggerPort definitions within domain layer cause confusion

**Location 1**: `src/brain_go_brrr/domain/ports/base.py:16`
```python
class LoggerPort(Protocol):
    def debug(self, message: str) -> None:
    def info(self, message: str) -> None:
```

**Location 2**: `src/brain_go_brrr/domain/abnormal/ports.py:67`  
```python
class LoggerPort(Protocol):
    # Check if signatures differ
```

**Fix Plan**:
1. Consolidate to single LoggerPort in `domain/protocols/logger.py`
2. Delete both existing definitions
3. Update all imports to use new unified location
4. Ensure consistent signature across all usages

---

### 2. RedisCache Duplicate Classes

**Problem**: Two different RedisCache implementations cause import confusion

**Location 1**: `src/brain_go_brrr/api/cache.py`
```python
class RedisCache:  # Async Redis implementation
    def __init__(self, redis_url: str):
        self.redis = redis.asyncio.from_url(redis_url)
```

**Location 2**: `src/brain_go_brrr/infra/cache.py`
```python
class RedisCache:  # Different implementation
    # Verify actual interface differences
```

**Fix Plan**:
1. Rename API version to `AsyncRedisCache` (it's async-specific)
2. Keep infra version as `RedisCache` (it's the general implementation)
3. Update API imports: `from brain_go_brrr.api.cache import AsyncRedisCache`
4. Consider unifying under single async/sync adapter pattern

---

### 3. YASAConfig Duplicate Classes

**Problem**: Two YASAConfig classes with different defaults cause confusion

**Location 1**: `src/brain_go_brrr/infra/external/config.py:6`
```python
@dataclass
class YASAConfig:
    eeg_channel: Optional[str] = None  # Auto-detect mode
```

**Location 2**: `src/brain_go_brrr/infra/external/yasa_adapter.py:25`
```python
@dataclass  
class YASAConfig:
    eeg_channel: str = "C4"  # Hardcoded default
```

**Fix Plan**:
1. Delete adapter version (line 25-30 in yasa_adapter.py)
2. Import from config: `from .config import YASAConfig`
3. Ensure auto-detect logic works with None default
4. Add validation that C4 is selected when None

---

### 4. FeatureExtractorPort Duplicates

**Problem**: Three definitions with slightly different contracts

**Locations**:
- `src/brain_go_brrr/domain/ports/base.py:42` - Basic extract method
- `src/brain_go_brrr/domain/ports/feature_extractor.py:11` - Extended with window methods
- `src/brain_go_brrr/infra/ports/__init__.py:41` - Re-exported confusion

**Fix Plan**:
1. Keep domain/ports/feature_extractor.py as SSOT (most complete)
2. Delete base.py:FeatureExtractorPort
3. Update infra/ports/__init__ to import from domain
4. Ensure all implementations follow the complete protocol

---

## 📝 P2 ISSUES - LOWER PRIORITY CLEANUP

### 5. Documentation Unsafe torch.load

**Location**: `docs/TRAINING.md` and other docs
**Problem**: Shows `torch.load(path)` without `weights_only` parameter
**Fix**: Update all examples to use `weights_only=True` or add `# nosec` comment

### 6. PyTorch Lightning in Dependencies

**Location**: `pyproject.toml`
**Problem**: Still listed despite "DO NOT USE" directive in CLAUDE.md
**Fix**: Remove from dependencies, update any imports to raise clear error

### 7. Incomplete Probe Migration

**Problem**: `EEGPTProbe` deprecated but still used in application code
**Fix**: Complete migration to `ProbeFactory.create()` pattern

---

## 🎯 IMPLEMENTATION ORDER

1. **LoggerPort** (30min) - Most likely to cause immediate TypeErrors
2. **RedisCache** (30min) - Import confusion is active pain point  
3. **YASAConfig** (30min) - Subtle bugs from different defaults
4. **FeatureExtractorPort** (45min) - Architectural clarity needed
5. **Documentation** (15min) - Quick fix for CI/CD compliance
6. **Remove Lightning** (15min) - Prevent accidental usage
7. **Probe Migration** (2hr) - Larger refactor, lower urgency

---

## ✅ DEFINITION OF DONE

- [ ] All duplicate types consolidated to single source
- [ ] No import ambiguity (unique names or single import path)
- [ ] All tests passing with consolidated types
- [ ] Documentation updated with safe examples
- [ ] PyTorch Lightning removed from dependencies
- [ ] Deprecated probe classes fully migrated
- [ ] CI/CD green on all branches

---

## 🚀 QUICK WINS (Do First)

Start with LoggerPort and RedisCache - these are most likely to cause immediate issues and are straightforward renames/deletes. The fixes are mechanical and low-risk.

The YASAConfig and FeatureExtractorPort require more careful analysis of usage patterns but will significantly improve code clarity.

Save probe migration for a dedicated refactoring session as it touches more code.