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

### 1. LoggerPort Protocol (INCOMPATIBLE SIGNATURES)

**Problem**: Two LoggerPort definitions with DIFFERENT signatures causing TypeErrors

**Location 1**: `src/brain_go_brrr/domain/ports/base.py:16`
```python
class LoggerPort(Protocol):
    def debug(self, message: str) -> None:  # Only message param
    def info(self, message: str) -> None:
    def warning(self, message: str) -> None:
    def error(self, message: str) -> None:
```

**Location 2**: `src/brain_go_brrr/domain/abnormal/ports.py:67`
```python
class LoggerPort(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # Flexible params
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
```

**Current Usage**:
- `infra/adapters/logger_adapter.py` imports from `domain.abnormal.ports`
- No imports found for `domain.ports.base.LoggerPort`

**Fix Plan** (VERIFIED):
1. Create unified LoggerPort in `domain/protocols/logger.py` with flexible signature (*args, **kwargs)
2. Delete both existing definitions
3. Update single import in `logger_adapter.py`: `from ...domain.protocols.logger import LoggerPort`
4. Run mypy to verify no type errors

---

### 2. RedisCache Name Collision

**Problem**: Two different RedisCache classes cause import confusion

**Location 1**: `src/brain_go_brrr/api/cache.py:19`
```python
class RedisCache:  # NOT ASYNC - just delegates to infra
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cache = get_infra_cache()  # Delegates to infra/cache.py

    def get(self, key: str) -> dict[str, Any] | None:  # Sync method
    def set(self, key: str, value: dict[str, Any], ttl: int | None = None) -> bool:
```

**Location 2**: `src/brain_go_brrr/infra/cache.py:70`
```python
class RedisCache:  # Actual Redis implementation
    def __init__(self, pool: RedisConnectionPool | None = None) -> None:
        self.pool = pool or get_redis_pool()

    def get(self, key: str) -> Any:
    def set(self, key: str, value: Any, expiry: int | None = None) -> bool:
```

**Fix Plan** (CORRECTED):
1. Rename API class to `APIRedisCache` or `AnalysisCache` (NOT AsyncRedisCache - it's not async!)
2. Keep infra version as `RedisCache` (the actual implementation)
3. Update API imports: `from brain_go_brrr.api.cache import APIRedisCache`
4. Update get_cache() in api/cache.py to return `APIRedisCache`

---

### 3. YASAConfig Classes with DIFFERENT Fields

**Problem**: Two YASAConfig classes with COMPLETELY DIFFERENT fields and purposes

**Location 1**: `src/brain_go_brrr/domain/sleep/analyzer_enhanced.py:46`
```python
@dataclass
class YASAConfig:
    # Domain-focused config
    use_consensus: bool = True
    use_single_channel: bool = False
    eeg_channels_preference: list[str] | None = None  # Channel preferences
    epoch_length: float = 30.0  # Epoch parameters
    resample_freq: float = 100.0
    apply_smoothing: bool = True
    smoothing_window_min: float = 7.5
```

**Location 2**: `src/brain_go_brrr/infra/external/yasa_adapter.py:74`
```python
@dataclass
class YASAConfig:
    # Adapter-specific config
    use_consensus: bool = True
    eeg_backend: str = "lightgbm"  # Backend selection
    eog_backend: str = "lightgbm"
    emg_backend: str = "lightgbm"
    freq_broad: tuple[float, float] = (0.5, 35.0)
    auto_alias: bool = True  # Channel aliasing for Sleep-EDF
```

**Fix Plan** (REVISED):
1. Keep domain version as `YASAConfig` (domain/sleep/analyzer_enhanced.py)
2. Rename infra version to `YASAAdapterConfig` (different purpose, different fields)
3. Update yasa_adapter.py class definition: `class YASAAdapterConfig:`
4. Consider adding converter method: `YASAAdapterConfig.from_domain_config(YASAConfig)`

---

### 4. FeatureExtractorPort & Related Port Duplicates

**Problem**: Multiple port definitions with overlapping responsibilities

**FeatureExtractorPort**:
- `src/brain_go_brrr/application/factories_types.py:94`
- `src/brain_go_brrr/domain/abnormal/ports.py:51`

**Related Ports**:
- `src/brain_go_brrr/domain/ports/base.py:36` - EEGModelPort
- `src/brain_go_brrr/domain/ports/base.py:60` - PreprocessorPort
- `src/brain_go_brrr/domain/abnormal/ports.py:19` - EEGPreprocessorPort

**Fix Plan**:
1. Consolidate FeatureExtractorPort to single location (domain/ports/)
2. Review if EEGModelPort and FeatureExtractorPort should merge
3. Unify PreprocessorPort and EEGPreprocessorPort
4. Update all imports to use consolidated versions

---

## 📝 P2 ISSUES - LOWER PRIORITY CLEANUP

### 5. Documentation Unsafe torch.load

**Location**: `docs/TRAINING.md:246`
```python
# UNSAFE - will fail CI/CD
checkpoint = torch.load("output/tuab_*/best_model.pt")
```

**Problem**: Missing `weights_only` parameter causes CI/CD failures

**Fix Plan**:
1. Update example to use safe loading:
   ```python
   # For pure tensor data
   checkpoint = torch.load("output/tuab_*/best_model.pt", weights_only=True)

   # OR for complex objects with justification
   checkpoint = torch.load("output/tuab_*/best_model.pt", weights_only=False)  # nosec:weights_only - contains optimizer state
   ```
2. Reference `brain_go_brrr.infra.safe_load` wrapper in docs

### 6. PyTorch Lightning in Dependencies

**Location**: `pyproject.toml:70`
```toml
"lightning>=2.1.0",  # FORBIDDEN - causes training hangs
```

**Problem**: Still in dependencies despite critical bug causing training to hang

**Current State**:
- No active imports in codebase (verified with grep)
- Only mentioned in comments/warnings
- Present in uv.lock (auto-generated)

**Fix Plan**:
1. Remove line 70 from pyproject.toml
2. Run `uv sync` to update lock file
3. Add guard in `__init__.py` if needed:
   ```python
   try:
       import lightning
       raise ImportError("PyTorch Lightning is banned due to training hang bug. Use pure PyTorch.")
   except ImportError:
       pass
   ```

### 7. Incomplete Probe Migration

**Problem**: `EEGPTProbe` still directly instantiated instead of using `ProbeFactory`

**Current Usage** (3 files):
1. `application/pipeline/eegpt_orchestration.py:67`
   ```python
   probe = EEGPTProbe(backbone=model, n_classes=2, architecture="linear")
   ```

2. `application/use_cases/tasks/abnormality_detection.py:13`
   ```python
   from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
   # Usage in function
   ```

3. `application/use_cases/tasks/enhanced_abnormality_detection.py:105`
   ```python
   probe = EEGPTProbe(
       backbone=eegpt_model,
       n_classes=2,
       architecture="two_layer"
   )
   ```

**Fix Plan**:
1. Replace all direct instantiations with:
   ```python
   from brain_go_brrr.infra.ml_models.probe_factory import ProbeFactory
   probe = ProbeFactory.create(
       task="abnormality",
       backbone=model,
       architecture="linear"  # or "two_layer"
   )
   ```
2. Update imports
3. Verify tests still pass

---

## 🎯 IMPLEMENTATION ORDER

1. **LoggerPort** (30min) - Single import to fix, prevents TypeErrors
2. **RedisCache** (30min) - Simple rename to APIRedisCache
3. **YASAConfig** (45min) - Rename to YASAAdapterConfig, may need converter
4. **FeatureExtractorPort** (45min) - Architectural clarity needed
5. **Documentation** (15min) - One-line fix in TRAINING.md:246
6. **Remove Lightning** (15min) - Delete one line from pyproject.toml
7. **Probe Migration** (2hr) - Update 3 files to use ProbeFactory

---

## ✅ DEFINITION OF DONE

- [ ] LoggerPort: Single protocol in domain/protocols/logger.py, zero duplicates
- [ ] RedisCache: API class renamed to APIRedisCache, no name collisions
- [ ] YASAConfig: Infra renamed to YASAAdapterConfig, no field confusion
- [ ] FeatureExtractorPort: Single definition in domain/ports/
- [ ] torch.load: TRAINING.md:246 uses weights_only parameter
- [ ] Lightning: Removed from pyproject.toml:70, uv.lock updated
- [ ] EEGPTProbe: All 3 usages migrated to ProbeFactory.create()
- [ ] CI/CD: All branches green, mypy passing

---

## 🚀 QUICK WINS (Do First)

**5-Minute Fixes**:
1. **Documentation torch.load** - Edit 1 line in TRAINING.md:246
2. **Remove Lightning** - Delete 1 line in pyproject.toml:70

**30-Minute Fixes**:
3. **LoggerPort** - Create domain/protocols/logger.py, update 1 import
4. **RedisCache** - Rename class to APIRedisCache, update imports

**45-Minute Fixes**:
5. **YASAConfig** - Rename to YASAAdapterConfig, test adapter still works
6. **FeatureExtractorPort** - Consolidate ports, update imports

**2-Hour Task**:
7. **Probe Migration** - Update 3 files to use ProbeFactory pattern

---

## 📊 IMPLEMENTATION DETAILS

### LoggerPort Unification
```bash
# Step 1: Create unified protocol
mkdir -p src/brain_go_brrr/domain/protocols
touch src/brain_go_brrr/domain/protocols/__init__.py
# Create logger.py with flexible signature

# Step 2: Update single import
sed -i 's/from.*abnormal.ports import LoggerPort/from ...domain.protocols.logger import LoggerPort/' \
  src/brain_go_brrr/infra/adapters/logger_adapter.py

# Step 3: Delete old definitions
# Remove lines 16-33 from domain/ports/base.py
# Remove lines 67-84 from domain/abnormal/ports.py
```

### RedisCache Rename
```bash
# Step 1: Rename class
sed -i 's/class RedisCache:/class APIRedisCache:/' src/brain_go_brrr/api/cache.py

# Step 2: Update function return type
sed -i 's/-> RedisCache/-> APIRedisCache/' src/brain_go_brrr/api/cache.py

# Step 3: Update instantiation
sed -i 's/= RedisCache()/= APIRedisCache()/' src/brain_go_brrr/api/cache.py

# Step 4: Update imports in API routers
grep -r "from.*api.cache import RedisCache" src/ | cut -d: -f1 | \
  xargs sed -i 's/import RedisCache/import APIRedisCache/'
```

### YASAConfig Rename
```bash
# Step 1: Rename infra config
sed -i 's/class YASAConfig:/class YASAAdapterConfig:/' \
  src/brain_go_brrr/infra/external/yasa_adapter.py

# Step 2: Update type hints in same file
sed -i 's/config: YASAConfig/config: YASAAdapterConfig/' \
  src/brain_go_brrr/infra/external/yasa_adapter.py

# Step 3: Add converter method (manual edit needed)
```
