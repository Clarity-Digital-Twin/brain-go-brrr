# 🚨 TECHNICAL DEBT - Priority Issues Requiring Resolution

**Created**: September 4, 2025  
**Updated**: September 4, 2025 (Critical Audit from First Principles)
**Status**: Active - Requires Immediate Attention  
**Focus**: Critical runtime issues from duplicate classes and API dimensionality mismatches

## ⚠️ CRITICAL AUDIT FINDINGS

After thorough first-principles analysis and **reviewing EEGPT literature**, I've identified critical issues:

1. **🔥 EEGPT API violates paper specification: passing 512-dim instead of 2048-dim (4×512)** - GUARANTEED CRASH
2. **✅ CONFIRMED: 6 duplicate class definitions with 2 having incompatible signatures** 
3. **📚 Documentation teaches unsafe torch.load that fails CI/CD**
4. **⚡ PyTorch Lightning remains in dependencies despite critical bug**
5. **🔄 Probe migration incomplete - deprecated code still in production**

**Paper Evidence**: EEGPT paper (Tables 12-13, Line 615) explicitly states the model outputs **"4 × 512 dimensional features"** that should be **flattened to 2048** for linear probing. Our API is only using 1 summary token (512 dims) instead of all 4 (2048 dims).

**Bottom Line**: The API dimensionality mismatch is a **VIOLATION OF THE EEGPT ARCHITECTURE** per the original paper, not just a bug.

## 🚨 PRIORITY RANKING (P0 = CRASH TODAY)

| Priority | Issue | Impact | Time to Fix |
|----------|-------|--------|-------------|
| **P0** | EEGPT API Endpoints (512→probe expecting 2048) | Runtime crash when API uses probes | 30 min |
| **P0** | Sleep Probe Trainer (512→probe expecting 2048) | Runtime crash if trainer is run | 15 min |
| **P0** | LoggerPort incompatible signatures | TypeError if called with *args/**kwargs | 30 min |
| **P1** | CLI streaming wrong dims | SSOT violation (no crash - doesn't use probe) | 15 min |
| **P1** | Domain services wrong dims | SSOT violation (no crash unless wired to probe) | 30 min |
| **P1** | RedisCache name collision | Import confusion risk | 30 min |
| **P1** | YASAConfig duplicate definitions | Wrong config = wrong results | 30 min |
| **P2** | JobData duplicates (intentional) | DTO vs domain model - keep but document | 0 min |
| **P2** | Documentation unsafe torch.load | CI/CD failures for new devs | 15 min |
| **P2** | Lightning in dependencies | Accidental usage risk | 15 min |
| **P2** | Probe migration incomplete | Tech debt accumulation | 2 hours |

---

## 🔴 CRITICAL ISSUE #1: EEGPT API Dimensionality Mismatch

### The Problem
**The EEGPT API routes are passing 512-dim features to probes expecting 2048-dim input**

API endpoints call `eegpt_model.extract_features(window_data, channel_names)` which defaults to `summary=True` returning 512 dimensions. But all probes (SleepStageProbe, AbnormalityProbe) expect 2048-dim input (4×512 flattened).

### 📚 EEGPT Paper Confirmation
From the EEGPT literature (`literature/markdown/EEGPT/EEGPT.md`):
- **Line 193**: "sets S learnable summary tokens (similar to [CLS] token)"
- **Line 297**: "large model, featuring an 8-layer, 512-embedding dimension, and **4 summary tokens**"
- **Line 597, 612**: Output structure shows "4 × 512" features that get flattened for linear probing
- **Line 615**: "The eegpt-encoder maps 64-length window segments to **4 (number of summary tokens) × 512 dimensional features**"

**Paper confirms**: EEGPT outputs 4 summary tokens × 512 dims = 2048 total dimensions for downstream tasks

### Evidence in Our Code
- **API call**: `src/brain_go_brrr/api/routers/eegpt.py:138`
  ```python
  features = eegpt_model.extract_features(window_data, channel_names)  # Returns 512 (summary=True default)
  ```
- **Wrapper default**: `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py:146`
  ```python
  def extract_features(self, x: torch.Tensor, chan_ids: torch.Tensor | None = None, summary: bool = True)
  # summary=True returns (B, 512), summary=False returns (B, 4, 512)
  ```
- **All probes expect 2048**: `src/brain_go_brrr/infra/ml_models/linear_probe.py`
  ```python
  class LinearProbeHead(nn.Module):
      def __init__(self, input_dim: int = 2048,  # 4 summary tokens x 512 dims
  ```
- **Tests mask the issue**: Tests mock the model to return 2048-dim arrays directly

### Runtime Failure Scenario
```python
# API endpoint gets 512-dim features (WRONG - only 1 summary token)
features = eegpt_model.extract_features(data)  # shape: (512,)

# Passes to probe expecting 2048 (4 summary tokens × 512)
probe = SleepStageProbe()  # Expects 2048 input per paper
result = probe.predict_proba(features)  # CRASH: RuntimeError: size mismatch (expected 2048, got 512)
```

### The Fix (Aligned with Paper)
```python
# Change all API calls to match EEGPT paper specification:
features = eegpt_model.extract_features(window_data, channel_names, summary=False)  # Returns (4, 512)
features = features.flatten()  # Now (2048,) as paper specifies
# This matches Table 12/13 in paper: "4 × 512" → "flatten,linear"
```

---

## 🔴 CRITICAL ISSUE #2: Duplicate Class Definitions

### The Problem
We have **6 unique classes with duplicate definitions** across different modules. Not all are dangerous:
- **4 DANGEROUS duplicates** that can cause runtime errors (LoggerPort, CachePort, RedisCache, YASAConfig)
- **2 INTENTIONAL patterns** that are OK (JobData DTO vs entity, test helpers)
- **Import confusion** - Wrong class gets imported = wrong behavior
- **Type errors** - LoggerPort incompatible signatures = runtime crash

### Deep Investigation Results

**⚠️ CRITICAL**: Not all duplicates are dangerous. Some are intentional architectural patterns.

#### DANGEROUS DUPLICATES (Must Fix)

##### 1. LoggerPort Protocol (INCOMPATIBLE - Will Crash)

**Location 1**: `src/brain_go_brrr/domain/ports/base.py:16`
```python
class LoggerPort(Protocol):
    def debug(self, message: str) -> None:  # Only accepts message
    def info(self, message: str) -> None:
```

**Location 2**: `src/brain_go_brrr/domain/abnormal/ports.py:67`
```python
class LoggerPort(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # Accepts varargs
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
```

**Impact**: If code expects varargs version but gets base version → TypeError at runtime

##### 2. CachePort Protocol (Interface Drift)
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

##### 3. RedisCache Class (Name Collision Risk)

**Location 1**: `src/brain_go_brrr/infra/cache.py:70` - Full Redis implementation
**Location 2**: `src/brain_go_brrr/api/cache.py:19` - API wrapper

**Impact**: Name collision could cause wrong cache to be imported

##### 4. YASAConfig Class (Different Fields)

**Location 1**: `src/brain_go_brrr/infra/external/yasa_adapter.py:74` - External adapter config
**Location 2**: `src/brain_go_brrr/domain/sleep/analyzer_enhanced.py:46` - Domain config

**Impact**: Wrong config type could be passed, missing expected fields

#### INTENTIONAL DUPLICATES (Keep but Document)

##### 1. JobData (API DTO vs Domain Model)

**Location 1**: `src/brain_go_brrr/api/schemas.py:21` - API Data Transfer Object
```python
class JobData:  # Frozen dataclass with job_id, analysis_type, etc.
```

**Location 2**: `src/brain_go_brrr/application/jobs/models.py:34` - Domain Entity  
```python
class JobData:  # Domain model with id, type, etc.
```

**Why it's OK**: Standard pattern - API layer has DTOs, domain has entities. Keep both but ensure clear mapping.

##### 2. Test Helpers (_NullModel, _NullPreprocessor)

Multiple null object implementations for testing - harmless test utilities

##### 3. FeatureExtractorPort Protocol

**Location 1**: `src/brain_go_brrr/domain/abnormal/ports.py:51`
**Location 2**: `src/brain_go_brrr/application/factories_types.py:94`

**Why it's OK**: Different layers defining their own contracts - acceptable in ports & adapters pattern

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

### PHASE 0: EMERGENCY FIX (DO NOW - 30 MINUTES)

#### Step 0: Fix EEGPT API Dimensionality IMMEDIATELY
```bash
# 0.1 - Fix all API extract_features calls
sed -i 's/extract_features(window_data, channel_names)/extract_features(window_data, channel_names, summary=False)/g' \
    src/brain_go_brrr/api/routers/eegpt.py

# 0.2 - Add flattening after feature extraction
# Manual edit needed to add .flatten() after each extract_features call

# 0.3 - Run API tests to verify
pytest tests/unit/api/routers/test_eegpt.py -xvs

# 0.4 - Update test mocks to match new behavior
# Tests should mock (4, 512) shape that gets flattened to (2048,)
```

### PHASE 1: CRITICAL FIXES (DO TODAY - 2 HOURS)

#### Step 1: Fix LoggerPort IMMEDIATELY (30 min)
```bash
# 1.1 - DELETE the incompatible version
rm src/brain_go_brrr/domain/abnormal/ports.py

# 1.2 - Update imports in domain/abnormal/detector.py
sed -i 's/from \.ports import LoggerPort/from brain_go_brrr.domain.ports import LoggerPort/g' \
    src/brain_go_brrr/domain/abnormal/detector.py

# 1.3 - Verify single definition remains
rg "class LoggerPort" --type py src/
# Should show ONLY domain/ports/base.py
```

#### Step 2: Rename RedisCache Classes (30 min)
```bash
# 2.1 - Rename API version
sed -i 's/class RedisCache:/class APIRedisCache:/g' \
    src/brain_go_brrr/api/cache.py

# 2.2 - Update API imports
sed -i 's/RedisCache/APIRedisCache/g' \
    src/brain_go_brrr/api/__init__.py \
    src/brain_go_brrr/api/dependencies.py

# 2.3 - The infra version stays as RedisCache (it's the real one)
# Already aliased correctly in cache_factory.py as InfraRedisCache
```

#### Step 3: Fix YASAConfig Naming (30 min)
```bash
# 3.1 - Rename domain version
sed -i 's/class YASAConfig:/class EnhancedYASAConfig:/g' \
    src/brain_go_brrr/domain/sleep/analyzer_enhanced.py

# 3.2 - Update all references in same file
sed -i 's/YASAConfig(/EnhancedYASAConfig(/g' \
    src/brain_go_brrr/domain/sleep/analyzer_enhanced.py

# 3.3 - Infra version stays as YASAConfig (external adapter)
```

#### Step 4: Fix JobData Classes (30 min)
```bash
# 4.1 - Rename API version to be explicit
sed -i 's/class JobData:/class APIJobData:/g' \
    src/brain_go_brrr/api/schemas.py

# 4.2 - Keep TypedDict version as JobDataDict (already named correctly)

# 4.3 - Application version stays as JobData (domain model)
```

### PHASE 2: CONSOLIDATION (DO TOMORROW - 4 HOURS)

#### Step 5: Create Protocol Registry
```bash
# 5.1 - Create protocols directory
mkdir -p src/brain_go_brrr/domain/protocols

# 5.2 - Move all protocols to central location
cat > src/brain_go_brrr/domain/protocols/__init__.py << 'EOF'
"""Central protocol definitions to prevent duplicates."""

from .cache import CachePort, AsyncCachePort
from .logger import LoggerPort
from .extractor import FeatureExtractorPort
from .model import ModelPort
from .preprocessor import PreprocessorPort

__all__ = [
    "CachePort",
    "AsyncCachePort", 
    "LoggerPort",
    "FeatureExtractorPort",
    "ModelPort",
    "PreprocessorPort",
]
EOF

# 5.3 - Create each protocol file
cat > src/brain_go_brrr/domain/protocols/logger.py << 'EOF'
"""Logger protocol - single source of truth."""
from typing import Protocol, Any

class LoggerPort(Protocol):
    """Unified logger interface."""
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with optional formatting."""
        ...
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with optional formatting."""
        ...
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with optional formatting."""
        ...
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with optional formatting."""
        ...
EOF
```

#### Step 6: Update All Imports
```bash
# 6.1 - Find all files that need updating
rg "from.*domain\.(ports|abnormal\.ports)" --type py src/ -l > files_to_update.txt

# 6.2 - Update each file to use central protocols
while read -r file; do
    sed -i 's/from brain_go_brrr\.domain\.ports import/from brain_go_brrr.domain.protocols import/g' "$file"
    sed -i 's/from brain_go_brrr\.domain\.abnormal\.ports import/from brain_go_brrr.domain.protocols import/g' "$file"
done < files_to_update.txt

# 6.3 - Verify no duplicate definitions remain
for class in CachePort LoggerPort FeatureExtractorPort ModelPort PreprocessorPort; do
    echo "Checking $class..."
    count=$(rg "^class $class" --type py src/ | wc -l)
    if [ "$count" -gt 1 ]; then
        echo "ERROR: $class still has $count definitions!"
        rg "^class $class" --type py src/ -n
    fi
done
```

### PHASE 3: VALIDATION & TESTING (1 HOUR)

#### Step 7: Run Type Checking
```bash
# 7.1 - Run mypy to catch type errors
make typecheck

# 7.2 - Fix any import errors mypy finds
# Common fixes:
# - Update type hints to use new names
# - Add missing imports
# - Remove old import statements
```

#### Step 8: Run All Tests
```bash
# 8.1 - Unit tests
make test

# 8.2 - Integration tests  
make test-integration

# 8.3 - If any fail, check for:
# - Wrong class being imported
# - Method signature mismatches
# - Missing attributes
```

### PHASE 4: PREVENTION (NEXT WEEK - 2 HOURS)

#### Step 9: Add Duplicate Detection
```bash
# 9.1 - Create pre-commit hook
cat > .pre-commit-hooks/check-duplicate-classes.py << 'EOF'
#!/usr/bin/env python3
"""Prevent duplicate class definitions."""
import sys
import re
from pathlib import Path

def find_duplicate_classes():
    classes = {}
    duplicates = []
    
    for py_file in Path("src").rglob("*.py"):
        content = py_file.read_text()
        for match in re.finditer(r'^class (\w+)', content, re.MULTILINE):
            class_name = match.group(1)
            if class_name.startswith('_'):  # Skip private classes
                continue
            
            if class_name in classes:
                duplicates.append(f"{class_name}: {classes[class_name]} and {py_file}")
            else:
                classes[class_name] = py_file
    
    if duplicates:
        print("ERROR: Duplicate class definitions found!")
        for dup in duplicates:
            print(f"  - {dup}")
        sys.exit(1)
    
    print(f"✓ Checked {len(classes)} classes - no duplicates found")

if __name__ == "__main__":
    find_duplicate_classes()
EOF

# 9.2 - Add to .pre-commit-config.yaml
cat >> .pre-commit-config.yaml << 'EOF'
  - repo: local
    hooks:
      - id: no-duplicate-classes
        name: Check for duplicate class names
        entry: python .pre-commit-hooks/check-duplicate-classes.py
        language: python
        pass_filenames: false
EOF
```

#### Step 10: Document Standards
```bash
# 10.1 - Add to CONTRIBUTING.md
cat >> CONTRIBUTING.md << 'EOF'

## Class Naming Standards

To prevent duplicate class definitions:

1. **Protocols/Interfaces**: Define in `domain/protocols/` ONLY
2. **Layer-Specific Classes**: Use prefixes:
   - API layer: `API` prefix (e.g., `APIJobData`)
   - Infrastructure: `Infra` prefix (e.g., `InfraRedisCache`)
   - Domain: No prefix (owns the base names)
3. **Configuration Classes**: Include context in name:
   - `YASAConfig` → `YASAAdapterConfig` (for adapter)
   - `YASAConfig` → `EnhancedYASAConfig` (for enhanced analyzer)
4. **Test Mocks**: Use `_` prefix and `Mock` suffix:
   - `_MockLogger`, `_MockCache`, etc.

Run `make check-duplicates` to verify no duplicates exist.
EOF
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

## 🔍 COMPREHENSIVE VERIFICATION SUITE

### Pre-Fix Baseline (RUN FIRST!)
```bash
# Capture current state before making changes
echo "=== BASELINE DUPLICATE COUNT ===" > duplicate_baseline.txt
for class in CachePort RedisCache YASAConfig JobData LoggerPort FeatureExtractorPort NumpyEncoder; do
    echo "$class: $(rg "^class $class" --type py src/ | wc -l) definitions" >> duplicate_baseline.txt
    rg "^class $class" --type py src/ -n >> duplicate_baseline.txt
done

# Save current test results
make test > test_baseline.txt 2>&1
echo "Test exit code: $?" >> test_baseline.txt

# Save current type check results  
make typecheck > typecheck_baseline.txt 2>&1
echo "Typecheck exit code: $?" >> typecheck_baseline.txt
```

### After Each Phase Verification
```bash
# Phase 1 Check - After renaming
echo "=== PHASE 1 VERIFICATION ==="
# Should show NO duplicates for renamed classes
rg "^class (RedisCache|YASAConfig|JobData|LoggerPort)" --type py src/ | grep -v "API\|Enhanced\|Infra"

# Phase 2 Check - After consolidation  
echo "=== PHASE 2 VERIFICATION ==="
# All protocols should be in domain/protocols/
ls -la src/brain_go_brrr/domain/protocols/
# Should show: cache.py, logger.py, extractor.py, model.py, preprocessor.py

# Phase 3 Check - After testing
echo "=== PHASE 3 VERIFICATION ==="
diff test_baseline.txt test_after.txt
# Should show SAME OR BETTER test results

# Final Check - Complete validation
./scripts/verify_no_duplicates.sh
```

### Automated Verification Script
```bash
cat > scripts/verify_no_duplicates.sh << 'EOF'
#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "🔍 Checking for duplicate class definitions..."

DUPLICATES=0
declare -A class_locations

# Find all class definitions
while IFS= read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    class_name=$(echo "$line" | grep -oP 'class \K\w+')
    
    # Skip private classes and test mocks
    if [[ $class_name == _* ]] || [[ $class_name == Mock* ]]; then
        continue
    fi
    
    if [[ -n "${class_locations[$class_name]:-}" ]]; then
        echo -e "${RED}❌ DUPLICATE: $class_name${NC}"
        echo "   - ${class_locations[$class_name]}"
        echo "   - $file"
        ((DUPLICATES++))
    else
        class_locations[$class_name]=$file
    fi
done < <(rg '^class \w+' --type py src/)

if [[ $DUPLICATES -eq 0 ]]; then
    echo -e "${GREEN}✅ SUCCESS: No duplicate classes found!${NC}"
    echo "   Checked ${#class_locations[@]} unique class definitions"
    exit 0
else
    echo -e "${RED}❌ FAILED: Found $DUPLICATES duplicate class definitions${NC}"
    exit 1
fi
EOF

chmod +x scripts/verify_no_duplicates.sh
```

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

---

## 🟡 ISSUE #3: Documentation Shows Unsafe torch.load Examples

### The Problem
Documentation demonstrates `torch.load()` without `weights_only=` parameter, violating CI/CD policy.

### Evidence
- **Location**: `docs/TRAINING.md:246`
- **CI Hook**: Pre-commit will fail on unsafe torch.load usage
- **Policy**: All torch.load calls must specify `weights_only=True` or include `# nosec:weights_only`

### The Fix
Update documentation to use safe loading:
```python
# Replace all doc examples with:
checkpoint = torch.load(path, map_location='cpu', weights_only=True)
# OR use the safe_load wrapper:
from brain_go_brrr.infra.safe_load import safe_load
checkpoint = safe_load(path)
```

---

## 🟡 ISSUE #4: PyTorch Lightning in Dependencies Despite Being Forbidden

### The Problem
`pyproject.toml` includes `lightning>=2.1.0` even though usage is explicitly forbidden per CLAUDE.md.

### Evidence
- **CLAUDE.md**: "DO NOT USE PYTORCH LIGHTNING FOR TRAINING! Lightning 2.5.2 has a critical bug"
- **CI Check**: "no Lightning imports" validation exists
- **Risk**: Developers might accidentally use it

### The Fix
Move Lightning to optional dependencies or remove entirely:
```toml
# Option 1: Remove completely
# Option 2: Move to optional
[project.optional-dependencies]
deprecated = ["lightning>=2.1.0"]  # DO NOT USE - has critical bugs
```

---

## 🟡 ISSUE #5: Probe Migration Incomplete

### The Problem  
EEGPTProbe is deprecated but still actively used in production code.

### Evidence
- **Still used in**: 
  - `src/brain_go_brrr/application/use_cases/tasks/abnormality_detection.py`
  - `src/brain_go_brrr/application/pipeline/eegpt_orchestration.py`
- **ProbeFactory exists but not fully adopted**
- **Deprecation warning present but ignored**

### The Fix
Complete migration to ProbeFactory pattern:
```python
# Replace all EEGPTProbe usage with:
from brain_go_brrr.infra.ml_models.probe_factory import ProbeFactory
probe = ProbeFactory.create_probe("abnormality", model_path)
```

---

## 📊 STATUS ACCURACY AUDIT

### What Previous Docs Claimed vs Reality

| Claim | Reality | Accuracy |
|-------|---------|----------|
| "ProbeFactory 100% complete" | Still using deprecated EEGPTProbe in production | ❌ False |
| "Experiments use src components" | True - imports from src/ | ✅ True |
| "Channel routing implemented" | Implemented and tested | ✅ True |
| "TUAB collate strictness" | Enforces 19 channels correctly | ✅ True |
| "Safe torch.load everywhere" | Documentation still shows unsafe usage | ❌ False |
| "No sys.path hacks" | Clean - all removed | ✅ True |
| "CI/CD fixed 86% coverage" | Working with .coveragerc.unit | ✅ True |
| "Duplicate classes resolved" | 6 classes still have duplicates | ❌ False |

**Reality Check**: 5/8 claims accurate (62.5%). Technical debt was marked "100% complete" prematurely.

---

## 📝 EXECUTIVE SUMMARY FOR SENIOR AUDITOR

### The Problem in One Sentence
**API endpoints pass 512 dims to probes expecting 2048 (per EEGPT paper) = guaranteed runtime crash + 4 dangerous duplicate classes = TypeErrors waiting.**

### Why This Matters NOW (Ranked by Impact)
1. **EEGPT API/Sleep Trainer** - Passing 512 dims to 2048-expecting probes = **RuntimeError on every call** (P0)
2. **LoggerPort incompatibility** - Different signatures = **TypeError if varargs used** (P0)
3. **Training is SAFE** - Correctly uses 2048 dims, unaffected by bugs (Good news!)
4. **RedisCache/YASAConfig** - Name collisions = wrong implementation imported (P1)
5. **CLI/Domain services** - Wrong dims but no crash path currently (P1 - SSOT violation)
6. **JobData duplicates** - Intentional DTO vs entity pattern (P2 - document only)

### The Real Cost of NOT Fixing This
- **Production Crashes**: TypeError exceptions when wrong class loaded
- **Silent Failures**: Wrong config type = wrong analysis results
- **Developer Time**: Every new dev will hit these landmines
- **Testing Nightmare**: Tests may pass with one version, fail with another

### Implementation Risk Assessment
| Risk | Mitigation | Residual Risk |
|------|------------|---------------|
| Breaking working code | Full test suite before/after | Low |
| Import errors | Automated verification script | Low |
| Type checking failures | Run mypy at each phase | Low |
| Regression | Git bisect if issues | Low |
| Time overrun | Can pause after Phase 1 | Low |

### Why This Plan Will Work
1. **Surgical Precision**: Each change is atomic and reversible
2. **Automated Verification**: Scripts catch issues immediately
3. **Incremental Approach**: Can stop at any phase if issues
4. **Prevention Built-In**: Pre-commit hooks prevent recurrence

### The Ask
- **Time**: 6-8 hours total (can spread over 3 days)
- **Resources**: 1 developer, full test environment
- **Authority**: Permission to rename classes across codebase
- **Timing**: Do it NOW while training runs (low activity)

### Success Metrics
- Zero duplicate class names (verified by script)
- All tests pass (same or better)
- Type checking passes (no new errors)
- No runtime TypeErrors in next sprint

### What Happens If We Don't Fix This
**P0 - Will crash TODAY:**
1. Call EEGPT API endpoint → **RuntimeError: size mismatch (expected 2048, got 512)**
2. Run sleep_probe_trainer.py → **RuntimeError: Linear layer expected 2048, got 512**
3. Use LoggerPort with varargs → **TypeError: unexpected keyword argument**

**P1 - Will cause confusion:**
4. Import wrong RedisCache → Different cache behavior than expected
5. Pass wrong YASAConfig → Missing fields, incorrect analysis
6. CLI/Domain using 512 dims → Violates paper spec, loses 75% of information

**P2 - Minor issues:**
7. New dev follows docs → CI/CD rejects unsafe torch.load
8. Someone accidentally imports Lightning → Critical bug from 2.5.2

**Priority**: 🔴 **CRITICAL - DO TODAY**  
**Estimated Effort**: 6-8 hours (2 today, 4 tomorrow, 2 next week)  
**Risk Level**: High impact, Low risk with this plan  
**Business Impact**: Prevents production crashes and data corruption

## SIGN-OFF CHECKLIST FOR SENIOR AUDITOR

- [ ] Reviewed duplicate class analysis
- [ ] Understood runtime failure scenarios  
- [ ] Approved renaming strategy
- [ ] Allocated developer time
- [ ] Approved incremental approach
- [ ] Committed to prevention measures

**Senior Auditor Notes**:
_________________________________
_________________________________
_________________________________

**Approved By**: _________________ **Date**: _______
**Developer Assigned**: ____________ **Start Date**: _______

---

## 🔬 VERIFICATION COMMANDS - RUN THESE TO CONFIRM

```bash
# 1. Confirm EEGPT API dimensionality issue
grep -n "extract_features.*summary" src/brain_go_brrr/api/routers/eegpt.py
# Expected: NO RESULTS (means using default summary=True → 512 dims)

# 2. Confirm duplicate LoggerPort with incompatible signatures
diff <(grep -A5 "class LoggerPort" src/brain_go_brrr/domain/ports/base.py) \
     <(grep -A5 "class LoggerPort" src/brain_go_brrr/domain/abnormal/ports.py)
# Expected: DIFFERENT signatures (one has *args/**kwargs, other doesn't)

# 3. Count duplicate class definitions
for class in LoggerPort CachePort RedisCache YASAConfig JobData FeatureExtractorPort; do
    echo "$class: $(grep "^class $class" -r src/ --include="*.py" | wc -l) definitions"
done
# Expected: 2+ definitions for each

# 4. Check unsafe torch.load in docs
grep -n "torch.load" docs/*.md | grep -v "weights_only"
# Expected: Finds examples without weights_only parameter

# 5. Check Lightning in dependencies
grep "lightning" pyproject.toml
# Expected: Shows lightning>=2.1.0 despite being forbidden

# 6. Check deprecated probe usage
grep -r "EEGPTProbe" src/brain_go_brrr/application/ --include="*.py"
# Expected: Still used in production code
```

**If ANY of these commands show the expected results, this technical debt is REAL and URGENT.**

---

## 🎯 SINGLE SOURCE OF TRUTH: EEGPT Feature Dimensionality

### The Architecture Specification (Per Paper)
- **EEGPT Large Model**: 4 summary tokens × 512 embedding dimensions = 2048 total features
- **Linear Probing**: Flatten 4×512 → 2048 → Linear layer for classification
- **Paper Evidence**: Tables 12-13, Lines 297, 615 in `literature/markdown/EEGPT/EEGPT.md`

### Current Implementation Status

#### ✅ CORRECT Implementations (Using 2048 dims)
1. **Training Script** (`experiments/eegpt_linear_probe/train_tuab_mne.py`)
   ```python
   features = model.extract_features(x, summary=False)  # (B, 4, 512)
   features = features.flatten(1)  # (B, 2048)
   ```

2. **Pipeline Orchestration** (`application/pipeline/eegpt_orchestration.py`)
   ```python
   features = model.extract_features(mini_batch, summary=False)  # (B, 4, 512)
   ```

3. **Linear Probe Classes** (`infra/ml_models/linear_probe.py`)
   ```python
   class LinearProbeHead(nn.Module):
       def __init__(self, input_dim: int = 2048,  # Expects 2048
   ```

#### ❌ INCORRECT Implementations (Using 512 dims)
1. **API Endpoints** (`api/routers/eegpt.py` lines 138, 271)
   ```python
   features = eegpt_model.extract_features(window_data, channel_names)
   # Missing summary=False → defaults to True → returns 512 not 2048
   ```

2. **Sleep Router** (`api/routers/sleep.py` line 486)
   ```python
   features = eegpt_model.extract_features(window_data, channel_names)
   # Same issue - missing summary=False
   ```

3. **CLI** (`cli.py` lines 152, 176)
   ```python
   features_tensor = self.encoder.extract_features(data_tensor)
   # No summary parameter passed
   ```

4. **Domain Layer** (multiple files)
   - `domain/abnormal/detector.py` lines 231, 479
   - `domain/quality/controller.py` lines 202, 542
   - `domain/preprocessing/features/extractor.py` lines 124, 353
   - All missing `summary=False` parameter

#### 🔧 The Wrapper's Behavior (`infra/ml_models/eegpt_wrapper.py`)
```python
def extract_features(self, x, chan_ids=None, summary: bool = True):
    # Line 171-172: When summary=True (default)
    return features.mean(dim=1)  # Averages 4 tokens → 512 dims (WRONG for probes)
    
    # Line 174: When summary=False
    return features  # Returns (B, 4, 512) → needs flatten → 2048 (CORRECT)
```

### 📊 Impact Matrix

| Component | Current Behavior | Expected Behavior | Impact | Severity |
|-----------|-----------------|-------------------|---------|----------|
| Training (tmux) | ✅ 2048 dims | 2048 dims | **SAFE - Training is correct** | N/A |
| API Endpoints | ❌ 512 dims | 2048 dims | **CRASH - RuntimeError: size mismatch** | P0 |
| Sleep Probe Trainer | ❌ 512 dims | 2048 dims | **CRASH if run - expects 2048** | P0 |
| CLI Streaming | ❌ 512 dims | Depends on use | **No crash - doesn't use probe** | P1 |
| Domain QC Services | ❌ 512 dims | Depends on use | **No crash - uses heuristics** | P1 |
| Domain Feature Extractor | ❌ 512 dims | Depends on use | **No crash - aggregates features** | P1 |

### 🚨 The Fix Pattern

**Every `extract_features` call must specify summary parameter explicitly:**

```python
# WRONG (defaults to summary=True → 512 dims)
features = model.extract_features(data, channels)

# CORRECT for probe usage (2048 dims)
features = model.extract_features(data, channels, summary=False)
features = features.flatten(1)  # (B, 4, 512) → (B, 2048)

# CORRECT for simple averaging (512 dims)
features = model.extract_features(data, channels, summary=True)
```

### 📝 Code Locations Requiring Fix

**P0 - WILL CRASH (Fix Immediately):**
1. `src/brain_go_brrr/api/routers/eegpt.py:138` - Add `summary=False` + flatten [API endpoint → probe]
2. `src/brain_go_brrr/api/routers/eegpt.py:271` - Add `summary=False` + flatten [API endpoint → probe]
3. `src/brain_go_brrr/api/routers/sleep.py:486` - Add `summary=False` + flatten [API endpoint → probe]
4. `src/brain_go_brrr/application/training/sleep_probe_trainer.py:110` - Add `summary=False` + flatten [Direct probe usage]
5. `src/brain_go_brrr/application/training/sleep_probe_trainer.py:193` - Add `summary=False` + flatten [Direct probe usage]

**P1 - Wrong Dims but No Crash (Fix Soon for SSOT):**
6. `src/brain_go_brrr/cli.py:152` - Add `summary=False` if probe usage added [Currently no probe]
7. `src/brain_go_brrr/cli.py:176` - Add `summary=False` if probe usage added [Currently no probe]
8. `src/brain_go_brrr/domain/quality/controller.py:202` - Uses heuristics, not probe [No crash path]
9. `src/brain_go_brrr/domain/quality/controller.py:542` - Uses heuristics, not probe [No crash path]
10. `src/brain_go_brrr/domain/preprocessing/features/extractor.py:124` - Aggregates features [No probe]
11. `src/brain_go_brrr/domain/preprocessing/features/extractor.py:353` - Aggregates features [No probe]
12. `src/brain_go_brrr/infra/adapters/model_adapter.py:49` - Adapter layer [Check downstream usage]
13. `src/brain_go_brrr/domain/abnormal/detector.py:231` - Check if feeds to probe
14. `src/brain_go_brrr/domain/abnormal/detector.py:479` - Check if feeds to probe

### 🎯 CRITICAL SSOT Summary

**The Golden Rule**: 
```python
# If passing to a probe → MUST use 2048 dims
features = model.extract_features(data, channels, summary=False)  # (B, 4, 512)
features = features.flatten(1)  # (B, 2048) for probe

# If NOT using probe → Can use 512 dims  
features = model.extract_features(data, channels, summary=True)  # (B, 512) averaged
```

**Why This Matters**: 
- EEGPT outputs 4 summary tokens (like 4 different "views" of the data)
- Averaging them (summary=True) loses 75% of the information
- Linear probes were trained on all 2048 dimensions per the paper
- Using 512 dims with a 2048-expecting probe = **GUARANTEED CRASH**

**Training Status**: Your tmux training is **SAFE** - it correctly uses summary=False + flatten