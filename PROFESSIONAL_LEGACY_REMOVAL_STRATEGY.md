# Professional Legacy Code Removal Strategy
## Using Industry-Standard Tools and Practices

### Phase 0: Discovery & Analysis (DO THIS FIRST!)

#### 0.1 Dead Code Detection with Vulture
```bash
# Find all potentially unused code
vulture src/brain_go_brrr --min-confidence 80 > vulture_report.txt

# Focus on our target modules
vulture src/brain_go_brrr/infra/ml_models/eegpt_compat.py --min-confidence 60

# Create whitelist for known false positives
vulture src/brain_go_brrr --make-whitelist > whitelist.py
```

#### 0.2 Coverage Analysis with Branch Coverage
```bash
# Run tests with branch coverage to find uncovered paths
coverage run --branch -m pytest tests/
coverage report --show-missing > coverage_report.txt
coverage html  # Generate HTML report

# Find code paths that are NEVER executed
coverage report -m | grep "0%"
```

#### 0.3 Import Dependency Analysis
```bash
# Find circular imports and unused imports
pyflakes src/brain_go_brrr > pyflakes_report.txt

# Visualize dependency graph (if pydeps available)
pip install pydeps
pydeps src/brain_go_brrr --cluster --max-bacon 2 -o deps.svg
```

#### 0.4 Git History Analysis
```bash
# Find when legacy code was introduced
git log -S "compat_coerce" --oneline

# Who wrote the legacy code (to consult them)
git blame src/brain_go_brrr/infra/ml_models/eegpt_compat.py | grep compat_coerce

# Find all commits touching legacy code
git log --follow src/brain_go_brrr/infra/ml_models/eegpt_compat.py
```

### Phase 1: Shadow Mode & Telemetry

#### 1.1 Add Usage Telemetry (Log but Don't Break)
```python
# In eegpt_compat.py
import logging
import os
from functools import wraps

legacy_usage_logger = logging.getLogger("legacy_usage")

def track_legacy_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('compat_coerce', False):
            legacy_usage_logger.warning(
                f"LEGACY USAGE: {func.__name__} called with compat_coerce=True "
                f"from {inspect.stack()[1].filename}:{inspect.stack()[1].lineno}"
            )
            # In production, could send to metrics system
            if os.getenv("STRICT_MODE"):
                raise DeprecationWarning(f"Legacy mode disabled in strict mode")
        return func(*args, **kwargs)
    return wrapper
```

#### 1.2 Run in Shadow Mode for N Days
- Deploy with telemetry
- Monitor logs for any legacy usage
- Identify all callers before removing

### Phase 2: Contract Testing

#### 2.1 Create Explicit Contract Tests
```python
# tests/contracts/test_eegpt_shape_contracts.py
import pytest
import numpy as np
from hypothesis import given, strategies as st

class TestEEGPTShapeContracts:
    """Immutable contracts that MUST NOT change."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        channels=st.just(20),
        samples=st.sampled_from([256, 512, 1024, 2048])
    )
    def test_summary_mode_contract(self, batch_size, channels, samples):
        """Property: summary=True ALWAYS returns (B, 512)"""
        data = np.random.randn(batch_size, channels, samples)
        model = EEGPTModel(auto_load=False)
        
        features = model.extract_features(data, summary=True)
        
        assert features.shape == (batch_size, 512)
        assert features.dtype == np.float32
        assert not np.isnan(features).any()
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32)
    )
    def test_token_mode_contract(self, batch_size):
        """Property: summary=False ALWAYS returns (B, 4, 512)"""
        data = np.random.randn(batch_size, 20, 1024)
        model = EEGPTModel(auto_load=False)
        
        features = model.extract_features(data, summary=False)
        
        assert features.shape == (batch_size, 4, 512)
```

#### 2.2 Mutation Testing to Verify Test Coverage
```bash
# Install mutation testing tool
pip install mutmut

# Run mutation tests on critical module
mutmut run --paths-to-mutate src/brain_go_brrr/infra/ml_models/eegpt_compat.py

# See what mutations survived (gaps in testing)
mutmut results
```

### Phase 3: Safe Removal Process

#### 3.1 Feature Flag Pattern
```python
# src/brain_go_brrr/config/feature_flags.py
import os

class FeatureFlags:
    ALLOW_LEGACY_SHAPES = os.getenv("ALLOW_LEGACY_SHAPES", "false").lower() == "true"
    STRICT_MODE = os.getenv("STRICT_MODE", "false").lower() == "true"
    LOG_LEGACY_USAGE = os.getenv("LOG_LEGACY_USAGE", "true").lower() == "true"

# Usage in code
if FeatureFlags.ALLOW_LEGACY_SHAPES:
    # Legacy path
else:
    # New strict path
```

#### 3.2 Gradual Rollout Strategy
1. **Week 1**: Log all legacy usage, don't break
2. **Week 2**: Warn on legacy usage in dev/staging
3. **Week 3**: Error on legacy usage in CI
4. **Week 4**: Error on legacy usage in staging
5. **Week 5**: Remove legacy code entirely

### Phase 4: Automated Guards

#### 4.1 Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: no-compat-coerce
        name: Check no compat_coerce in production
        entry: bash -c 'rg "compat_coerce\s*=\s*True" src/ && exit 1 || exit 0'
        language: system
        files: \.py$
```

#### 4.2 CI Gates
```yaml
# .github/workflows/legacy-guard.yml
name: Legacy Code Guard
on: [push, pull_request]

jobs:
  check-no-legacy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check no compat_coerce in src
        run: |
          if rg "compat_coerce\s*=\s*True" src/; then
            echo "❌ Found compat_coerce=True in production code!"
            exit 1
          fi
          
      - name: Check all calls have explicit summary
        run: |
          if rg "extract_features\([^)]*\)" src/ | grep -v "summary="; then
            echo "❌ Found extract_features without explicit summary!"
            exit 1
          fi
```

### Phase 5: Verification & Metrics

#### 5.1 Before/After Metrics
```bash
# Measure before removal
echo "=== BEFORE METRICS ===" > metrics_before.txt
echo "Lines of code:" >> metrics_before.txt
wc -l src/brain_go_brrr/infra/ml_models/eegpt_compat.py >> metrics_before.txt
echo "Cyclomatic complexity:" >> metrics_before.txt
radon cc src/brain_go_brrr/infra/ml_models/eegpt_compat.py -s >> metrics_before.txt
echo "Test coverage:" >> metrics_before.txt
coverage report --include="*eegpt_compat*" >> metrics_before.txt
```

#### 5.2 Performance Benchmarks
```python
# tests/benchmarks/test_legacy_removal_perf.py
import timeit

def benchmark_before_removal():
    # Benchmark with compat_coerce=True
    pass

def benchmark_after_removal():
    # Benchmark with strict mode
    pass

# Should see performance improvement without shape coercion
```

### Phase 6: Documentation & Communication

#### 6.1 Migration Guide
```markdown
# Migration Guide: EEGPT Shape Changes

## Breaking Changes
- `compat_coerce` parameter removed
- No automatic shape coercion
- Strict shape validation

## Migration Steps
1. Replace ambiguous calls:
   ```python
   # OLD
   features = model.extract_features(data)
   
   # NEW
   features = model.extract_features(data, summary=True)  # For (B, 512)
   # OR
   features = model.extract_features(data, summary=False) # For (B, 4, 512)
   ```

2. Handle shape explicitly:
   ```python
   # OLD (relied on magic reshape)
   features = model.extract_features(data)  # Got (4, 512) sometimes
   
   # NEW (explicit)
   features = model.extract_features(data, summary=False)  # (1, 4, 512)
   features = features.squeeze(0)  # (4, 512) if needed
   ```
```

### Phase 7: Rollback Plan

#### 7.1 Git Tags for Each Phase
```bash
git tag -a pre-legacy-removal-v1 -m "Before removing compat_coerce"
git tag -a shadow-mode-v1 -m "With telemetry added"
git tag -a strict-mode-v1 -m "With strict validation"
```

#### 7.2 Emergency Rollback
```bash
# If something breaks in production
git checkout pre-legacy-removal-v1
git checkout -b hotfix/restore-legacy

# Or use feature flag
export ALLOW_LEGACY_SHAPES=true
```

### Phase 8: Success Criteria

✅ **Must have before merging:**
1. Zero vulture warnings for removed code
2. 100% branch coverage on new strict paths
3. All contract tests passing
4. Zero legacy usage in telemetry for 7 days
5. Performance benchmarks show improvement
6. Migration guide reviewed by team
7. Rollback tested in staging

### Tools Summary

| Tool | Purpose | Command |
|------|---------|---------|
| vulture | Find dead code | `vulture src/ --min-confidence 80` |
| coverage | Branch coverage | `coverage run --branch -m pytest` |
| pyflakes | Unused imports | `pyflakes src/` |
| mutmut | Test quality | `mutmut run --paths-to-mutate src/` |
| radon | Complexity | `radon cc src/ -s` |
| git blame | Code archaeology | `git blame -L 100,200 file.py` |
| hypothesis | Property testing | `pytest tests/contracts/` |

### Implementation Order

1. **TODAY**: Run vulture + coverage analysis
2. **TOMORROW**: Add telemetry + shadow mode
3. **DAY 3**: Create contract tests
4. **DAY 4**: Add CI gates
5. **WEEK 2**: Remove legacy in staging
6. **WEEK 3**: Remove legacy in production

This is how Netflix, Google, and other tech giants remove legacy code safely.