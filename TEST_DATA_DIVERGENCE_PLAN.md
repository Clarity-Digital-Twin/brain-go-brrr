# TEST DATA DIVERGENCE PLAN

## Current State (DIVERGENT - NEEDS ALIGNMENT)

### Sleep-EDF Tests
- **Strategy**: REAL DATA ONLY
- **Behavior**: Tests SKIP without real files
- **Problem**: CI fails without mounted data
- **Path Resolution**: ✅ Uses DataConfig (just fixed)

### TUAB/TUEV Tests  
- **Strategy**: SYNTHETIC DATA ONLY
- **Behavior**: Tests ALWAYS pass
- **Problem**: May hide real data edge cases
- **Path Resolution**: ❌ Some hardcoded paths remain

## The Divergence Problem

This inconsistency creates:
1. **Different failure modes** - Sleep-EDF fails in CI, TUAB never fails
2. **Hidden bugs** - TUAB synthetic data may not catch real edge cases
3. **Confusion** - Developers don't know which pattern to follow
4. **CI instability** - Sleep-EDF tests randomly skip based on data availability

## Recommended Solution: Dual-Mode Testing

### Principle: Explicit Test Tiers
```
Unit Tests (default) → Synthetic only, no I/O, fast
Integration Tests → Real data when available, synthetic fallback  
E2E Tests → Real data ONLY, no fallbacks
```

### Implementation Plan

#### Phase 1: Document Policy (IMMEDIATE)
- [ ] Create `docs/TEST_DATA_POLICY.md` with clear rules
- [ ] Mark ALL real-data tests with `@pytest.mark.data`
- [ ] Ensure `--run-data` flag gates all real data access

#### Phase 2: Sleep-EDF Alignment (CURRENT SPRINT)
- [x] Centralize paths in DataConfig ✅ DONE
- [ ] Add synthetic fallback fixture (TEST-ONLY)
- [ ] Keep real data as primary path
- [ ] Environment variable: `BGB_ALLOW_SYNTH_SLEEP_EDF=1`

#### Phase 3: TUAB/TUEV Alignment (NEXT SPRINT)
- [ ] Add DataConfig methods: `tuab_root()`, `tuev_root()`
- [ ] Create real-data integration tests with `@pytest.mark.data`
- [ ] Keep existing synthetic unit tests
- [ ] Remove hardcoded paths (1 remaining in test_accuracy.py)

#### Phase 4: Guardrails (BEFORE MERGE)
- [ ] Pre-commit hook blocking hardcoded dataset paths
- [ ] CI job for nightly real-data tests
- [ ] Document environment variables in README

## Specific Changes Needed

### 1. Sleep-EDF Synthetic Fallback
```python
# tests/conftest.py - ADD synthetic generator
def _make_synthetic_sleep_edf(tmp_path: Path) -> Path:
    """TEST-ONLY: Generate minimal 2-channel Sleep-EDF-like data."""
    # 2 channels (Fpz-Cz, Pz-Oz), 2 minutes, 256Hz
    # Returns path to synthetic EDF
```

### 2. TUAB/TUEV Config Integration
```python
# src/brain_go_brrr/application/config/base.py
@property
def tuab_root(self) -> Path:
    """Get TUAB dataset root with env override."""
    override = os.environ.get("BGB_TUAB_DIR")
    if override:
        return Path(override)
    return self.data_path / "datasets" / "tuab"

@property  
def tuev_root(self) -> Path:
    """Get TUEV dataset root with env override."""
    override = os.environ.get("BGB_TUEV_DIR")
    if override:
        return Path(override)
    return self.data_path / "datasets" / "tuev"
```

### 3. Test Marking Audit
Current state:
- `@pytest.mark.data`: 6 occurrences (TOO FEW!)
- Need to audit ALL tests using real files

Files needing `@pytest.mark.data`:
- tests/unit/domain/sleep/test_analysis.py ✅ DONE
- tests/unit/infra/external/test_yasa_compliance.py ❌ MISSING
- tests/integration/test_yasa_channel_aliasing.py ❌ MISSING
- tests/integration/api/test_api_sleep_edf.py ✅ HAS IT
- tests/integration/test_sleep_enhanced.py ❌ MISSING
- tests/integration/test_eegpt_integration.py ✅ HAS IT

### 4. Environment Variables (DOCUMENT)
```bash
# Dataset roots
BGB_DATA_ROOT=/path/to/data         # Root for all datasets
BGB_SLEEP_EDF_DIR=/path/to/sleep    # Override Sleep-EDF location
BGB_TUAB_DIR=/path/to/tuab          # Override TUAB location  
BGB_TUEV_DIR=/path/to/tuev          # Override TUEV location

# Test control
BGB_ALLOW_SYNTH_SLEEP_EDF=1         # Allow synthetic Sleep-EDF in tests
```

## Acceptance Criteria

### Must Have (Before Merge)
- [ ] Zero hardcoded dataset paths (except mocks/config)
- [ ] All real-data tests marked with `@pytest.mark.data`
- [ ] DataConfig owns ALL dataset resolution
- [ ] Tests pass in BOTH modes:
  - With real data: `pytest --run-data`
  - Without data: `pytest` (skips or synthetic)

### Should Have (This Sprint)
- [ ] Sleep-EDF synthetic fallback for CI
- [ ] TUAB/TUEV config methods
- [ ] Pre-commit hook for path literals

### Nice to Have (Future)
- [ ] Golden sample dataset (~10MB) for E2E
- [ ] Automated data fetching script
- [ ] Performance benchmarks with real vs synthetic

## Verification Commands

```bash
# Check for hardcoded paths (should return 0)
grep -r "SC4001E0-PSG.edf" tests/ scripts/ src/ | grep -v mock | wc -l
grep -r "sleep-edf-database-expanded" tests/ scripts/ src/ | grep -v config | wc -l

# Check for unsorted globs (should return 0)  
grep -r '\.glob(' tests/ scripts/ | grep 'PSG.edf' | grep -v sorted | wc -l

# Count data marks (should be >20 when complete)
grep -r '@pytest.mark.data' tests/ | wc -l

# Test both modes
pytest --run-data  # Real data mode
pytest             # Synthetic/skip mode
```

## Timeline

- **Week 1**: Complete Sleep-EDF alignment ✅ DONE
- **Week 2**: Add synthetic fallback, mark all tests
- **Week 3**: TUAB/TUEV config integration
- **Week 4**: Guardrails and documentation

## Notes

The goal is NOT perfection but CONSISTENCY:
- Same pattern for all datasets
- Clear separation of test tiers
- No surprises in production
- Fast CI with optional real-data validation

This divergence was technical debt from rapid development. Fixing it now prevents future confusion and ensures reliable testing at all levels.