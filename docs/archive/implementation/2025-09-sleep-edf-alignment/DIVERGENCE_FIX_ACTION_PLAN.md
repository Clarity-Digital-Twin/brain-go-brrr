# DIVERGENCE FIX ACTION PLAN

## Current State Summary
✅ **DONE**: Sleep-EDF paths centralized in DataConfig
⚠️ **ISSUE**: Test strategy inconsistent between datasets
❌ **TODO**: Many tests unmarked, docs outdated, no guardrails

## PHASE 1: Mark Tests (IMMEDIATE - This Week)

### Tests Needing @pytest.mark.data
Add marker to these 13 files that touch real data:
```bash
# Integration tests (should ALL have @pytest.mark.data)
tests/integration/test_yasa_channel_aliasing.py
tests/integration/test_train_sleep_probe.py  
tests/integration/test_sleep_enhanced.py
tests/integration/test_end_to_end.py

# Smoke tests
tests/smoke/test_end_to_end_wiring.py

# Unit tests that touch real data
tests/unit/cli/test_streaming.py
tests/unit/data/test_edf_loader_unit.py
tests/unit/infra/data/test_edf_loader.py
tests/unit/infra/external/test_yasa_compliance.py
tests/unit/domain/sleep/test_montage_detection.py
tests/unit/domain/preprocessing/test_pipeline.py
tests/unit/domain/preprocessing/test_flexible.py
tests/unit/infra/adapters/test_autoreject.py
```

### How to Fix
For each file:
1. Check if it loads real EDF files
2. If yes, add `@pytest.mark.data` to class or test functions
3. Ensure it skips cleanly without data

## PHASE 2: Add Synthetic Fallback (This Week)

### Sleep-EDF Test Fixture Enhancement
```python
# tests/conftest.py - ADD this helper
def _create_synthetic_sleep_edf(tmp_path: Path) -> Path:
    """TEST-ONLY: Create minimal 2-channel Sleep-EDF-like data."""
    import mne
    import numpy as np
    
    # 2 channels like Sleep-EDF, 2 minutes, 256Hz
    sfreq = 256
    duration = 120
    n_samples = sfreq * duration
    
    # Generate simple EEG-like signals
    t = np.arange(n_samples) / sfreq
    data = np.array([
        20e-6 * np.sin(2 * np.pi * 10 * t),  # ~10Hz alpha
        15e-6 * np.sin(2 * np.pi * 3 * t)    # ~3Hz delta
    ])
    
    ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz"]  # Sleep-EDF channel names
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    
    # Export as EDF
    edf_path = tmp_path / "synthetic_sleep.edf"
    raw.export(str(edf_path), fmt="edf")
    return edf_path

# Modify sleep_edf_path fixture
@pytest.fixture
def sleep_edf_path(project_root, tmp_path) -> Path:
    """Get Sleep-EDF path - real or synthetic based on env."""
    from brain_go_brrr.application.config import DataConfig
    
    config = DataConfig(data_path=project_root / "data")
    path = config.get_sleep_edf_psg_file()
    
    if path:
        return path
    
    # Allow synthetic fallback for CI
    if os.environ.get("BGB_ALLOW_SYNTH_SLEEP_EDF") == "1":
        return _create_synthetic_sleep_edf(tmp_path)
    
    pytest.skip("Sleep-EDF not available. Set BGB_ALLOW_SYNTH_SLEEP_EDF=1 for synthetic")
```

## PHASE 3: TUAB/TUEV Parity (Next Sprint)

### Add to DataConfig
```python
# src/brain_go_brrr/application/config/base.py
@property
def tuab_root(self) -> Path:
    """Get TUAB root directory."""
    override = os.environ.get("BGB_TUAB_DIR")
    if override:
        return Path(override)
    return self.data_path / "datasets" / "tuab"

@property
def tuev_root(self) -> Path:
    """Get TUEV root directory."""
    override = os.environ.get("BGB_TUEV_DIR")
    if override:
        return Path(override)
    return self.data_path / "datasets" / "tuev"
```

### Fix TUAB Hardcoded Path
- File: `tests/unit/domain/abnormal/test_accuracy.py`
- Replace: `Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf/train")`
- With: `config.tuab_root / "v3.0.1/edf/train"`

## PHASE 4: Documentation Update (Next Sprint)

### Files to Update
1. `docs/TRAINING.md` - Update Sleep-EDF paths
2. `docs/QUICK_START.md` - Update all dataset paths
3. `CLAUDE.md` - Update examples to use env vars
4. `AGENTS.md` - Update dataset references

### Standard Examples to Use
```bash
# Instead of hardcoded paths, show:
export BGB_DATA_ROOT=/path/to/data
export BGB_SLEEP_EDF_DIR=/custom/sleep-edf

# Or use defaults:
# Sleep-EDF: $BGB_DATA_ROOT/datasets/sleep-edf/sleep-edf-database-expanded-1.0.0/
# TUAB: $BGB_DATA_ROOT/datasets/tuab/
# TUEV: $BGB_DATA_ROOT/datasets/tuev/
```

## PHASE 5: Guardrails (Before Next Release)

### Pre-commit Hook
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Block hardcoded dataset literals

if grep -r "SC4001E0-PSG\.edf" tests/ scripts/ src/ | grep -v mock; then
    echo "❌ ERROR: Hardcoded Sleep-EDF filename detected!"
    exit 1
fi

if grep -r "sleep-edf-database-expanded-1\.0\.0" tests/ scripts/ src/ | grep -v "application/config"; then
    echo "❌ ERROR: Hardcoded Sleep-EDF version detected!"
    exit 1
fi

if grep -r "datasets/external/" tests/ scripts/ src/ | grep -E "(sleep-edf|tuab|tuev)"; then
    echo "❌ ERROR: Legacy external path detected!"
    exit 1
fi
```

## Verification After Each Phase

### Phase 1 Complete When:
```bash
grep -r "@pytest.mark.data" tests/ | wc -l  # Should be ~20+
```

### Phase 2 Complete When:
```bash
# CI passes without real data
unset BGB_DATA_ROOT
export BGB_ALLOW_SYNTH_SLEEP_EDF=1
pytest tests/unit/domain/sleep -v  # Should pass or skip cleanly
```

### Phase 3 Complete When:
```bash
# No TUAB/TUEV literals
grep -r "tuh_eeg_abnormal/v3" tests/ | wc -l  # Should be 0
python -c "from brain_go_brrr.application.config import DataConfig; c=DataConfig(); print(c.tuab_root)"  # Should work
```

### Phase 4 Complete When:
```bash
# No old paths in docs
grep -r "datasets/external/sleep-edf" docs/ | wc -l  # Should be 0
```

### Phase 5 Complete When:
```bash
# Pre-commit blocks new literals
echo 'x = "SC4001E0-PSG.edf"' >> test.py
git add test.py
git commit -m "test"  # Should FAIL
```

## Priority Order

1. **TODAY**: Mark the 13 tests with @pytest.mark.data
2. **THIS WEEK**: Add synthetic Sleep-EDF fallback
3. **NEXT WEEK**: TUAB/TUEV config methods
4. **LATER**: Update docs, add pre-commit hook

## Success Criteria

✅ All tests properly marked with @pytest.mark.data
✅ CI passes without real data (synthetic mode)
✅ No hardcoded paths outside config/mocks
✅ Documentation uses env vars consistently
✅ Pre-commit prevents regression