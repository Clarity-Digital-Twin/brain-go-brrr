# Final Alignment Action Plan

## Executive Summary
The Sleep-EDF centralization is COMPLETE. TUAB/TUEV need parity. Documentation needs cleanup. A few technical debt items remain.

**VERIFICATION STATUS (2025-08-31)**:
- Sleep-EDF hardcoded paths: ✅ Only in mocks (verified acceptable)
- TUAB hardcoded paths: ❌ Still present in test_accuracy.py
- Pre-commit hook: ✅ Exists but needs expansion
- Documentation drift: 🟡 Partially fixed

## Critical Path Items (MUST DO)

### 1. Fix Synthetic EDF Export (conftest.py:335)
**Problem**: Using deprecated `raw.export()` API
```python
# Current (line 335):
raw.export(str(edf_path), fmt="edf")

# Should be:
from mne.export import export_raw
export_raw(raw, str(edf_path), fmt="edf", physical_range=(None, None))
```

### 2. Fix Channel Names (conftest.py:329)
**Problem**: Using "EEG " prefix unnecessarily
```python
# Current:
ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz"]

# Should be:
ch_names = ["Fpz-Cz", "Pz-Oz"]
```

### 3. Add Missing @pytest.mark.data
**Files to mark**:
- `tests/integration/test_yasa_channel_aliasing.py` - reads real EDF
- `tests/unit/domain/abnormal/test_accuracy.py` - has hardcoded TUAB paths
- Any other test that calls `sleep_edf_path` or `sleep_edf_dir` fixtures

### 4. Expand Pre-commit Patterns
**File**: `.pre-commit-hooks/check_hardcoded_paths.py`
Add patterns:
```python
patterns = [
    # ... existing ...
    (r'v3\.0\.1', 'Hardcoded TUAB version - use DataConfig.tuab_version'),
    (r'01_tcp_ar', 'Hardcoded TUAB protocol - use DataConfig methods'),
    (r'tuh_eeg_abnormal', 'Legacy TUAB name - use DataConfig.tuab_root'),
    (r'v2\.0\.\d+', 'Hardcoded TUEV version - use DataConfig.tuev_version'),
    (r'/abnormal/01_tcp_ar', 'Hardcoded TUAB structure - use DataConfig.get_tuab_sample_file()'),
    (r'/normal/01_tcp_ar', 'Hardcoded TUAB structure - use DataConfig.get_tuab_sample_file()'),
]
```

## TUAB/TUEV Parity Items (IMPORTANT)

### 5. Add DataConfig Methods
**File**: `src/brain_go_brrr/application/config/base.py`
```python
@property
def tuab_version(self) -> str:
    return os.environ.get("BGB_TUAB_VERSION", "v3.0.1")

def get_tuab_sample_file(self, split="train", label="abnormal") -> Path | None:
    """Get deterministic TUAB file."""
    explicit = os.environ.get("BGB_TUAB_FILE")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    
    base = self.tuab_root / self.tuab_version / "edf" / split / label / "01_tcp_ar"
    if not base.exists():
        return None
    
    files = sorted(base.glob("*.edf"))
    files = [f for f in files if not f.name.startswith("._")]
    return files[0] if files else None
```

### 6. Add Synthetic Fallbacks
**File**: `tests/conftest.py`
```python
def _create_synthetic_tuab(tmp_path: Path) -> Path:
    """TEST-ONLY: Create TUAB-like data."""
    # 19 channels (no Fz), 256Hz, 2 minutes
    ch_names = ["FP1", "FP2", "F7", "F3", "F4", "F8", 
                "T3", "C3", "CZ", "C4", "T4",
                "T5", "P3", "PZ", "P4", "T6", 
                "O1", "O2", "A1"]
    # ... create synthetic data ...
```

### 7. Add Fixtures
**File**: `tests/conftest.py`
```python
@pytest.fixture
def tuab_sample_path(project_root, tmp_path) -> Path:
    config = DataConfig(data_path=project_root / "data")
    path = config.get_tuab_sample_file()
    if path:
        return path
    if os.environ.get("BGB_ALLOW_SYNTH_TUAB") == "1":
        return _create_synthetic_tuab(tmp_path)
    pytest.skip("TUAB data not available")
```

## Documentation Cleanup (NICE TO HAVE)

### 8. Update AGENTS.md
- Replace `data/datasets/external/sleep-edf` examples
- Show DataConfig usage instead

### 9. Remove Unused File
- Delete `tests/unit/safe_tests.txt` if confirmed unused

### 10. Create Root Policy Docs
```markdown
# DATA_PATHS_POLICY.md
- SSOT: DataConfig for all datasets
- Env vars: BGB_DATA_ROOT, BGB_*_DIR, BGB_*_VERSION
- Resolution order: explicit > env > default

# TEST_DATA_POLICY.md  
- Unit tests: synthetic only
- Integration: real data with @pytest.mark.data
- CI: skip cleanly or use synthetic fallback
```

## Verification Commands

```bash
# After each fix, verify:

# 1. No hardcoded paths
rg 'SC4001E0|v3\.0\.1|01_tcp_ar' src/ tests/ --type py | grep -v config

# 2. All globs sorted
rg '\.glob\(' src/ tests/ | grep -v sorted

# 3. Check @pytest.mark.data coverage
for f in $(rg -l 'read_raw_edf|tuab_sample|tuev_sample' tests/); do
    rg -q '@pytest.mark.data' "$f" || echo "Missing: $f"
done

# 4. Run tests (should skip/pass)
pytest -m "not data" --tb=short
BGB_ALLOW_SYNTH_SLEEP_EDF=1 pytest -k sleep --tb=short
```

## Definition of Done

✅ Zero hardcoded paths outside DataConfig
✅ All datasets have deterministic file selection
✅ All real-data tests marked with @pytest.mark.data
✅ Pre-commit catches all patterns
✅ Tests pass in all three modes (no data, synthetic, real)
✅ Documentation shows DataConfig usage

## Priority Order

1. **HIGH**: Fix items 1-4 (technical debt + missing marker)
2. **MEDIUM**: Implement items 5-7 (TUAB/TUEV parity)
3. **LOW**: Complete items 8-10 (documentation)

## Why This Matters

- **No Parallel Universes**: Using EXISTING classes, just adding methods
- **DeepMind/Google Standard**: Synthetic units + golden real-data checks
- **CI Stability**: Tests don't randomly fail
- **Maintainability**: Clear patterns for future datasets