# TUAB/TUEV Dataset Path Alignment Plan

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



**STATUS**: Plan created, NOT YET IMPLEMENTED
**LAST UPDATED**: 2025-08-30

## Current State Analysis (VERIFIED 2025-08-30)

### ✅ What's Already Working
1. **DataConfig Integration**:
   - `DataConfig.tuab_root` and `DataConfig.tuev_root` properties already exist
   - Environment overrides via `BGB_TUAB_DIR` and `BGB_TUEV_DIR` supported

2. **Dataset Classes Exist**:
   - `TUABDataset` in `src/brain_go_brrr/infra/data/tuab_dataset.py`
   - `TUEVDataset` in `src/brain_go_brrr/infra/data/tuev_dataset.py`
   - Both take `root_dir` parameter (no hardcoded paths!)

3. **Test Infrastructure**:
   - `tests/fixtures/tuab_fixtures.py` exists but uses tiny synthetic data
   - No TUEV fixtures found yet

### ❌ What Needs Alignment (Like Sleep-EDF)

1. **No Real-Data Fixtures**: TUAB/TUEV don't have real-data fixtures like `sleep_edf_path`
2. **No Synthetic Fallbacks**: No `_create_synthetic_tuab/tuev()` functions
3. **No @pytest.mark.data Tests**: Integration tests not properly marked
4. **No Deterministic File Selection**: No `get_tuab_sample_file()` methods
5. **Hardcoded Paths Still Exist**: Found in `test_accuracy.py` and possibly others

## Implementation Plan (NO PARALLEL UNIVERSES!)

### Phase 1: Extend DataConfig (Like Sleep-EDF)

```python
# In src/brain_go_brrr/application/config/base.py

class DataConfig(BaseModel):
    # ... existing ...

    @property
    def tuab_version(self) -> str:
        """Get TUAB version from env or default."""
        return os.environ.get("BGB_TUAB_VERSION", "v3.0.1")

    def get_tuab_sample_file(self, split: str = "train", label: str = "abnormal") -> Path | None:
        """Get a TUAB EDF file deterministically.

        Args:
            split: Dataset split (train/eval/test)
            label: Label type (normal/abnormal)

        Returns:
            Path to EDF file or None if not found
        """
        explicit = os.environ.get("BGB_TUAB_FILE")
        if explicit:
            p = Path(explicit)
            return p if p.exists() else None

        base = self.tuab_root / self.tuab_version / "edf" / split / label / "01_tcp_ar"
        if not base.exists():
            return None

        files = sorted(base.glob("*.edf"))
        files = [f for f in files if not f.name.startswith("._")]
        return files[0] if files else None

    def get_tuev_sample_file(self) -> Path | None:
        """Get a TUEV EDF file deterministically."""
        # Similar implementation
```

### Phase 2: Create Test Fixtures (tests/conftest.py)

```python
def _create_synthetic_tuab(tmp_path: Path) -> Path:
    """TEST-ONLY: Create minimal TUAB-like data.

    Creates 19-channel EDF mimicking TUAB structure for testing
    when real data is not available.
    """
    sfreq = 256
    duration = 120  # 2 minutes
    n_samples = sfreq * duration

    # TUAB standard 19 channels (no Fz!)
    ch_names = ["FP1", "FP2", "F7", "F3", "F4", "F8",
                "T3", "C3", "CZ", "C4", "T4",
                "T5", "P3", "PZ", "P4", "T6",
                "O1", "O2", "A1"]

    # Generate simple EEG-like signals
    data = np.random.randn(19, n_samples) * 50e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    edf_path = tmp_path / "synthetic_tuab.edf"
    mne.export.export_raw(raw, str(edf_path), fmt="edf", physical_range=(None, None))
    return edf_path

@pytest.fixture
def tuab_sample_path(project_root, tmp_path) -> Path:
    """Get path to TUAB sample file.

    Uses DataConfig to resolve paths deterministically.
    Falls back to synthetic data if BGB_ALLOW_SYNTH_TUAB=1.
    """
    from brain_go_brrr.application.config import DataConfig

    config = DataConfig(data_path=project_root / "data")
    path = config.get_tuab_sample_file()

    if path:
        return path

    # Allow synthetic fallback for CI (TEST-ONLY)
    if os.environ.get("BGB_ALLOW_SYNTH_TUAB") == "1":
        return _create_synthetic_tuab(tmp_path)

    pytest.skip("TUAB data not available. Set BGB_TUAB_DIR or use synthetic.")
```

### Phase 3: Mark Real-Data Tests

```python
# Any test using tuab_sample_path or tuev_sample_path:

@pytest.mark.integration
@pytest.mark.data  # Uses real TUAB data via tuab_sample_path fixture
class TestTUABAbnormalityDetection:
    def test_tuab_preprocessing(self, tuab_sample_path):
        """Test TUAB preprocessing pipeline."""
        raw = mne.io.read_raw_edf(tuab_sample_path, preload=False)
        # ... test logic ...
```

### Phase 4: Update Existing Code

1. **Fix test_accuracy.py**:
   - Remove: `base_path = config.tuab_root / "v3.0.1/edf/train"`
   - Use: `config.get_tuab_sample_file()` or fixture

2. **Update TUABDataset usage**:
   - Ensure all instantiations use `config.tuab_root` not literals
   - Example: `TUABDataset(root_dir=config.tuab_root / config.tuab_version)`

3. **Add to pre-commit hook patterns**:
   ```python
   patterns = [
       # ... existing ...
       (r'v3\.0\.1/edf', 'Hardcoded TUAB version - use DataConfig.tuab_version'),
       (r'01_tcp_ar', 'Hardcoded TUAB protocol - use DataConfig methods'),
       (r'tuh_eeg_abnormal', 'Legacy TUAB name - use DataConfig.tuab_root'),
   ]
   ```

### Phase 5: Create Integration Tests

```python
# tests/integration/test_tuab_integration.py

@pytest.mark.integration
@pytest.mark.data
class TestTUABIntegration:
    """Integration tests for TUAB dataset processing."""

    def test_tuab_dataloader(self, tuab_sample_path):
        """Test TUAB can be loaded and processed."""
        from brain_go_brrr.application.config import DataConfig
        from brain_go_brrr.infra.data.tuab_dataset import TUABDataset

        config = DataConfig()
        dataset = TUABDataset(
            root_dir=config.tuab_root / config.tuab_version,
            split="train",
            window_duration=4.0,
            max_files=5  # Small subset for testing
        )

        assert len(dataset) > 0
        window, label = dataset[0]
        assert window.shape == (19, 1024)  # 19 channels, 4s @ 256Hz
        assert label in [0, 1]  # normal or abnormal
```

## Acceptance Criteria

- [ ] DataConfig has `tuab_version`, `get_tuab_sample_file()`, `get_tuev_sample_file()`
- [ ] Fixtures `tuab_sample_path` and `tuev_sample_path` in conftest.py
- [ ] Synthetic fallback functions `_create_synthetic_tuab()` and `_create_synthetic_tuev()`
- [ ] All TUAB/TUEV tests marked with `@pytest.mark.data`
- [ ] Pre-commit hook catches TUAB/TUEV hardcoded paths
- [ ] At least one integration test per dataset that uses real data
- [ ] No hardcoded paths/versions anywhere except DataConfig

## Verification Commands

```bash
# Check for hardcoded TUAB paths
rg 'v3\.0\.1|01_tcp_ar|tuh_eeg_abnormal' --type py | grep -v config

# Check for hardcoded TUEV paths
rg 'v2\.0\.0|tuev_v2' --type py | grep -v config

# Verify all real-data tests are marked
rg -l 'tuab_sample_path|tuev_sample_path' tests | xargs -I{} rg -q '@pytest.mark.data' {}

# Run tests without data (should skip cleanly)
pytest -m "not data" tests/

# Run tests with synthetic fallback
BGB_ALLOW_SYNTH_TUAB=1 BGB_ALLOW_SYNTH_TUEV=1 pytest -m data tests/
```

## Why This Approach

1. **NO PARALLEL UNIVERSES**: We're extending EXISTING classes, not creating new ones
2. **CONSISTENT WITH SLEEP-EDF**: Same pattern (DataConfig → fixtures → tests)
3. **DETERMINISTIC**: sorted() globs, first file selection
4. **CI-FRIENDLY**: Synthetic fallbacks for environments without data
5. **MAINTAINABLE**: Single source of truth in DataConfig

## Next Steps

1. Implement DataConfig extensions
2. Create fixtures with synthetic fallbacks
3. Mark existing tests with @pytest.mark.data
4. Add one integration test per dataset
5. Update pre-commit hook patterns
6. Run verification commands
