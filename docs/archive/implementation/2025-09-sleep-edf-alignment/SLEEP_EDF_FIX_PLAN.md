# SLEEP-EDF PATH FIX PLAN - NO BULLSHIT, NO PARALLEL UNIVERSES

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



## THE PROBLEM
- Other AI spread "sleep-edf-database-expanded-1.0.0" and "SC4001E0-PSG.edf" EVERYWHERE
- 25 hardcoded filename references (SC4001E0-PSG.edf)
- 18 version string duplications (sleep-edf-database-expanded-1.0.0)
- Only 5 tests marked with @pytest.mark.data (many data tests unmarked)
- 4 unsorted glob() calls for PSG files
- Created fixtures but didn't use existing config system

## THE EXISTING INFRASTRUCTURE WE MUST USE
```
src/brain_go_brrr/application/config/
├── __init__.py       # Exports Config, DataConfig, etc.
├── base.py           # HAS DataConfig class with data_path field
└── abnormality_config.py
```

**DataConfig ALREADY EXISTS** - we extend it, we don't create new shit.

## THE ONLY CORRECT FIX

### 1. Extend EXISTING DataConfig (in base.py)
```python
# ADD to existing DataConfig class in application/config/base.py:

@property
def sleep_edf_version(self) -> str:
    """Get Sleep-EDF version from env or default."""
    return os.environ.get("BGB_SLEEP_EDF_VERSION", "sleep-edf-database-expanded-1.0.0")

@property
def sleep_edf_root(self) -> Path:
    """Get Sleep-EDF root directory with env override."""
    # Check explicit override first
    override = os.environ.get("BGB_SLEEP_EDF_DIR")
    if override:
        return Path(override)

    # Use data_path (which already exists in this class!)
    base = self.data_path / "datasets" / "sleep-edf" / self.sleep_edf_version
    if base.exists():
        return base

    # Legacy fallback (temporary)
    legacy = self.data_path / "datasets" / "external" / "sleep-edf"
    return legacy

@property
def sleep_edf_cassette_dir(self) -> Path:
    """Get Sleep-EDF cassette directory."""
    return self.sleep_edf_root / "sleep-cassette"

def get_sleep_edf_psg_file(self, explicit: str = None) -> Path | None:
    """Get a PSG file deterministically."""
    if explicit or os.environ.get("BGB_SLEEP_EDF_FILE"):
        p = Path(explicit or os.environ.get("BGB_SLEEP_EDF_FILE"))
        return p if p.exists() else None

    # Get first file sorted (deterministic)
    files = sorted(self.sleep_edf_cassette_dir.glob("*-PSG.edf"))
    # Filter out macOS resource forks
    files = [f for f in files if not f.name.startswith("._")]
    return files[0] if files else None
```

### 2. Fix app code to use config
```python
# src/brain_go_brrr/application/pipeline/parallel.py
from ..config import DataConfig

config = DataConfig()
edf_path = config.get_sleep_edf_psg_file()
if not edf_path:
    logger.error("No Sleep-EDF PSG found. Set BGB_DATA_ROOT or BGB_SLEEP_EDF_DIR.")
    return
```

### 3. Fix test fixtures to use config
```python
# tests/conftest.py
from brain_go_brrr.application.config import DataConfig

@pytest.fixture
def sleep_edf_path(project_root) -> Path:
    """Get Sleep-EDF path from config."""
    config = DataConfig(data_path=project_root / "data")
    path = config.get_sleep_edf_psg_file()
    if not path:
        pytest.skip("Sleep-EDF data not available. Use --run-data.")
    return path

@pytest.fixture
def sleep_edf_dir(project_root) -> Path:
    """Get Sleep-EDF directory from config."""
    config = DataConfig(data_path=project_root / "data")
    dir = config.sleep_edf_cassette_dir
    if not dir.exists():
        pytest.skip("Sleep-EDF directory not available. Use --run-data.")
    return dir
```

### 4. Remove all hardcoded strings
- Replace ALL "SC4001E0-PSG.edf" with config or fixture (25 occurrences)
- Replace ALL "sleep-edf-database-expanded-1.0.0" with config (18 occurrences)
- Add sorted() to ALL glob() calls (4 unsorted PSG globs)
- Add @pytest.mark.data to ALL tests using real datasets

### 5. Fix the channel hack
```python
# WRONG (what's there now):
if all(ch_type == "misc" for ch_type in raw.get_channel_types()):
    raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))

# RIGHT:
eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False)
if not eeg_picks:
    pytest.skip("No EEG channels found")
raw.pick(eeg_picks)
```

## WHAT WE'RE NOT DOING
- ❌ NOT creating new `registry.py`
- ❌ NOT creating new `paths.py`
- ❌ NOT creating new config systems
- ❌ NOT adding pydantic Settings
- ❌ NOT making new directories

## FILES TO MODIFY (ONLY)
1. `src/brain_go_brrr/application/config/base.py` - ADD properties to DataConfig
2. `src/brain_go_brrr/application/pipeline/parallel.py` - USE config
3. `tests/conftest.py` - SIMPLIFY fixtures to use config
4. Test files with hardcoded paths - USE fixtures
5. `tests/unit/domain/sleep/test_analysis.py` - FIX channel hack

## VERIFICATION
```bash
# After fixes, these should return 0:
grep -r "SC4001E0-PSG.edf" src/ tests/ scripts/ | grep -v mock | wc -l  # Should be 0 (currently 25)
grep -r "sleep-edf-database-expanded-1.0.0" src/ tests/ scripts/ | grep -v config | wc -l  # Should be 0 (currently 18)
grep -r "\.glob(" tests/ scripts/ | grep "PSG.edf" | grep -v sorted | wc -l  # Should be 0 (currently 4)

# Unsorted glob locations to fix:
# - tests/integration/api/test_api_sleep_edf.py:146
# - tests/unit/domain/sleep/test_analysis.py (2 locations)
# - scripts/testing/test_sleep_analysis.py:71
```

## ENVIRONMENT VARIABLES (DOCUMENT THESE)
- `BGB_DATA_ROOT` - Root data directory (default: "data")
- `BGB_SLEEP_EDF_VERSION` - Version string (default: "sleep-edf-database-expanded-1.0.0")
- `BGB_SLEEP_EDF_DIR` - Override entire Sleep-EDF root
- `BGB_SLEEP_EDF_FILE` - Specific PSG file to use

## THIS IS THE WAY
One config. One source of truth. No parallel universes. No new files.
