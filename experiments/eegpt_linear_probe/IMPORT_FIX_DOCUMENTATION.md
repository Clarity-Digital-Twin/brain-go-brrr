# Import Issue Root Cause Analysis and Permanent Fix

## Problem Statement
Training script was failing with `ModuleNotFoundError: No module named 'brain_go_brrr'` when importing EEGPTWrapper.

## Root Cause Analysis

### 1. Package Structure
```
brain-go-brrr/
├── src/
│   └── brain_go_brrr/        # Python package
│       ├── __init__.py
│       ├── _typing.py         # Type protocols
│       └── infra/
│           └── ml_models/
│               ├── eegpt_wrapper.py
│               └── eegpt_compat.py
├── experiments/               # NOT a package (no __init__.py in experiments/)
│   └── eegpt_linear_probe/
│       └── train_tuab_mne.py
└── pyproject.toml            # Defines package installation
```

### 2. Package Installation
- `pyproject.toml` specifies: `[tool.setuptools.packages.find] where = ["src"]`
- This means `brain_go_brrr` is installed as a top-level package
- Installation happens via `uv sync` in development mode (editable install)

### 3. The Import Problem

#### Wrong Approach (BROKEN):
```python
# This doesn't work because 'src' is not a package
from src.brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
```

#### Correct Approach (FIXED):
```python
# This works because brain_go_brrr is the installed package
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
```

### 4. Why `python3` Failed but `uv run python` Works

- `python3` uses system Python without the virtual environment
- `uv run python` activates the project's virtual environment where `brain-go-brrr` is installed
- Package is installed as: `brain-go-brrr 1.0.0 /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr`

## Permanent Fixes Applied

### Fix 1: Corrected Import Path
**File**: `train_tuab_mne.py` line 28
```python
# OLD (wrong):
from src.brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

# NEW (correct):
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
```

### Fix 2: Launch Script Uses uv
**File**: `scripts/launch_tuab_mne.sh` lines 73, 84
```bash
# OLD (wrong):
python train_tuab_mne.py

# NEW (correct):
uv run python train_tuab_mne.py
```

## Import Strategy Best Practices

### 1. For Code Inside `src/brain_go_brrr/`:
```python
# Use relative imports within the package
from brain_go_brrr._typing import MNERaw
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
```

### 2. For Scripts Outside the Package (experiments/):
```python
# Add to sys.path if needed for local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import installed package normally
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

# Import local experiment modules
from experiments.eegpt_linear_probe.datasets.tuab_mne_dataset import TUABMNEDataset
```

### 3. Always Use `uv run`:
```bash
# Wrong:
python script.py

# Correct:
uv run python script.py

# Or use the Makefile:
make run SCRIPT=script.py
```

## Verification Tests

### Test 1: Package Installation
```bash
uv pip list | grep brain-go-brrr
# Output: brain-go-brrr 1.0.0 /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr
```

### Test 2: Import Chain
```bash
uv run python -c "
from brain_go_brrr._typing import MNERaw
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper
print('✅ Imports work')
"
```

### Test 3: Training Script
```bash
cd experiments/eegpt_linear_probe
uv run python train_tuab_mne.py --help
# Should show help message without import errors
```

## Key Lessons

1. **Package vs Directory**: `src/` is just a directory, `brain_go_brrr` is the package
2. **Virtual Environment**: Always use `uv run` to ensure correct environment
3. **Editable Install**: Development uses editable install via `uv sync`
4. **No Hacks**: Don't manipulate sys.path unnecessarily - use proper package structure

## Environment Variables

The training script properly handles environment variables:
- `BGB_DATA_ROOT`: Base directory for data/models/cache
- Resolution happens in `resolve_env_vars()` function
- Launch script exports: `export BGB_DATA_ROOT="$DATA_ROOT"`

## Summary

The import issues were caused by:
1. Wrong import path (`src.brain_go_brrr` instead of `brain_go_brrr`)
2. Not using `uv run` to activate the virtual environment
3. Package not being properly installed (but it was via `uv sync`)

All issues have been permanently fixed by:
1. Correcting import statements
2. Using `uv run python` in launch scripts
3. Proper environment variable handling

The system is now ready for training with proper Python package management.