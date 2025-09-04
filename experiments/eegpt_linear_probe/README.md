# EEGPT Linear Probe Experiments

Training scripts for EEGPT linear probing on TUAB and TUEV datasets.

## Training Scripts

- `train_tuab_mne.py` - TUAB abnormality detection (binary) with MNE preprocessing
- `train_tuev_mne.py` - TUEV event detection (6-class) with MNE preprocessing

## Channel Specifications

### TUAB (19 channels - no Fz)
Expected channels: FP1, FP2, F7, F3, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, OZ, O2

### TUEV (20 channels - includes Fz)
Expected channels: FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, OZ, O2

## Usage

```bash
# Set environment variables
export BGB_DATA_ROOT=/path/to/data
export BGB_CACHE_DIR=/path/to/cache

# Train TUAB
uv run python experiments/eegpt_linear_probe/train_tuab_mne.py

# Train TUEV
uv run python experiments/eegpt_linear_probe/train_tuev_mne.py
```

## Architecture

All experiments follow the unified architecture:
- Import datasets from `src/brain_go_brrr/infra/data/`
- Use preprocessing from `src/brain_go_brrr/infra/preprocessing/`
- Import models from `src/brain_go_brrr/infra/ml_models/`
- No parallel implementations - everything uses src/ components

## Configuration

See `configs/` directory for YAML configuration files with hyperparameters.

## Important Notes

- Normalization happens in the EEGPT wrapper (SSOT)
- Datasets provide Volts (SI units), not millivolts
- Use ProbeFactory for creating probes (EEGPTProbe is deprecated)
- No sys.path hacks - proper imports only
