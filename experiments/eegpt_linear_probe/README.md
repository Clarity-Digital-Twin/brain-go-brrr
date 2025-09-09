# EEGPT Linear Probe Experiments

Training scripts for EEGPT linear probing on TUAB and TUEV datasets.

## Training Scripts

- `train_tuab_mne.py` - TUAB abnormality detection (binary) with MNE preprocessing
- `train_tuev_mne.py` - TUEV event detection (6-class) with MNE preprocessing

## Channel Specifications

### TUAB (19 channels - no Fz)
Expected channels (mixed‑case): Fp1, Fp2, F7, F3, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, Oz, O2

### TUEV (paper parity: 23 raw channels + learned 23→20 mapper)
Raw channels kept (mixed‑case, no synthesis):
Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, A1, A2, Fz, Cz, Pz, T1, T2

These 23 raw channels are mapped by a learnable Conv2d 23→20 mapper (BN/GELU + depthwise 1×55 + BN/Dropout 0.8) before EEGPT, matching the paper.

## Usage

```bash
# Set environment variables
export BGB_DATA_ROOT=/path/to/data
export BGB_CACHE_DIR=/path/to/cache

# Train TUAB
uv run python experiments/eegpt_linear_probe/train_tuab_mne.py

# Train TUEV (paper parity)
uv run python experiments/eegpt_linear_probe/train_tuev_mne.py --config experiments/eegpt_linear_probe/configs/tuev_paper_parity.yaml
# or use the launch script
./experiments/eegpt_linear_probe/scripts/launch_tuev_paper_parity.sh
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
- TUEV 20‑channel preprocessing is legacy; use paper parity (23‑ch + mapper)
- No sys.path hacks - proper imports only
