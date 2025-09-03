# EEGPT Linear Probe Documentation

Essential documentation for the EEGPT linear probe experiments.

## Key Documents

- **[CHANNEL_SPECIFICATIONS.md](CHANNEL_SPECIFICATIONS.md)** - Critical channel specifications for TUAB (19ch) and TUEV (20ch) datasets
- **[MNE_INTEGRATION_README.md](MNE_INTEGRATION_README.md)** - MNE preprocessing pipeline integration guide

## Training Scripts

- `train_tuab_mne.py` - TUAB abnormality detection training with MNE preprocessing
- `train_tuev_mne.py` - TUEV event detection (6-class) with MNE preprocessing

## Important Notes

- All datasets should import from `src/brain_go_brrr/infra/data/`
- All preprocessing should use `src/brain_go_brrr/infra/preprocessing/`
- No `sys.path.insert` hacks - everything imports cleanly from src
- Normalization happens in the EEGPT wrapper, datasets provide raw mV data

## Archive

Historical fix documentation has been archived in `archive/fix_history/` for reference.
