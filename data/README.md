# Data Directory Structure

**⚠️ IMPORTANT**: All data files are gitignored. You must obtain datasets separately with proper licensing.

## Directory Layout

```
data/
├── cache/                    # Preprocessed data caches
│   ├── tuab_4s_final/       # TUAB cached windows (ACTIVE)
│   └── tuev_table13/        # TUEV cached windows (ACTIVE)
│
├── datasets/                 # Raw EEG datasets (NOT included)
│   ├── tuab/                # Temple University Abnormal EEG
│   ├── tuev/                # Temple University EEG Events
│   └── external/
│       └── sleep-edf/       # PhysioNet Sleep-EDF recordings
│
└── models/                   # Model weights and checkpoints
    └── pretrained/          # EEGPT pretrained weights
        └── eegpt_mcae_58chs_4s_large4E.ckpt  # Download from Figshare
```

## Dataset Requirements

### TUAB (Abnormal EEG Corpus)
- **Source**: Temple University Hospital
- **Size**: ~120GB compressed
- **Access**: Academic agreement required
- **URL**: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

### TUEV (EEG Events)
- **Source**: Temple University Hospital
- **Size**: ~60GB compressed
- **Access**: Academic agreement required
- **URL**: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

### Sleep-EDF
- **Source**: PhysioNet
- **Size**: 197 PSG recordings
- **Access**: Free with registration
- **URL**: https://physionet.org/content/sleep-edfx/1.0.0/

## Model Weights

### EEGPT Pretrained Model
1. Download from [Figshare](https://figshare.com/s/e37df4f8a907a866df4b)
2. Navigate to: `Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
3. Place in: `data/models/pretrained/`
4. Size: ~40MB

## Cache Information

### Active Caches
- `tuab_4s_final/`: 2,998 preprocessed TUAB windows (4s @ 256Hz)
- `tuev_table13/`: TUEV train/eval caches with padding

These caches are used by the training scripts in `experiments/eegpt_linear_probe/`.

## Notes
- All data files are gitignored for size and licensing reasons
- OSS contributors must obtain their own dataset copies
- See `docs/TRAINING.md` for detailed setup instructions
