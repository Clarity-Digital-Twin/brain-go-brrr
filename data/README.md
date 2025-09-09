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

### EEGPT Pretrained Model (Primary)
1. Download from [Figshare](https://figshare.com/s/e37df4f8a907a866df4b)
2. Navigate to: `Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
3. Place in: `data/models/pretrained/`
4. Size: 974MB
5. **Purpose**: Feature extraction for TUAB/TUEV/TUSZ

### SeizureTransformer Weights (Research/Comparison)
1. Download from [Google Drive](https://drive.google.com/drive/folders/17pKhwFc4x1_2zwXTndKawoNKlaXIW-VE)
2. File: `seizure_transformer_wu2025.pth`
3. Place in: `data/models/pretrained/`
4. Size: 169MB
5. **Note**: For comparison only - we use EEGPT + BiLSTM for TUSZ

## Cache Information

### Active Caches
- `tuab_4s_final/`: 2,998 preprocessed TUAB windows (4s @ 256Hz)
- `tuev_table13/`: TUEV train/eval caches with padding
- `tuev_23ch_paper_parity/`: TUEV 23-channel cache (building in tmux)

### Cache Status
- ✅ TUAB: Ready for training
- 🔄 TUEV: 23-channel cache building (`tmux attach -t tuev_cache`)
- ⏳ TUSZ: To be built when implementing temporal detection

These caches are used by the training scripts in `experiments/eegpt_linear_probe/`.

## Notes
- All data files are gitignored for size and licensing reasons
- OSS contributors must obtain their own dataset copies
- See `docs/TRAINING.md` for detailed setup instructions
