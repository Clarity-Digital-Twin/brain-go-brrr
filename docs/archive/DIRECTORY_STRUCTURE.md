# Brain-Go-Brrr Directory Structure

## Source of Truth Directory Layout

```
brain-go-brrr/
├── data/                           # ALL data-related files (gitignored)
│   ├── datasets/                   # Raw datasets
│   │   └── external/
│   │       ├── tuab/              # TUAB EEG dataset
│   │       └── sleep-edf/         # Sleep-EDF dataset
│   ├── cache/                      # Processed data caches
│   │   ├── tuab_index.json        # TUAB file index
│   │   ├── tuab_8s/               # 8-second window cache
│   │   └── tuab_4s/               # 4-second window cache (to be created)
│   └── models/                     # ALL models (pretrained + trained)
│       ├── pretrained/             # Downloaded pretrained models
│       │   └── eegpt_mcae_58chs_4s_large4E.ckpt
│       └── trained/                # Our trained models
│           ├── linear_probes/      # Linear probe experiments
│           │   ├── tuab_8s_best_0.8133.pt
│           │   └── tuab_4s_best_TBD.pt
│           └── checkpoints/        # Training checkpoints
│
├── src/brain_go_brrr/              # Source code
│   ├── data/                       # Data loading modules
│   ├── models/                     # Model definitions
│   ├── training/                   # Training utilities
│   └── inference/                  # Inference utilities
│
├── experiments/                    # Experiment scripts and configs
│   └── eegpt_linear_probe/
│       ├── configs/                # Training configurations
│       ├── scripts/                # Training/evaluation scripts
│       └── results/                # Results and analysis
│
├── tests/                          # Test suite
├── docs/                           # Documentation
└── logs/                           # Runtime logs (gitignored)
```

## Path Configuration

All paths should use environment variables:
- `BGB_DATA_ROOT` = `/path/to/brain-go-brrr/data`
- Models: `${BGB_DATA_ROOT}/models/`
- Datasets: `${BGB_DATA_ROOT}/datasets/`
- Cache: `${BGB_DATA_ROOT}/cache/`

## What Goes Where

### `/data/` (gitignored)
- Raw datasets
- Processed caches
- All model files (pretrained and trained)
- Any large files

### `/src/brain_go_brrr/`
- Python source code
- Model definitions (code, not weights)
- Training/inference logic

### `/experiments/`
- Experiment-specific scripts
- Configuration files
- Results analysis
- Documentation

### `/models/` directory (TO BE REMOVED)
This directory should be removed. All models belong in `/data/models/`