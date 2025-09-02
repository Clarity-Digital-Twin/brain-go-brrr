# Experiments Directory

Training scripts and research experiments. All experiments import from `src/brain_go_brrr/`.

## 🏗️ Architecture Rules

**CRITICAL**: Experiments are THIN LAYERS that import from src/
- ✅ Import datasets, models, utils from `src/brain_go_brrr/`
- ✅ Keep only training loops and configs here
- ❌ NEVER reimplement what exists in src/
- ❌ NEVER use `sys.path.insert` hacks

## 📁 Structure

```
experiments/
├── README.md                      # This file
└── eegpt_linear_probe/           # EEGPT linear probe training
    ├── train_tuab_mne.py         # TUAB abnormality detection
    ├── train_tuev_mne.py         # TUEV 6-class event detection
    ├── test_*.py                 # Essential test scripts
    ├── configs/                  # Training configurations
    │   ├── tuab.yaml            # TUAB config (19 channels)
    │   └── tuev.yaml            # TUEV config (20 channels)
    ├── docs/                     # Essential documentation
    │   ├── README.md            # Navigation guide
    │   ├── CHANNEL_SPECIFICATIONS.md  # Critical channel specs
    │   └── MNE_INTEGRATION_README.md  # MNE preprocessing guide
    ├── mne_integration/          # Cache building utilities
    │   └── cache_builder.py     # MNE cache builder
    ├── scripts/                  # Launch and monitoring scripts
    └── archive/                  # Historical reference
        ├── fix_history/         # 11 temporary fix docs
        └── old_tests/          # 3 one-off test scripts
```

## ✅ Clean Architecture

All components import from src:
- **Datasets**: `from brain_go_brrr.infra.data import TUABDataset`
- **Models**: `from brain_go_brrr.infra.ml_models import EEGPTWrapper`
- **Utils**: `from brain_go_brrr.utils import collate_tuab_batch`
- **Preprocessing**: `from brain_go_brrr.infra.preprocessing import TUEVPreprocessor`

## 🚀 Active Experiments

### EEGPT Linear Probe
- **Purpose**: Fine-tune linear probes on frozen EEGPT backbone
- **Datasets**: TUAB (abnormality), TUEV (6-class events)
- **Docs**: See `eegpt_linear_probe/docs/` for specifications
- **Scripts**: Use `scripts/launch_*.sh` for training

## 📏 Standards

1. **Imports**: Always from `src/brain_go_brrr/`, never relative
2. **Size**: Experiment files should be <200 lines (thin layers)
3. **Documentation**: Essential docs in `docs/`, temporary in `archive/`
4. **Cache**: Use `mne_integration/cache_builder.py` for preprocessing
5. **Testing**: Keep only essential tests, archive one-offs
