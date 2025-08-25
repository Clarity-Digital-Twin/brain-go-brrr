# Experiments Directory

This directory contains experimental training scripts and research implementations.

## Structure

```
experiments/
├── README.md                      # This file
└── eegpt_linear_probe/           # EEGPT linear probing experiments
    ├── README.md                 # Experiment documentation
    ├── CURRENT_STATUS.md         # Current issues and fixes needed
    ├── train_*.py                # Training scripts (in root)
    ├── *_dataset.py              # Dataset implementations (in root)
    ├── configs/                  # Training configurations
    ├── scripts/                  # Launch and build scripts
    ├── output/                   # Training outputs (gitignored)
    ├── logs/                     # Training logs
    └── archive/                  # Old versions (reference only)
```

**NOTE**: Training scripts and dataset files are in the experiment root directory,
not in a `src/` subfolder. This is intentional - the scripts have relative imports
expecting files in the same directory.

## Active Experiments

### 1. EEGPT Linear Probe  
- **Status**: ✅ FIXED & RUNNING with BCEWithLogitsLoss
- **Purpose**: Fine-tune linear probes on frozen EEGPT for TUAB/TUEV
- **Progress**: Batch 98/29,143 - Loss healthy (~0.55-0.65), no longer zero!
- **See**: `eegpt_linear_probe/README.md`

## Guidelines

1. **Version Control**: Archive old versions, don't delete
2. **Documentation**: Each experiment needs its own README
3. **Outputs**: Use `outputs/` for results (gitignored)
4. **Scripts**: Keep launch scripts in `scripts/`
5. **Configs**: YAML configs in `configs/`

## Standards

- Use descriptive names (not `train_v2_final_FIXED.py`)
- Document findings in experiment README
- Archive failed attempts with explanations
- Keep active code clean and minimal
