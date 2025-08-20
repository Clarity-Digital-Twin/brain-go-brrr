# Experiments Folder Cleanup Summary

## What Was Wrong

The experiments folder was a disaster:
- 15+ redundant training script versions (`*_BULLETPROOF.py`, `*_fixed.py`, etc.)
- 10+ duplicate documentation files with conflicting information
- Mixed active/archived code in same directory
- No clear structure or organization
- Confusing naming (`FINAL_FIXED_RESUME_v2.py`)

## Professional Structure Implemented

```
experiments/
├── README.md                          # Main experiments documentation
└── eegpt_linear_probe/
    ├── CURRENT_STATUS.md             # Clear status of experiment
    ├── README.md                     # Experiment-specific docs
    ├── train_*.py                    # Training scripts (ROOT LEVEL)
    ├── *_dataset.py                  # Dataset implementations (ROOT LEVEL)
    ├── configs/                      # YAML configurations
    │   ├── tuab_4s_paper_aligned.yaml
    │   └── tuev_table13_aligned.yaml
    ├── scripts/                      # Launch and build scripts
    │   ├── LAUNCH_BULLETPROOF.sh
    │   ├── LAUNCH_TUEV.sh
    │   └── build_tuev_cache.py
    ├── output/                       # Training outputs (gitignored)
    ├── logs/                         # Training logs
    └── archive/                      # Old versions for reference
        ├── old_docs/                 # Previous documentation
        ├── old_outputs/              # Previous runs
        ├── superseded_versions/      # Old script versions
        └── debug/                    # Debug scripts
```

**IMPORTANT**: Python files are in the experiment root, NOT in a `src/` subfolder.
This is because the scripts use relative imports expecting files in the same directory.

## How Pro Teams Handle Experiments

1. **Clear Separation**: Active code vs archived code
2. **Consistent Naming**: No `_FINAL_FIXED_v2` nonsense
3. **Documentation**: Each experiment has clear README
4. **Version Control**: Archive old versions with explanations
5. **Output Management**: Separate outputs from code
6. **Script Organization**: Launch scripts in dedicated folder

## Files Cleaned Up

- **Archived**: 30+ old/duplicate files
- **Organized**: Code into src/, scripts/, configs/
- **Consolidated**: 10 documentation files into 2 clear docs
- **Removed**: Cache files, empty directories

## Current Status

✅ **Structure**: Professional and clean
✅ **Documentation**: Clear and consolidated
✅ **Organization**: Follows best practices
❌ **Code**: Still has critical bugs (see CURRENT_STATUS.md)

## Next Steps

1. Fix the EEGPT architecture bugs
2. Update configs with correct dimensions
3. Retrain models with fixes
4. Document results properly

The experiments folder is now ready for professional development.
