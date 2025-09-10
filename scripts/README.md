# Scripts Directory

This directory contains **generic utility scripts** for the Brain-Go-Brrr project.

## Directory Structure (FIXED Sep 10, 2025)

```
scripts/
├── data/               # Dataset management
│   ├── download_*.py   # Download various datasets
│   └── verify_*.py     # Verify dataset integrity
├── testing/            # Testing and debugging utilities
│   ├── test_*.py       # Component tests
│   ├── benchmark_*.py  # Performance benchmarks
│   └── debug_*.py      # Debugging tools
├── tools/              # Development tools
│   ├── coverage_*.py   # Test coverage tools
│   └── mypy_*.sh       # Type checking tools
├── validate_before_push.sh  # Run all checks before git push
└── guard_no_oz.sh           # Check for forbidden Oz channel
```

## IMPORTANT: CI Scripts are in `/.ci/` NOT here!
The `.ci/` directory contains GitHub Actions scripts. This `/scripts/` directory is for developer utilities only!

## Important Notes

### What Goes Here
- Generic utilities used across the project
- Dataset download/verification scripts
- Testing utilities that aren't experiment-specific
- Development tools

### What DOESN'T Go Here  
- **Training scripts** → Use `/experiments/`
- **Model implementations** → Use `/src/brain_go_brrr/`
- **Experiment-specific scripts** → Use `/experiments/*/scripts/`
- **Temporary debug scripts** → Delete after use

### Training is in `/experiments/`
- TUAB training: `/experiments/eegpt_linear_probe/train_tuab_mne.py`
- TUEV training: `/experiments/eegpt_linear_probe/train_tuev_events.py`
- Launch scripts: `/experiments/eegpt_linear_probe/scripts/`

## Archive Note
Old scripts have been moved to archive subdirectories for reference but should not be used.
