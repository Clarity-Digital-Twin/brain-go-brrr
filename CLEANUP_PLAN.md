# Root Directory Cleanup Plan

## Current Issues
- Multiple test scripts in root (test_*.py)
- Debug scripts in root (debug_dataloader.py)
- Training scripts in root (run_hinton.sh, launch_fixed_training.sh)
- Cache file in root (tuab_index.json)
- Log file in root (training_paper_aligned.log)

## Proposed Organization

### 1. Scripts Organization
```
scripts/
├── training/
│   ├── launch_fixed_training.sh
│   └── run_hinton.sh
├── debug/
│   ├── debug_dataloader.py
│   ├── test_fixed_dataset.py
│   ├── test_window_sizes.py
│   └── test_ci_minimal.py
└── data_prep/
    └── build_tuab_index.py (already exists)
```

### 2. Cache/Index Files
```
data/
├── cache/
│   └── tuab_index.json (move from root)
└── logs/
    └── training_paper_aligned.log (move from root)
```

### 3. Files to Keep in Root
- Essential configs: pyproject.toml, Makefile, Dockerfile, etc.
- Documentation: README.md, CHANGELOG.md, CLAUDE.md, etc.
- Python configs: pytest.ini, mypy.ini, conftest.py
- Git/CI configs: .gitignore, .pre-commit-config.yaml

## Implementation Steps

1. Create new directories
2. Move files carefully
3. Update any hardcoded paths
4. Test that nothing breaks
5. Update .gitignore if needed