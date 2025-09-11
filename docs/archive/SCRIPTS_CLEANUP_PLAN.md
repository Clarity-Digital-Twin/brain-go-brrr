# Scripts Directory Cleanup Plan

## Current Problems
1. **Overlap**: Cache validation exists in both directories
2. **Confusion**: Some experiment-specific scripts in `/scripts/`
3. **Redundancy**: Old/unused scripts still present
4. **Unclear boundaries**: What belongs where?

## Proposed Structure

### `/scripts/` - GENERIC PROJECT UTILITIES
**Purpose**: Cross-project tools, dataset management, CI/CD

```
scripts/
├── data/                    # Dataset download & verification
│   ├── download_datasets.py   # Multi-dataset downloader
│   ├── verify_tuab_dataset.py # TUAB integrity check
│   └── verify_tuev_dataset.py # TUEV integrity check
├── testing/                 # Generic testing utilities
│   ├── test_sleep_analysis.py # Sleep pipeline test
│   └── benchmark_end_to_end.py # Performance benchmark
├── tools/                   # Development tools
│   ├── coverage_report.py    # Coverage analysis
│   └── mypy_daemon.sh        # Type checking
└── ci/                      # CI/CD scripts
    ├── validate_before_push.sh # Pre-push validation
    └── guard_no_oz.sh         # Channel constraint check
```

### `/experiments/eegpt_linear_probe/scripts/` - EXPERIMENT-SPECIFIC
**Purpose**: Scripts that ONLY work with this specific experiment

```
experiments/eegpt_linear_probe/scripts/
├── launch_tuab_training.sh    # Launch TUAB training
├── launch_tuev_training.sh    # Launch TUEV training (with safe params)
└── monitor_training.sh        # Monitor tmux sessions
```

## Scripts to DELETE/MOVE

### DELETE (obsolete/redundant):
- `/scripts/add_archive_banner.py` - One-off, no longer needed
- `/scripts/verify_p2_progress.sh` - Old milestone check
- `/experiments/*/scripts/validate_cache.py` - Redundant with dataset verification
- `/experiments/*/scripts/build_tuev_23ch_cache.sh` - Cache is built automatically now
- `/experiments/*/scripts/launch_tuev_paper_parity.sh` - Outdated (before stability fixes)

### MOVE:
- `/scripts/data/download_tusz.sh` → Keep (generic dataset tool)
- `/scripts/data/monitor_tusz_download.sh` → Delete (too specific)
- `/experiments/*/scripts/run_smoke_test.sh` → Move to `/scripts/testing/`

### KEEP AS-IS:
- `/scripts/testing/debug_tuev_training.py` - Useful debug tool
- `/scripts/tools/run_green_baseline.sh` - CI helper
- `/scripts/tools/run_nightly_tests.sh` - Automated testing

## New Scripts to CREATE

### `/experiments/eegpt_linear_probe/scripts/launch_tuev_safe.sh`
```bash
#!/bin/bash
# Safe TUEV training launcher with WSL2 stability fixes
# Uses parity mode and disabled workers to prevent crashes

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tmux new -d -s tuev_parity "CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  uv run python ../train_tuev_events.py \
  --data_dir data/datasets/tuev \
  --eegpt_checkpoint data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt \
  --use_parity \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 0 \
  --save_dir output/tuev_parity_$TIMESTAMP \
  2>&1 | tee logs/tuev_parity_$TIMESTAMP.log"
```

## Rules Going Forward

1. **Generic utilities** → `/scripts/`
2. **Experiment launchers** → `/experiments/*/scripts/`
3. **One-off debug scripts** → Delete after use
4. **Dataset tools** → `/scripts/data/`
5. **CI/CD tools** → `/scripts/ci/` (new)

## Benefits
- Clear separation of concerns
- No redundancy
- Easy to find scripts
- OSS contributors know where to look
- CI scripts separated from experiment scripts