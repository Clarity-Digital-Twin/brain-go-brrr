# Scripts Directory Cleanup Summary

## ✅ CLEANUP COMPLETED

### Before
- **7 subdirectories** with mixed debugging/testing scripts
- **43 scripts** in root directory  
- Confusing mix of old experiments, debugging tools, and actual utilities

### After  
- **4 essential scripts** kept in root
- **47 scripts** archived in organized categories
- **Clean, documented structure**

## Essential Scripts (KEPT)

1. **test_sleep_analysis.py** - Referenced in CLAUDE.md for sleep analysis testing
2. **benchmark_end_to_end.py** - Performance benchmarking  
3. **run_benchmarks.py** - EEGPT benchmarks
4. **quick_api_test.py** - API testing utility

## Archive Organization

```
archive/
├── debugging/        # 17 files - Old debugging/monitoring scripts
├── old_fixes/        # 5 files - Scripts used to fix past issues  
├── setup/            # 6 files - Setup and fixture creation
├── testing/          # 14 files - Old test scripts
└── training/         # 2 files - Old training launchers
```

## Key Decision: Why These Were Kept

- **test_sleep_analysis.py**: Explicitly referenced in CLAUDE.md as example
- **benchmark_end_to_end.py**: Validates 20-min EEG in <2 min requirement
- **run_benchmarks.py**: Performance testing utility
- **quick_api_test.py**: API integration testing

## Note

The main EEGPT training pipeline is in `/experiments/eegpt_linear_probe/`.
Training is currently running at ~5 it/s in tmux session 'eegpt_fast'.