# Scripts Directory - CLEANED

## Essential Scripts (KEPT)

### Performance Testing
- `benchmark_end_to_end.py` - End-to-end performance benchmark on Sleep-EDF data
- `run_benchmarks.py` - EEGPT performance benchmarks

### API/Integration Testing  
- `quick_api_test.py` - Quick API test with mock data
- `test_sleep_analysis.py` - Sleep analysis pipeline test (referenced in CLAUDE.md)

## Archived Scripts

All old/debugging/experimental scripts have been moved to `/archive/` subdirectories:

- `/archive/debugging/` - Old debugging and monitoring scripts
- `/archive/testing/` - Old test scripts (replaced by proper tests in /tests)
- `/archive/setup/` - Setup and fixture creation scripts
- `/archive/old_fixes/` - Scripts used to fix past issues
- `/archive/training/` - Old training launch scripts (now use experiments/eegpt_linear_probe)

## Note

The main training pipeline is now in `/experiments/eegpt_linear_probe/` directory.
Use `train_paper_aligned.py` there for EEGPT linear probe training.