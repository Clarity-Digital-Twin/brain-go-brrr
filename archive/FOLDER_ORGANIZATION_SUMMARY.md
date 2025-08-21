# Folder Organization Summary

## Archive Structure

The `archive/` folder has been reorganized into a clean, logical structure:

```
archive/
├── development/
│   ├── benchmarks/     # Benchmark results and data
│   ├── configs/        # Old configuration files
│   ├── logs/           # All historical logs (test, extraction, etc.)
│   ├── temp/           # Temporary files
│   └── testing/        # Test results and error reports
├── documentation/
│   ├── eegpt/          # EEGPT-specific docs (fixes, TUAB/TUEV)
│   ├── infrastructure/ # Infrastructure and implementation reports
│   ├── refactoring/    # Refactoring plans and reports
│   └── releases/       # Release documentation
└── scripts/
    ├── old/            # Deprecated scripts
    └── testing/        # Old test scripts
```

## Scripts Structure

The `scripts/` folder is now organized by purpose:

```
scripts/
├── data/               # Data download and verification
│   ├── download_chain.sh
│   ├── download_tuev_secure.exp
│   └── verify_tuev_dataset.py
├── testing/            # Test and benchmark scripts
│   ├── benchmark_end_to_end.py
│   ├── quick_api_test.py
│   ├── run_benchmarks.py
│   └── test_sleep_analysis.py
├── tools/              # Development tools and utilities
│   ├── coverage_report.py
│   ├── fix_pass_statements.py
│   ├── mypy_daemon.sh
│   ├── run_green_baseline.sh
│   ├── run_nightly_tests.sh
│   ├── validate_before_push.sh
│   └── verify_no_skips.sh
└── archive/            # Archived/deprecated scripts
```

## Updated References

The following files were updated to reflect the new script locations:

1. **Makefile**:
   - `scripts/validate_before_push.sh` → `scripts/tools/validate_before_push.sh`

2. **CLAUDE.md**:
   - `scripts/test_sleep_analysis.py` → `scripts/testing/test_sleep_analysis.py`

## Verification

✅ All changes have been tested:
- Import sanity tests pass
- Makefile commands work correctly
- No broken references found

## Benefits

1. **Better Organization**: Scripts are grouped by purpose
2. **Cleaner Archive**: Historical files are properly categorized
3. **Easier Navigation**: Clear folder names indicate content
4. **Maintainability**: Easier to find and manage files
