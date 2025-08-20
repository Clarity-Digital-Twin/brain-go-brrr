# ROOT DIRECTORY CLEANUP REPORT
*Date: August 12, 2025*
*Branch: feature/architecture-refactor*

## ✅ CLEANUP COMPLETED

### Files That MUST Stay in Root:
```
Essential Configuration:
- pyproject.toml         # Package configuration
- uv.lock               # Dependency lock
- Makefile              # Build commands
- LICENSE               # Apache 2.0 license
- .gitignore            # Git ignore rules
- .env                  # Environment variables (if exists)

Python Configuration:
- pytest.ini            # Test configuration
- mypy.ini              # Type checking config
- conftest.py           # Root test fixtures

Container Configuration:
- Dockerfile            # Container definition
- docker-compose.yml    # Container orchestration

Documentation (Essential):
- README.md             # Project readme
- CLAUDE.md             # AI instructions (CRITICAL)
- CHANGELOG.md          # Version history

Current Work (Temporary):
- ARCHITECTURE_REVIEW.md      # Current architecture analysis
- ARCHITECTURE_DEEP_DIVE.md   # Deep architecture analysis
- REFACTORING_PLAN.md        # Refactoring execution plan
- PROJECT_STATUS.md          # Project status tracking

CI/Build:
- mkdocs.yml            # Documentation build
- run_nightly_tests.sh  # Nightly test runner
```

### Files MOVED to Archive:
```
Fix Scripts (6 files) → archive/old_scripts/
- fix_all_mypy_errors_now.py
- fix_all_type_errors.py
- fix_final_errors.py
- fix_lightning_hparams.py
- fix_remaining_type_errors.py
- FIX_ALL_TESTS.py

Test Runners (6 files) → scripts/archive/
- run_test_directly.py
- simple_test_runner.py
- test_simple_coverage.py
- get_coverage_fast.py
- analyze_coverage.py
- run_full_sleep_report.py
- run_sleep_analysis_demo.py

Log Files (7 files) → logs/archive/
- benchmark_results.log
- benchmark_test_results.log
- complete_test_results.log
- gpu_test_results.log
- integration_test_results.log
- unit_test_results.log

Reports (3 files) → archive/
- mypy_errors.txt
- skip-report.txt
- bench-local.json
```

### Archive Consolidation NEEDED:
We have archive directories scattered EVERYWHERE:
```
./archive/                                    # Main archive ✓
./docs/archive/                              # Docs archive
./experiments/eegpt_linear_probe/archive/    # Experiment archive
./logs/archive/                              # Logs archive
./logs/archive_old/                          # Duplicate!
./scripts/archive/                           # Scripts archive
```

## 🎯 RECOMMENDED NEXT STEPS

### 1. Consolidate Archives (Optional but Clean)
```bash
# Move specialized archives under main archive
mkdir -p archive/docs
mkdir -p archive/experiments
mkdir -p archive/logs
mkdir -p archive/scripts

# Move contents (not the directories themselves)
cp -r docs/archive/* archive/docs/
cp -r experiments/eegpt_linear_probe/archive/* archive/experiments/
cp -r logs/archive/* archive/logs/
cp -r logs/archive_old/* archive/logs/
cp -r scripts/archive/* archive/scripts/

# Then remove old archive directories (after verifying)
# rm -rf docs/archive experiments/eegpt_linear_probe/archive logs/archive logs/archive_old scripts/archive
```

### 2. Clean Up Duplicate Config Files
```
We have 3 mypy configs:
- mypy.ini          # Main config ✓
- mypy-fast.ini     # Duplicate?
- mypy_fast.ini     # Duplicate?

Should consolidate to one!
```

### 3. Clean Up Empty Directories
```
- autoreject_cache/  # Check if used
- coverage_html_report/  # Old coverage?
- htmlcov/           # Coverage HTML
- output/            # Empty?
- outputs/           # Duplicate of output?
- stubs/             # Type stubs - needed?
```

## 📊 FINAL ROOT STATUS

### Before Cleanup:
- 78 files in root (39 .md files!)
- Mix of scripts, logs, reports, configs
- No clear organization

### After Cleanup:
- 37 items in root (mostly directories)
- Only essential configs and docs
- Clear purpose for each file

## ✅ ROOT IS NOW CLEAN!

Essential files remain:
- Configuration files (pyproject.toml, Makefile, etc.)
- Critical documentation (README, CLAUDE.md, CHANGELOG)
- Current architecture work (3 new files)
- Required directories (src/, tests/, docs/, etc.)

All clutter moved to appropriate archive locations!
