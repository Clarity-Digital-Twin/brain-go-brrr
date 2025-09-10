# CI vs Scripts Directory Clarification

## IMPORTANT: Two Different Purposes!

### `/.ci/` - GitHub Actions CI Scripts (DO NOT TOUCH!)
**Purpose**: Automated checks run by GitHub Actions on every push/PR

```
.ci/
├── check_channels_ssot.sh      # Verify channel constraints
├── check_experiments_shims.sh  # Check experiment structure
├── check_meta_schema.sh        # Validate metadata
├── check_no_compat.sh          # No compatibility shims
├── check_no_lightning.sh       # Block PyTorch Lightning (has bug)
├── check_no_parallel_impl.sh   # Prevent duplicate implementations
├── check_no_sys_path.sh        # No sys.path hacks
└── check_script_arguments.sh   # Validate script args
```

**Used by**: `.github/workflows/ci.yml`
**When run**: Automatically on git push/PR
**Who uses**: GitHub Actions ONLY

### `/scripts/` - Developer Helper Scripts
**Purpose**: Manual utilities for developers to run locally

```
scripts/
├── data/                        # Dataset management
│   ├── download_datasets.py    # Download Sleep-EDF, etc.
│   ├── verify_tuab_dataset.py  # Check TUAB integrity
│   └── verify_tuev_dataset.py  # Check TUEV integrity
├── testing/                     # Testing utilities
│   ├── test_sleep_analysis.py  # Test sleep pipeline
│   ├── benchmark_*.py          # Performance tests
│   └── debug_tuev_training.py  # Debug training issues
├── tools/                       # Development tools
│   ├── coverage_report.py      # Generate coverage
│   └── mypy_daemon.sh          # Type checking
├── validate_before_push.sh     # Run ALL checks locally (calls make)
└── guard_no_oz.sh              # Manual channel check
```

**Used by**: Developers manually
**When run**: `./scripts/validate_before_push.sh` before pushing
**Who uses**: Developers locally

## Key Differences

| Aspect | `.ci/` | `/scripts/` |
|--------|--------|-------------|
| Purpose | Automated CI checks | Manual developer tools |
| Execution | GitHub Actions | Developer runs manually |
| Scope | Specific checks | Broader utilities |
| Modification | Carefully - affects CI/CD | Safe to modify |
| Examples | `check_no_lightning.sh` | `test_sleep_analysis.py` |

## Common Confusion Points

1. **guard_no_oz.sh** - This is a DEVELOPER script, not CI!
   - Location: `/scripts/` ✅
   - Purpose: Manual check for Oz channel
   
2. **validate_before_push.sh** - Developer helper, not CI!
   - Location: `/scripts/` ✅  
   - Purpose: Runs `make check-all` locally
   
3. **check_no_lightning.sh** - This IS CI!
   - Location: `/.ci/` ✅
   - Purpose: Blocks PyTorch Lightning in CI

## Best Practices

### For CI Scripts (`.ci/`)
- DO NOT move or rename without updating `.github/workflows/`
- Keep focused on single checks
- Must exit with proper codes (0=pass, 1=fail)
- Should be fast (<30 seconds)

### For Developer Scripts (`/scripts/`)
- Can be more complex/interactive
- Should have help text
- Can take longer to run
- Should be documented in scripts/README.md

## What Goes Where?

**New CI check?** → `.ci/check_*.sh` + update `.github/workflows/ci.yml`
**Dataset tool?** → `/scripts/data/`
**Testing utility?** → `/scripts/testing/`
**Dev helper?** → `/scripts/tools/`
**Training launcher?** → `/experiments/*/scripts/`

## DO NOT MIX THEM UP!
- CI scripts are for GitHub Actions
- Developer scripts are for local use
- They serve different purposes
- Keep them separate!