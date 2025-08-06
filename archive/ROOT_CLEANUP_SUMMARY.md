# Root Directory Cleanup Summary

## ✅ CLEANUP COMPLETED

### Essential Files KEPT

#### Core Project Files
- **CLAUDE.md** - Main project instructions and guidelines
- **README.md** - Project documentation
- **LICENSE** - Apache 2.0 license
- **CHANGELOG.md** - Version history (current: v0.6.0)
- **PROJECT_STATUS.md** - Current training status

#### Configuration Files
- **pyproject.toml** - Python project configuration
- **uv.lock** - Dependency lock file
- **Makefile** - Build and test commands
- **pytest.ini** - Pytest configuration
- **conftest.py** - Test fixtures
- **mypy.ini** - Type checking config
- **mypy-fast.ini** - Fast type checking config (no hanging)
- **.coveragerc** - Coverage configuration
- **.pre-commit-config.yaml** - Pre-commit hooks

#### Docker & CI/CD
- **Dockerfile** - Container configuration
- **docker-compose.yml** - Multi-container orchestration
- **mkdocs.yml** - Documentation site config
- **CI_ALIGNMENT_GUIDE.md** - CI/CD alignment guide
- **run_nightly_tests.sh** - Nightly test runner

#### Git Files
- **.gitignore** - Git ignore patterns
- **.cursorrules** - Cursor editor rules
- **.env** - Environment variables

### Files ARCHIVED

#### Old Logs & Test Results → `/archive/old_logs/`
- cache_build.log
- integration_test_results.log
- test-results.xml
- test_output.log
- test_results.log

#### Benchmark Results → `/archive/benchmark_results/`
- end_to_end_performance.txt

### Directory Structure

```
brain-go-brrr/
├── src/              # Source code
├── tests/            # Test suite
├── experiments/      # EEGPT training experiments
├── docs/             # Documentation
├── data/             # Data and models (git-ignored)
├── configs/          # Configuration files
├── scripts/          # Utility scripts (cleaned)
├── reference_repos/  # Reference implementations
├── archive/          # Archived old files
│   ├── old_logs/
│   ├── old_scripts/
│   ├── extraction_logs/
│   ├── temp_files/
│   └── benchmark_results/
├── logs/             # Training logs (git-ignored)
├── output/           # Training outputs (git-ignored)
├── outputs/          # Lightning logs (git-ignored)
├── htmlcov/          # Coverage reports (git-ignored)
└── autoreject_cache/ # Autoreject cache (empty)
```

## Key Decisions

1. **Kept mypy-fast.ini** - Useful for fast type checking without hanging
2. **Kept all config files** - Essential for development workflow
3. **Archived old logs** - Not needed in root, but preserved for reference
4. **Kept CI/CD files** - Essential for GitHub Actions and local testing

## Notes

- Training currently running at ~5 it/s in tmux session 'eegpt_fast'
- All git-ignored directories (htmlcov, output, outputs) left in place
- autoreject_cache kept as placeholder for future caching