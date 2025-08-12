# Final Root Directory Organization

## ✅ CLEANED & ORGANIZED

### Cache Directories Cleaned
- ✅ Removed `__pycache__`
- ✅ Removed `.mypy_cache` and `.mypy_cache_strict` (128MB freed)
- ✅ Removed `.pytest_cache`
- ✅ Removed `.ruff_cache`
- ✅ Removed `.benchmarks` (empty)
- ✅ Cleaned `htmlcov/*` (kept dir for CI)
- ✅ Cleaned `autoreject_cache/*` (kept dir)
- ✅ Cleaned `output/*` and `outputs/*`

### Documentation Consolidated
- ✅ Moved redundant cleanup summaries to `/archive/`
- ✅ Kept essential docs in root:
  - **CLAUDE.md** - Primary instructions
  - **README.md** - Project overview
  - **PROJECT_STATUS.md** - Current status
  - **CI_ALIGNMENT_GUIDE.md** - CI/CD guide
  - **CHANGELOG.md** - Version history

### Logs Organized
- ✅ Archived old log directories (40 subdirs) to `logs/archive_old/`
- ✅ Kept only today's logs accessible

## 📁 Clean Directory Structure

```
brain-go-brrr/
├── .github/          # GitHub workflows (7 workflows)
├── .claude/          # Claude settings (keep for Claude Code)
├── .venv/            # Python virtual environment
│
├── src/              # Source code
├── tests/            # Test suite
├── stubs/            # Type stubs
│
├── experiments/      # Active experiments
│   └── eegpt_linear_probe/  # Current training
│
├── configs/          # Global configs (2 files)
├── scripts/          # Utility scripts (cleaned, 4 files)
│
├── docs/             # Documentation
│   ├── 00-overview/
│   ├── 01-architecture/
│   ├── 02-implementation/
│   ├── 03-api/
│   ├── 04-testing/
│   ├── 05-deployment/
│   ├── 06-clinical/
│   └── archive/      # Old docs
│
├── data/             # Data & models (git-ignored)
│   ├── cache/        # Dataset caches
│   ├── datasets/     # Raw datasets
│   └── models/       # Model weights
│
├── reference_repos/  # Reference implementations
│   ├── EEGPT/
│   ├── autoreject/
│   ├── mne-python/
│   └── yasa/
│
├── literature/       # Research papers
│   ├── markdown/
│   └── pdfs/
│
├── archive/          # Archived files
│   ├── old_logs/
│   ├── old_scripts/
│   ├── extraction_logs/
│   ├── temp_files/
│   └── benchmark_results/
│
├── logs/             # Training logs (git-ignored)
│   └── archive_old/  # Old logs
│
├── examples/         # Example scripts
│
├── output/           # Training outputs (git-ignored, empty)
├── outputs/          # Lightning outputs (git-ignored, empty)
├── htmlcov/          # Coverage reports (git-ignored, empty)
└── autoreject_cache/ # AR cache (empty placeholder)
```

## 🔧 Configuration Files (All Essential)

### Python/Package
- `pyproject.toml` - Package configuration
- `uv.lock` - Dependency lock
- `pytest.ini` - Test configuration
- `conftest.py` - Test fixtures
- `.coveragerc` - Coverage config

### Type Checking
- `mypy.ini` - Standard type checking
- `mypy-fast.ini` - Fast mode (no hanging)

### CI/CD & Docker
- `Makefile` - Build commands
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-container
- `mkdocs.yml` - Documentation site
- `run_nightly_tests.sh` - Nightly tests
- `.pre-commit-config.yaml` - Pre-commit hooks

### Git & Editor
- `.gitignore` - Ignore patterns
- `.cursorrules` - Cursor editor
- `.env` - Environment variables

## 🎯 Key Improvements Made

1. **Freed ~130MB** by removing cache directories
2. **Consolidated docs** - removed redundant summaries
3. **Archived old logs** - cleaner logs directory
4. **Removed empty dirs** - `.benchmarks`
5. **Cleaned outputs** - ready for fresh runs

## ⚠️ What We KEPT (Important!)

- `.github/` - GitHub Actions workflows
- `.claude/` - Claude Code settings
- `.venv/` - Virtual environment
- All config files - needed for development
- Empty placeholder dirs - needed by tools

## 📊 Training Status

- **Session**: `tmux attach -t eegpt_fast`
- **Progress**: ~20% (continuing at ~5.5 it/s)
- **Target AUROC**: 0.869