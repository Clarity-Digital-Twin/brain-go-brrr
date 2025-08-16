# Release Notes - v0.6.1

**Release Date**: August 16, 2025
**Status**: STABLE ✅

## 🎯 v0.6.1 - CI/CD Pipeline Fixed & Documentation Cleanup

This release fixes all CI/CD pipeline issues and provides a clean, organized repository structure ready for production deployment.

### 🔧 Critical Fixes

#### CI/CD Pipeline Stabilization
- **Ruff 0.6.9 Pinned**: Consistent formatter version across all environments
- **Line Endings Fixed**: LF enforced via `.gitattributes` - no more CRLF conflicts
- **Pre-commit Non-destructive**: CI checks without modifying files
- **Git Hook Universal**: Pre-commit hook works from both Windows and WSL

#### Package Structure Fixes
- **infra.data Module**: Fixed missing module imports
- **Safe torch.load**: Security wrapper for all model loading
- **Proper .gitignore**: Source code vs data payload separation

### 📚 Documentation Updates

- **Root Cleanup**: Moved 12+ development docs to `docs/archive/development-history/`
- **Updated Status**: PROJECT_STATUS.md reflects current state
- **CHANGELOG**: Complete history with all recent changes
- **Branch Sync**: main/staging/development all aligned at same commit

### 🚀 What's Working

| Component | Status | Performance |
|-----------|--------|------------|
| **YASA Sleep Staging** | ✅ Ready | 87.46% accuracy |
| **Quality Control** | ✅ Ready | >95% bad channel detection |
| **Abnormality Detection** | 🔄 Training | Target: 0.869 AUROC |
| **API Endpoints** | ✅ Ready | <100ms response time |
| **CI/CD Pipeline** | ✅ Fixed | All checks passing |

### 📦 Installation

```bash
# With pip
pip install brain-go-brrr==0.6.1

# With uv (recommended)
uv pip install brain-go-brrr==0.6.1

# Development setup
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
make dev-setup
```

### 🔄 Migration from v0.6.0

No breaking changes - just update and enjoy the stable CI/CD:

```bash
git pull origin main
uv sync
make test  # Should all pass!
```

### 🐛 Bug Fixes in This Release

- Fixed VS Code git hook "pre-commit not found" error
- Resolved formatter version conflicts in CI
- Fixed missing `brain_go_brrr.infra.data` module imports
- Corrected torch.load security warnings
- Eliminated CRLF/LF line ending conflicts

### 📊 Statistics

- **Tests**: 812 passing (66.85% coverage)
- **Linting**: 0 errors (ruff 0.6.9)
- **Type Checking**: 0 errors (mypy strict)
- **Pre-commit**: All hooks passing
- **CI/CD**: Green across all branches

### 🙏 Acknowledgments

Thanks to the community for patience while we sorted out the CI/CD pipeline issues. Special thanks to contributors who helped identify the formatter version mismatch issue.

### 📝 Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete history.

---

**Next Release**: v0.7.0 will include completed EEGPT training with 0.869 AUROC target achieved.
