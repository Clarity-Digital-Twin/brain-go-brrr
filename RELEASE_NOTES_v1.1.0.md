# Release v1.1.0 - Architecture Unification & Security Hardening

## Overview

This release represents a major architectural overhaul, eliminating parallel implementations between `src/` and `experiments/`, adding HIPAA-compliant PHI protection, and establishing professional CI/CD practices. The codebase is now ready for serious TUAB and TUEV model training.

## 🏗️ Major Architectural Improvements

### Unified Architecture (No More Parallel Implementations)
- **FIXED**: Eliminated duplicate dataset/model implementations between `src/` and `experiments/`
- All experiments now use components from `src/` (single source of truth)
- Normalization centralized in EEGPT wrapper (datasets emit raw mV only)
- Channel validation enforces correct ordering per dataset
- META schema unified across all datasets

### CI/CD Professional Setup
- Removed pre-commit from CI (no more auto-mutations in CI)
- CI runs tools directly with `uv run` in check-only mode
- Perfect alignment between local and CI environments
- Added architecture drift guards to prevent future divergence
- Python version constrained to 3.11-3.12 (3.13 blocked due to scipy issues)

## 🔒 Security & Privacy Enhancements

### PHI Protection (HIPAA Compliance)
- Implemented path masking for all file paths in logs
- Added `PathMaskingFilter` to prevent PHI leakage
- Integrated masking in CLI, API, and all logging outputs
- Unit tests verify no paths leak in production logs

### Security Fixes
- Added `weights_only=True` to all `torch.load()` calls
- AST-based checker enforces safe loading patterns
- Escape hatch with `# nosec:weights_only` for trusted checkpoints

## 🐛 Critical Fixes

### Preprocessing & Data Pipeline
- Fixed TUAB channel mapping (19 channels, no Fz)
- Fixed TUEV channel mapping (20 channels, includes Fz, no Fpz)
- Corrected T3→T7, T4→T8, T5→P7, T6→P8 mappings
- Fixed 4-second window extraction for EEGPT
- Removed TCP/bipolar code from TUEV (was causing errors)

### CI/CD & Dependencies
- Fixed UTC import compatibility (Python 3.11.0-3.11.8 vs 3.11.9+)
- Disabled ruff's UP017 rule to prevent UTC mutations
- Fixed mypy decorator issues with FastAPI/Typer
- Resolved all type checking errors in CI

## 📊 By The Numbers

- **420** significant commits (feat/fix/refactor)
- **558** files changed
- **25,070** insertions, **58,814** deletions (massive cleanup)
- **~34,000** lines of duplicate/dead code removed
- **100%** CI/CD green (except coverage, tracking separately)

## 🚀 What's Next

With the architecture unified and CI/CD professional, we're ready to:
1. Train TUAB linear probe (target: 87% AUROC)
2. Train TUEV 6-class classifier
3. Deploy production API with confidence

## Installation

```bash
pip install brain-go-brrr==1.1.0
```

Or using uv (recommended):
```bash
uv pip install brain-go-brrr==1.1.0
```

## Breaking Changes

None. Full backward compatibility maintained.

## Contributors

Thanks to everyone who provided feedback on the CI/CD setup and architecture improvements.

---

Full Changelog: [v1.0.0...v1.1.0](https://github.com/Clarity-Digital-Twin/brain-go-brrr/compare/v1.0.0...v1.1.0)
