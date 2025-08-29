# CI/CD Setup and Developer Workflow

## Overview

This project uses a **clear separation** between local development tooling and CI/CD checks:

- **Local Development**: Uses pre-commit hooks for auto-fixing and immediate feedback
- **CI/CD**: Runs checks directly with `uv run` in check-only mode (no auto-fixing)

## Local Development Setup

### 1. Install Pre-commit Hooks

```bash
# Install pre-commit hooks for your local git repository
pre-commit install

# Optional: Set up pre-commit cache location to avoid permission issues
export PRE_COMMIT_HOME=./.pre-commit-cache
```

### 2. Local Workflow

When you commit locally:
1. Pre-commit hooks run automatically
2. **Auto-fixing is ENABLED** - ruff will fix formatting/linting issues
3. If fixes are made, you'll need to stage the changes and commit again
4. This provides immediate feedback and maintains code quality

### 3. Manual Checks

You can also run checks manually:

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific tools directly
uv run ruff format src/           # Auto-format
uv run ruff check src/ --fix      # Auto-fix linting
uv run mypy src/brain_go_brrr     # Type checking
```

## CI/CD Pipeline

### Philosophy

**CI/CD NEVER auto-fixes code**. It only validates that code meets standards.

### What CI Runs

The Code Quality job runs these checks **directly** (not through pre-commit):

1. **Format Check**: `uv run ruff format --check`
   - Fails if code isn't properly formatted
   - Does NOT modify files

2. **Lint Check**: `uv run ruff check`
   - Fails if linting issues exist
   - Does NOT auto-fix

3. **Type Check**: `uv run mypy --config-file mypy.ini src/brain_go_brrr`
   - Uses the project's synced environment
   - Has access to all dependencies

4. **Architecture Guards**: Custom bash scripts
   - No parallel implementations
   - No sys.path.insert hacks
   - Channel naming consistency
   - etc.

### Why This Separation?

1. **No Surprises**: CI never modifies code mid-flight
2. **Single Environment**: CI uses `uv sync --dev`, same as local
3. **Version Consistency**: No drift between pre-commit's isolated env and project env
4. **Clear Responsibility**:
   - Developers fix issues locally (with auto-fix help)
   - CI validates that standards are met

## Common Issues and Solutions

### Issue: CI fails on formatting but local passes

**Solution**: Ensure you're using the same ruff version:
```bash
uv run ruff --version  # Should match CI
```

### Issue: MyPy behaves differently locally vs CI

**Solution**: Always run mypy through `uv run`:
```bash
uv run mypy src/brain_go_brrr  # NOT just 'mypy'
```

### Issue: Pre-commit is slow locally

**Solution**: Set up cache directory:
```bash
export PRE_COMMIT_HOME=./.pre-commit-cache
```

## Summary

- **Local**: Pre-commit with auto-fix for developer convenience
- **CI**: Direct tool execution, check-only, no mutations
- **Both**: Use same tool versions via `uv` package manager

This ensures consistency while maintaining developer productivity.
