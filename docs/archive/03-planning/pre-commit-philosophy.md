# Pre-commit Hook Philosophy

## TL;DR

**Local hooks should be fast (< 2 seconds), CI should be strict.**

## Why This Approach?

We've optimized our pre-commit hooks for developer happiness while maintaining code quality through strict CI checks. Here's the philosophy:

### Local Pre-commit Hooks (Fast & Friendly)

- **Goal**: Never block developers for more than 2 seconds
- **What runs**: Only auto-fixable formatting and critical lint rules
- **Auto-fix everything**: Format and simple lint issues are fixed automatically

### CI Pipeline (Strict & Comprehensive)

- **Goal**: Ensure code quality before merging
- **What runs**: Full linting, strict type checking, all tests, security scans
- **No compromises**: All rules enabled, no warnings ignored

## What Each Layer Does

### 1. Pre-commit Hooks (< 2 seconds)

```yaml
# Always auto-fix, never block
- ruff format (auto-formats code)
- ruff check --select I,E,F,W,B,UP (only fast, critical rules)
- mypy (only on changed files, ignoring missing imports)
- file checks (trailing whitespace, merge conflicts, large files)
```

### 2. CI Quality Checks

```yaml
# Full strictness
- ruff check --select ALL (all rules enabled)
- mypy src (strict mode, no missing imports allowed)
- bandit (security scanning)
- pydocstyle (docstring coverage)
```

### 3. CI Tests

```yaml
# Comprehensive testing
- pytest (all tests including slow/integration)
- coverage reporting
- performance benchmarks
```

## Common Scenarios

### "I just want to commit my WIP"
✅ Fast pre-commit hooks let you commit in seconds. Your code is auto-formatted and basic issues are fixed.

### "I'm pushing to a feature branch"
✅ CI runs full checks. You'll see any issues but won't block others.

### "I'm merging to main"
❌ CI must pass. All quality gates enforced.

## Configuring Your Environment

### Install pre-commit hooks
```bash
pre-commit install
```

### Skip hooks temporarily (emergency only!)
```bash
git commit --no-verify -m "WIP: emergency fix"
```

### Run hooks manually
```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run mypy --all-files
```

## Adding New Hooks

Before adding a new pre-commit hook, ask:

1. Can it run in < 0.5 seconds on 100 files?
2. Can it auto-fix issues (not just report them)?
3. Is it catching critical issues (not style preferences)?

If any answer is "no", put it in CI instead.

## Hook Maintenance

- Review hook performance monthly
- Remove hooks that frequently block commits
- Move slow hooks to CI
- Keep total pre-commit time under 2 seconds

## Philosophy

> "The best pre-commit hook is one developers don't notice."

Fast, automatic, and helpful - never annoying or blocking.
