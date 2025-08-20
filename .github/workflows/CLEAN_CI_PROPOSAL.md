# 🎯 CLEAN CI/CD ARCHITECTURE PROPOSAL

## Current Problem: 8 Workflows with Overlapping Responsibilities
You have 8 workflows running similar checks multiple times, burning CI minutes and creating confusion.

## Proposed Solution: 3 Core Workflows + 2 Utility

### 1️⃣ **`ci.yml`** - Main Pipeline (On Every Push/PR)
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, staging, development]
  pull_request:
    branches: [main, staging, development]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # STAGE 1: Quick Checks (fail fast)
  quality:
    name: Code Quality ✨
    runs-on: ubuntu-latest
    steps:
      - Pre-commit hooks (formatting, EOF, trailing whitespace)
      - Ruff (linting + formatting check)
      - Mypy (type checking)
      - Import smoke test
      - Unsafe torch.load check
    
  # STAGE 2: Security (parallel with quality)
  security:
    name: Security Scan 🔒
    runs-on: ubuntu-latest
    steps:
      - Bandit (Python security)
      - Trivy (dependency vulnerabilities)
      - Secret scanning (optional)
  
  # STAGE 3: Tests (after quality passes)
  test:
    name: Test ${{ matrix.python-version }}
    needs: quality
    strategy:
      matrix:
        python-version: [3.11, 3.12]
        os: [ubuntu-latest]
    steps:
      - Unit tests (with coverage)
      - Smoke tests
      - Coverage gate enforcement (64%)
  
  # STAGE 4: Build & Deploy (only on main)
  deploy:
    if: github.ref == 'refs/heads/main'
    needs: [test, security]
    steps:
      - Build Docker image
      - Build docs
      - Publish to PyPI (on tags)
```

### 2️⃣ **`nightly.yml`** - Extended Testing (Scheduled)
```yaml
name: Nightly Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  integration:
    name: Integration Tests
    steps:
      - Full integration test suite
      - GPU tests (if available)
      - Memory leak detection
      - Performance benchmarks with historical comparison
  
  compatibility:
    name: Compatibility Matrix
    strategy:
      matrix:
        python-version: [3.11, 3.12, 3.13]
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - Full test suite on all OS/Python combinations
```

### 3️⃣ **`pr-checks.yml`** - PR-Specific Validation
```yaml
name: PR Validation

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  benchmarks:
    name: Performance Impact
    steps:
      - Run benchmarks
      - Compare with base branch
      - Comment on PR with results
      - Fail if >20% regression
  
  docs:
    name: Documentation
    steps:
      - Build docs
      - Check for broken links
      - Auto-generate API docs diff
```

### 4️⃣ **`claude.yml`** - AI Assistant (Keep As-Is)
Already clean - triggers on `@claude` mentions.

### 5️⃣ **`release.yml`** - Release Automation
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    steps:
      - Run full test suite
      - Build wheels
      - Publish to PyPI
      - Create GitHub release
      - Deploy docs to GitHub Pages
```

## 🗑️ Workflows to DELETE:
- `quality-checks.yml` → Merged into `ci.yml`
- `pre-commit.yml` → Merged into `ci.yml` quality job
- `smoke-test.yml` → Part of `ci.yml` test job
- `benchmarks.yml` → Moved to `pr-checks.yml`
- `auto-doc.yml` → Moved to `pr-checks.yml`

## 🚀 Benefits:

1. **50% Fewer Workflows** (8 → 4-5)
2. **No Duplicate Checks** - Each test runs ONCE
3. **Clear Separation**:
   - CI: Every push (fast feedback)
   - Nightly: Extended testing
   - PR: Additional validation
   - Release: Deployment only

4. **Faster CI**:
   - Concurrency groups cancel old runs
   - Parallel stages (quality || security)
   - Fail-fast on quality issues

5. **Cost Savings**:
   - Less duplicate compute
   - Smart caching
   - Only run expensive tests when needed

## 📊 CI Time Comparison:

**Current (8 workflows):**
- Quality Checks: 2m
- Pre-commit: 19s
- CI Test: 3m
- Smoke Test: Xs
- Total: ~6-8 minutes of compute

**Proposed (1 main CI):**
- Quality: 30s (parallel)
- Security: 30s (parallel)
- Tests: 2-3m (after quality)
- Total: ~3-4 minutes

## 🎯 Implementation Priority:

1. **Phase 1**: Consolidate quality checks
   - Merge pre-commit.yml + quality-checks.yml → ci.yml

2. **Phase 2**: Consolidate testing
   - Merge smoke-test.yml → ci.yml
   - Move nightly to scheduled only

3. **Phase 3**: Optimize
   - Add caching
   - Add concurrency groups
   - Add parallel test splitting

## Example: Consolidated Quality Job

```yaml
quality:
  name: Code Quality
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v3
    
    - name: Set up Python
      run: uv python install 3.11
    
    - name: Install deps
      run: uv sync
    
    - name: All Quality Checks
      run: |
        # One command to rule them all
        make check-all
      # This runs: pre-commit, ruff, mypy, import tests
    
    - name: Upload results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: quality-results
        path: |
          .ruff_cache/
          .mypy_cache/
```

This follows **Single Responsibility Principle** - each workflow has ONE clear purpose, no overlap.