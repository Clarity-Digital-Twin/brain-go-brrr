# CI/CD Analysis - Slop and Redundancy Report

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



## 🔴 CRITICAL ISSUES FOUND

### 1. DISABLED/BROKEN CHECKS (6 disabled checks!)
```yaml
# Pre-commit hooks DISABLED:
- mypy type checking (line 21-32) - "pydantic issue"
- import-linter (line 37-43) - "command not found"
- core-independent check (line 61-67) - "fixing imports"
- detect-secrets (line 69-74) - "missing baseline"
```

**Problem**: Half the checks are disabled! What's the point of CI if it doesn't check anything?

### 2. REDUNDANT EXCLUSIONS
```yaml
# Same exclusions repeated everywhere:
exclude: ^(archive/|scripts/archive/|reference_repos/|experiments/|notebooks/|research/|literature/)
```

**Problem**: Excluding experiments/ from ALL linting means we can't catch the slop!

### 3. USELESS COMPLEXITY
- **3 different branch strategies** (development/staging/main) with minimal difference
- **Matrix testing** only for Python 3.12 (was for multiple versions, now pointless)
- **Benchmark job** that always succeeds with `|| true`

### 4. NO DRIFT PREVENTION
- **NOTHING prevents experiments vs src duplication**
- **NOTHING checks for sys.path.insert hacks**
- **NOTHING enforces using src/ components**

## ✅ WHAT'S ACTUALLY WORKING
- Ruff linting/formatting (but excludes experiments/)
- Basic architecture checks (domain-pure, infra-independent)
- Safe torch.load checker
- Standard pre-commit hooks (yaml, trailing whitespace)

## 🎯 FIXES NEEDED

### Fix 1: Enable All Checks
```yaml
# Enable mypy - fix the actual pydantic issue
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.10.0
  hooks:
    - id: mypy
      additional_dependencies: [pydantic>=2.0]
      files: ^(src|experiments)/  # CHECK BOTH!

# Enable import-linter - install it properly
- id: import-linter
  entry: uv run lint-imports
  language: system
```

### Fix 2: Add Drift Prevention Checks
```bash
# New hook: check-no-parallel-implementations
- id: no-parallel-impl
  name: no-parallel-implementations
  entry: .ci/check_no_parallel_impl.sh
  language: script
  files: ^experiments/.*\.py$

# New hook: check-no-sys-path-insert
- id: no-sys-path
  name: no-sys-path-insert
  entry: bash -c 'if grep -r "sys.path.insert" experiments/ --include="*.py"; then exit 1; fi'
  language: system
```

### Fix 3: Simplify CI Strategy
```yaml
# REMOVE pointless complexity:
- Kill the matrix testing (only 1 Python version anyway)
- Remove staging branch special logic (same as main)
- Make benchmarks actually fail on regression
```

### Fix 4: Include experiments/ in Linting
```yaml
# Change from:
files: ^(src|tests|scripts)/
exclude: ^experiments/

# To:
files: ^(src|tests|scripts|experiments)/
exclude: ^experiments/eegpt_linear_probe/reference_repos/
```

## 🚨 IMMEDIATE ACTIONS

1. **Re-enable mypy** - Fix the pydantic issue properly
2. **Install import-linter** - Add to dev dependencies
3. **Create drift prevention script** - Check experiments imports from src
4. **Remove || true from benchmarks** - Let them fail properly
5. **Consolidate branch strategies** - development=quick, main=full, no staging logic

## 📝 New Check Scripts Needed

### `.ci/check_no_parallel_impl.sh`
```bash
#!/usr/bin/env bash
# Prevent parallel implementations in experiments/

# Check if experiments has its own preprocessing
if find experiments/ -name "*preprocess*.py" -exec grep -l "class.*Preprocessor" {} \; | grep -v "import.*brain_go_brrr"; then
  echo "❌ Parallel preprocessing implementation found!"
  exit 1
fi

# Check if experiments has its own datasets (not shims)
if find experiments/ -path "*/datasets/*.py" -exec wc -l {} \; | awk '$1 > 50 {exit 1}'; then
  echo "❌ Non-shim dataset found in experiments!"
  exit 1
fi

echo "✅ No parallel implementations"
```

### `.ci/check_imports_from_src.sh`
```bash
#!/usr/bin/env bash
# Ensure experiments imports from src

# Count imports from brain_go_brrr vs from experiments
SRC_IMPORTS=$(grep -r "from brain_go_brrr" experiments/ --include="*.py" | wc -l)
EXP_IMPORTS=$(grep -r "from experiments" experiments/ --include="*.py" | grep -v "#" | wc -l)

if [ $EXP_IMPORTS -gt $SRC_IMPORTS ]; then
  echo "❌ Too many internal experiments imports ($EXP_IMPORTS) vs src imports ($SRC_IMPORTS)"
  echo "Experiments should import from src/brain_go_brrr!"
  exit 1
fi

echo "✅ Properly importing from src"
```

## 🔥 THE REAL PROBLEM

**We have CI theatre** - looks like we're checking things but half is disabled and the rest doesn't catch real issues. The experiments/ folder is completely excluded from quality checks which is how the parallel implementation disaster happened.

**Fix this properly or it WILL happen again.**
