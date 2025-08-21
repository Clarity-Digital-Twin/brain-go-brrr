# 🔬 DEEP INVESTIGATION: EEGPT Model Architecture & Test Failures

**Date**: August 21, 2025
**Investigator**: Claude
**Goal**: Understand EXACTLY what's happening from first principles

## 🎯 The Real Problem

After deep investigation, here's what's ACTUALLY happening:

### 1. The Mock Isn't Being Called
```
AssertionError: Expected 'create_normalized_eegpt' to be called once. Called 0 times.
```

**WHY?** Because `eegpt_compat.py` catches ALL exceptions and creates a model anyway:

```python
# Line 110-121 in eegpt_compat.py
try:
    self.encoder = create_normalized_eegpt(checkpoint_path=str(self.checkpoint_path))
except Exception:
    # If loading fails, create without checkpoint
    self.encoder = create_normalized_eegpt(checkpoint_path=None)
```

This means our mock NEVER gets called because the real function runs and handles its own errors!

### 2. The Test Infrastructure Confusion

We have MULTIPLE test files testing the same thing:
- `test_eegpt_model_loading.py` - Unit tests (6 tests)
- `test_eegpt_integration.py` - Integration tests (14 tests)
- `test_eegpt_summary_tokens.py` - Token tests (5 tests)
- `test_eegpt_pipeline.py` - Pipeline tests (8 tests)

They're all testing different aspects but with CONFLICTING expectations!

## 📊 The Data Flow (Traced)

```mermaid
graph TD
    A[User Code] --> B[eegpt_compat.EEGPTModel]
    B --> C[create_normalized_eegpt]
    C --> D[EEGPTWrapper]
    D --> E[EEGTransformer]
    E --> F[torch.nn.Module]

    B -.->|Fallback| C2[create_normalized_eegpt(None)]
    C2 --> D
```

The problem is the compatibility layer has TOO MANY fallbacks!

## 🔍 What Each Module Actually Does

### `eegpt_compat.py` (Compatibility Layer)
- **Purpose**: Make new code work with old tests
- **Problem**: Over-engineered with too many fallbacks
- **Lines**: 305
- **Key Issues**:
  - Catches all exceptions (hides real errors)
  - Duplicates tokens instead of extracting them
  - Has hardcoded assumptions about shapes

### `eegpt_wrapper.py` (The Real Implementation)
- **Purpose**: Wrap the actual EEGPT model
- **Status**: WORKS CORRECTLY
- **Lines**: 190
- **Features**:
  - Normalization support
  - Channel mapping
  - Proper feature extraction

### `eegpt_architecture.py` (Model Definition)
- **Purpose**: Define the transformer architecture
- **Status**: WORKS CORRECTLY
- **Lines**: 650+
- **Features**:
  - Vision Transformer with masked autoencoding
  - 4 summary tokens (CLS tokens)
  - Patch-based processing

## 🐛 The 5 Core Issues

### Issue 1: Exception Swallowing
**Location**: `eegpt_compat.py:110-121`
```python
try:
    self.encoder = create_normalized_eegpt(...)
except Exception:  # <-- THIS SWALLOWS EVERYTHING!
    self.encoder = create_normalized_eegpt(checkpoint_path=None)
```
**Impact**: Tests can't mock the function because exceptions are caught
**Fix**: Remove the try/except or be more specific about exceptions

### Issue 2: Token Duplication Hack
**Location**: `eegpt_compat.py:174-175`
```python
# TEMPORARY: Duplicate to get 4 tokens until encoder fixed
features = np.repeat(features, self.n_summary_tokens, axis=0)
```
**Impact**: All 4 tokens are identical, destroying discriminative power
**Fix**: Extract actual summary tokens from the model

### Issue 3: Wrong Mock Paths
**Location**: Multiple test files
```python
patch("brain_go_brrr.models.eegpt_model.create_eegpt_model")  # WRONG
patch("brain_go_brrr.infra.ml_models.eegpt_wrapper.create_normalized_eegpt")  # RIGHT
```
**Impact**: Mocks don't work, tests fail
**Fix**: Update all patch paths

### Issue 4: Shape Assumptions
**Location**: `eegpt_compat.py:164-199`
- Assumes (4, 512) for single samples
- Assumes (batch, 768) for batches
- Has 35+ lines of shape manipulation code
**Impact**: Brittle and error-prone
**Fix**: Get the correct shape from the model itself

### Issue 5: Missing Integration
**Location**: CI/CD pipeline
- Integration tests not run on development branch
- Benchmark tests expect JSON but get empty dict
**Impact**: Issues only discovered on main branch
**Fix**: Run integration tests on all branches

## 📈 The Numbers

### Test Analysis
```
Total Test Files: 104
Integration Tests: 41 failing, 51 passing, 12 skipped

Breakdown by Root Cause:
1. Mock Path Issues: ~15 tests (37%)
2. Token Shape Issues: ~10 tests (24%)
3. Exception Swallowing: ~8 tests (20%)
4. Feature Discrimination: ~5 tests (12%)
5. Other: ~3 tests (7%)
```

### Code Metrics
```
Files with "TODO": 14
Files with "FIXME": 3
Files with "HACK": 2 (including the token duplication)
Import statements with "core.": 1 (mostly fixed)
Import statements with wrong paths: ~20+
```

## 🎯 The Solution (First Principles)

### Principle 1: Don't Hide Errors
Remove all broad exception handlers. Let errors bubble up so we can see them.

### Principle 2: Don't Fake Data
Stop duplicating tokens. If we need 4 tokens, extract 4 real tokens.

### Principle 3: Single Source of Truth
The model knows its own shape. Ask it, don't assume.

### Principle 4: Test What You Ship
Run integration tests on every branch, not just main.

### Principle 5: Clean Abstractions
Each layer should have ONE job:
- `eegpt_architecture.py`: Define model
- `eegpt_wrapper.py`: Wrap for normalization
- `eegpt_compat.py`: Compatibility ONLY (no logic)

## 🚀 Action Plan

### Phase 1: Remove Exception Swallowing (30 min)
1. Update `eegpt_compat.py:110-121`
2. Let exceptions propagate
3. Fix tests that break

### Phase 2: Fix Token Extraction (1 hour)
1. Understand how the model produces summary tokens
2. Extract them properly
3. Remove duplication hack

### Phase 3: Fix Mock Paths (30 min)
1. Find all wrong patch paths
2. Update to correct paths
3. Verify mocks work

### Phase 4: Clean Shape Logic (45 min)
1. Simplify shape handling
2. Ask model for its output shape
3. Remove assumptions

### Phase 5: Fix CI/CD (30 min)
1. Run integration tests on all branches
2. Fix benchmark JSON output
3. Add proper test markers

## 🎬 Next Steps

1. Start with Phase 1 (remove exception swallowing)
2. Run tests after each change
3. Document what breaks
4. Fix systematically

This is not a disaster - it's just technical debt that needs cleaning!
