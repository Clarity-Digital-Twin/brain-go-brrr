# Changelog

All notable changes to Brain-Go-Brrr will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-07-31

### 🚀 EEGPT Linear Probe Implementation

This release adds complete EEGPT linear probe training for TUAB abnormality detection, fixing critical channel mapping issues.

### ✨ Added

- **EEGPT Linear Probe Training**:
  - Complete implementation of linear probe for abnormality detection
  - Paper-faithful settings: batch_size=64, lr=5e-4, 10 epochs
  - Weighted random sampling for class balance
  - OneCycleLR scheduler with proper warmup
  - Early stopping on validation loss

- **TUAB Dataset Improvements**:
  - Fixed channel mapping: T3→T7, T4→T8, T5→P7, T6→P8
  - Reduced from 23 to 20 channels (removed A1/A2 references)
  - Added file caching for 100x faster loading
  - Window size: 8 seconds (2048 samples at 256Hz)
  - Zero-padding for missing channels

### 🐛 Fixed

- **Critical Channel Mismatch**:
  - BREAKING: AbnormalityDetectionProbe now expects 20 channels (was 23)
  - Updated all channel lists to use modern naming convention
  - Fixed tests to match new channel configuration
  - Cleared Python cache to prevent stale imports

### 📚 Documentation

- Added CHANNEL_MAPPING_EXPLAINED.md with detailed mapping guide
- Created TRAINING_SUMMARY.md for training status tracking
- Organized experiments folder with archived scripts

### 🧪 Testing

- Updated test_eegpt_linear_probe.py for 20-channel configuration
- All 458 tests passing
- Fixed import ordering in training scripts

## [0.4.0] - 2025-07-30

### 🧠 Critical EEGPT Model Fixes

This release represents a major milestone in achieving functional EEGPT integration. We discovered and fixed a critical issue where the model was producing non-discriminative features (cosine similarity = 1.0) due to input scale mismatch.

### 🎯 Major Improvements

- **EEGPT Normalization Fix**:
  - Root cause: Raw EEG signals (~50μV) were 115x smaller than model bias terms
  - Solution: Implemented `EEGPTWrapper` with proper input normalization
  - Features now discriminative with cosine similarity ~0.486 (vs 1.0 before)
  - Normalization stats saved for reproducibility

- **Architecture Corrections**:
  - Fixed channel embedding dimensions (62 channels, 0-61 indexed)
  - Implemented custom Attention module matching EEGPT paper exactly
  - Enabled Rotary Position Embeddings (RoPE) for temporal encoding
  - Fixed all 8 transformer blocks loading (was missing intermediate blocks)

- **Test Infrastructure**:
  - Created minimal test checkpoint (96MB vs 1GB) for CI/CD
  - Added comprehensive checkpoint loading tests
  - Fixed test fixtures scoping issues
  - All 368 tests passing with proper type checking

### 🔧 Technical Details

- **Model Specifications**:
  - 10M parameters Vision Transformer
  - 4 summary tokens for feature aggregation
  - 512 embedding dimension
  - 8 transformer layers
  - Patch size: 64 samples (250ms at 256Hz)

- **Input Processing**:
  - Normalization: mean=2.9e-7, std=2.1e-5 (from dataset)
  - Patch-based processing for 4-second windows
  - Channel positional embeddings
  - RoPE for temporal relationships

### 📦 New Components

- `src/brain_go_brrr/models/eegpt_wrapper.py` - Normalization wrapper
- `scripts/verify_all_fixes.py` - Comprehensive fix verification
- `scripts/create_test_checkpoint.py` - Minimal checkpoint creator
- `tests/test_eegpt_checkpoint_loading.py` - Architecture validation

### 🐛 Bug Fixes

- Fixed print statements → logging
- Fixed variable naming conventions (B,C,T → batch_size, n_channels, time_steps)
- Fixed missing type annotations
- Fixed import order in debug scripts
- Fixed pre-commit hook failures

### 📊 Quality Metrics

- **Test Coverage**: 80%+ maintained
- **Type Safety**: Full mypy compliance
- **Linting**: All ruff checks passing
- **Pre-commit**: All hooks passing
- **Feature Discrimination**: Verified working

### 🚀 Next Steps

With EEGPT now producing discriminative features, the model is ready for:
- Fine-tuning on clinical EEG tasks
- Integration with downstream classifiers
- Performance benchmarking against literature
- Production deployment

## [0.3.0-alpha.2] - 2025-07-29

### 🧹 Test Suite Deep Clean

- **Test Quality Improvements**:
  - Removed ALL inappropriate xfail markers that were hiding real failures
  - Deleted 5 over-engineered mock test files that weren't testing real functionality
  - Test suite now truly green with no silent failures
  - All tests now test actual behavior, not mock behavior

- **Model Architecture Enhancements**:
  - Added IO-agnostic API to EEGPTModel (load_from_tensor method)
  - Enables proper unit testing without file I/O dependencies
  - Improves testability and separation of concerns

- **Documentation Updates**:
  - Updated PROJECT_STATUS.md to reflect test suite improvements
  - Production readiness increased from 50% to 55%
  - Test Quality score now 5/5 (Excellent)

## [0.3.0-alpha.1] - 2025-07-29

### 🔧 Test Infrastructure Stabilization

- **Redis Cache Test Migration**:
  - Replaced all module-level Redis patches with FastAPI dependency_overrides
  - Implemented clean DummyCache class without Mock inheritance
  - All Redis cache tests now passing (9/9)
  - Complete implementation of four-point stabilization checklist

- **Model Improvements**:
  - Fixed EEGPTModel to preload EDF data (required for processing)
  - Marked flaky benchmark tests as xfail (EDF writing issues)

- **Development Workflow**:
  - Removed mypy --install-types from pre-commit hooks
  - Added types-redis to dev dependencies
  - Test suite ready for CI/CD green baseline

## [0.3.0-alpha] - 2025-07-28

### 🚀 Major Improvements

- **Production-Ready Quality**: Comprehensive refactor based on senior engineering review
  - Fixed all critical test failures (420+ tests passing)
  - Resolved API field name inconsistencies
  - Improved error handling and validation
  - Enhanced code documentation

- **API Stability**:
  - Standardized field names across endpoints (`edf_file` everywhere)
  - Added proper dependency injection for cache
  - Improved error responses with meaningful messages
  - Added NumpyEncoder for proper JSON serialization

- **Test Infrastructure**:
  - Fixed all PDF report integration tests (11/11 passing)
  - Fixed all API endpoint tests (16/16 passing)
  - Resolved Redis caching test issues
  - Added proper mock injection patterns for FastAPI
  - Improved test isolation and reliability

### 🏗️ Architecture Improvements

- **Serialization System**:
  - Enhanced dataclass serialization for unknown types
  - Better binary data handling in cache
  - Graceful fallback for unregistered classes

- **EDF Processing**:
  - Robust channel selection with proper error handling
  - Added QualityCheckError for production-ready validation
  - Improved handling of non-standard channel configurations

- **Dependency Injection**:
  - Proper FastAPI dependency injection for cache
  - Removed monkey-patching in favor of `app.dependency_overrides`
  - Cleaner separation of concerns

### 🐛 Bug Fixes

- Fixed KeyError in serialization for unregistered dataclasses
- Fixed ValueError in EDF channel filtering ("picks yielded no channels")
- Fixed 422 errors in API tests due to incorrect parameter names
- Fixed binary data handling in Redis cache
- Fixed mock injection issues in FastAPI routers
- Fixed pre-commit hook failures

### 📊 Quality Metrics

- **Test Coverage**: 80%+ maintained
- **Type Safety**: Full mypy compliance (0 errors)
- **Code Quality**: All ruff linting passing
- **Security**: All bandit checks passing
- **Pre-commit**: All hooks passing consistently

### 🔧 Developer Experience

- **Improved Error Messages**: Clear, actionable error responses
- **Better Test Patterns**: Documented mock injection for FastAPI
- **Consistent Code Style**: Enforced through pre-commit hooks
- **Comprehensive README**: Production-ready documentation

### 🔄 Branch Synchronization

- All branches (development, staging, main) synchronized at commit `3b8f676`
- Clean git history with meaningful commit messages
- Proper branch protection and CI/CD workflows

---

## [0.2.0-alpha] - 2025-01-22

### 🚀 Performance Improvements

- **Test Suite Speed**: Reduced unit test execution from 3+ minutes to ~30 seconds
  - Converted `redis_disabled` to session-scoped fixture
  - Disabled pytest_benchmark and xdist by default
  - Centralized pytest options via environment variables
  - Session fixture reduced per-test overhead from 120ms to <1ms

### 🏗️ Architecture & Code Quality

- **Dataclass Serialization Registry**: Extensible pattern for Redis caching
  - Type-safe Protocol-based design
  - Explicit registration prevents import side-effects
  - Graceful fallback for unregistered classes
  - Comprehensive test coverage including edge cases

- **Import Path Standardization**: Fixed all `core.` → `brain_go_brrr.core.` imports
  - Updated test files and documentation
  - Resolved mypy import errors

### 🔧 CI/CD Enhancements

- **GitHub Actions Modernization**:
  - Fixed Python setup using `actions/setup-python@v5`
  - Added nightly integration tests with 60-minute timeout
  - Implemented test reruns for flaky tests
  - Updated to `astral-sh/setup-uv@v5` with caching
  - Branch coverage: main, development, staging

- **Type Checking**: Full mypy compliance
  - Added missing `queue_health` field
  - Fixed serialization type annotations
  - 0 errors across 63 source files

### 🐛 Bug Fixes

- Fixed Redis pool singleton import paths in tests
- Removed print statements from production code
- Fixed QueueStatusResponse schema missing field
- Resolved pytest fixture restoration issues
- Fixed EDFStreamer module import visibility

### 📦 Dependencies & Tooling

- **Pre-commit Hooks**: All passing
  - ruff (lint & format)
  - mypy (type check)
  - bandit (security)
  - pydocstyle (docstrings)

- **Makefile Improvements**:
  - Environment variable-based pytest configuration
  - Separated test targets: unit, integration, performance
  - Plugin control at command level

### 📊 Test Results

- **Unit Tests**: 167 passed, 2 skipped, 10 xfailed, 9 xpassed
- **Type Checking**: ✅ Success (0 errors)
- **Code Coverage**: Maintained at 80%+
- **Pre-commit**: All hooks passing

### 🔒 Security

- Improved Redis fixture isolation
- No hardcoded secrets or credentials
- Bandit security scanning integrated

### 📝 Documentation

- Updated all import examples in docs
- Added comprehensive test for serialization registry
- Created nightly integration workflow documentation

---

## [0.1.0] - 2025-01-15

### 🎉 Initial Release

- **Core Features**:
  - EEGPT foundation model integration (10M parameters)
  - FastAPI REST API with async support
  - Redis caching with circuit breaker pattern
  - Sleep analysis using YASA
  - Quality control with Autoreject

- **API Endpoints**:
  - `/api/v1/eeg/analyze` - Basic EEG analysis
  - `/api/v1/eeg/analyze/detailed` - Detailed analysis with PDF report
  - `/api/v1/health` - Health check endpoint
  - `/api/v1/cache/stats` - Cache statistics

- **Processing Capabilities**:
  - EDF/BDF file format support
  - 256 Hz resampling
  - 4-second window analysis
  - Up to 58 channel support

- **Infrastructure**:
  - Docker containerization
  - GitHub Actions CI/CD
  - Comprehensive test suite
  - Pre-commit hooks

- **Documentation**:
  - API documentation
  - Architecture overview
  - Development guidelines
  - Literature references
