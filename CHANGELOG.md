# Changelog

All notable changes to Brain-Go-Brrr will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
