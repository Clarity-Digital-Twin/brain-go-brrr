# Changelog

All notable changes to Brain-Go-Brrr will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### 🎯 Next Steps

- Enable branch protection rules
- Run manual nightly integration test
- Deploy v0.2.0-alpha to staging
- Monitor performance metrics

---

## [0.1.0] - 2025-01-15

- Initial release
- Basic EEGPT integration
- FastAPI endpoints for EEG analysis
- Redis caching support
- Sleep analysis with YASA
- Quality control with Autoreject
