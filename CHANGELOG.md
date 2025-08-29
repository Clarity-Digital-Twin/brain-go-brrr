# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-08-29

### Added
- PHI path masking for HIPAA compliance in all logging outputs
- Architecture drift guards in CI to prevent parallel implementations
- AST-based torch.load safety checker
- Comprehensive CI/CD documentation

### Changed
- **MAJOR**: Unified architecture - eliminated parallel implementations between src/ and experiments/
- CI/CD now runs tools directly (removed pre-commit from CI)
- Normalization centralized in EEGPT wrapper (SSOT)
- Python version constraint to >=3.11,<3.13

### Fixed
- TUAB channel mapping (19 channels, excludes Fz)
- TUEV channel mapping (20 channels, includes Fz, excludes Fpz)
- UTC import compatibility across Python 3.11.x versions
- MyPy decorator issues with FastAPI/Typer
- 4-second window extraction for EEGPT
- Channel naming (T3→T7, T4→T8, T5→P7, T6→P8)

### Security
- Added weights_only=True to all torch.load() calls
- Implemented PHI protection with path masking

### Removed
- ~34,000 lines of duplicate/dead code
- TCP/bipolar code from TUEV dataset
- Pre-commit from CI workflow (local only now)

## [1.0.0] - 2025-08-15

### Added
- Initial release with core functionality
- Quality Control with AutoReject integration
- Abnormality Detection with EEGPT linear probe
- Sleep Analysis with YASA adapter
- Event Detection framework
- FastAPI REST API
- CLI interface
- PDF/Markdown report generation
- 790+ passing tests

### Fixed
- API smoke tests using TestClient
- Quality grade output format
- RejectLog iteration compatibility
- Version tests with importlib.metadata

[Unreleased]: https://github.com/Clarity-Digital-Twin/brain-go-brrr/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Clarity-Digital-Twin/brain-go-brrr/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Clarity-Digital-Twin/brain-go-brrr/releases/tag/v1.0.0