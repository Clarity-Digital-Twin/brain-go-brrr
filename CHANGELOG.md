# Changelog

All notable changes to Brain-Go-Brrr will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Pipeline Error Handling**: Added `traceback` field to error results in parallel pipeline (backward compatible, additive field)
- **Test Infrastructure**: Added `requires_network` fixture for gating network-dependent tests (use `BGB_ALLOW_NET=1` to enable)
- **Pre-commit Hooks**: Added torch.load security check and bare except prevention

### Fixed
- **Security**: All `torch.load` calls now use `weights_only=True` with backward compatibility guard for older PyTorch versions
- **Test Stability**: Fixed FakeMNERaw data aliasing, TUAB test brittleness, and time formatting test reliability

### Changed
- **Testing**: Reduced default pytest timeout to 120s with faulthandler for faster failure detection
- **Coverage**: Excluded `experiments/` directory from coverage calculations

## [0.6.0] - 2025-08-05

### 🎯 Dual Pipeline Architecture & 4-Second Window Training

This release implements the complete dual pipeline architecture for autonomous EEG analysis and fixes critical window size issues for EEGPT training. The system now supports parallel processing of abnormality detection and sleep staging with hierarchical epileptiform detection.

### ✨ Major Features

#### **Critical Discovery: 4-Second Windows**
- **BREAKING**: EEGPT was pretrained on 4-second windows, not 8-second
- Rewrote entire training pipeline for correct window size
- Target AUROC: 0.869 (paper performance) vs 0.81 with 8s windows
- Complete pure PyTorch implementation avoiding Lightning bugs

#### **Dual Pipeline Architecture**
- **Hierarchical Pipeline**: EEG → Normal/Abnormal → IED Detection
- **Parallel Pipeline**: Simultaneous YASA sleep staging
- Full async/await support for concurrent processing
- Automatic triage system (URGENT/EXPEDITE/ROUTINE/NORMAL)

#### **YASA Sleep Staging Integration**
- Complete YASA adapter implementation with consensus models
- 5-stage classification (W, N1, N2, N3, REM)
- Hypnogram generation and sleep metrics
- Real-time processing with confidence scores

#### **TDD Implementation**
- 454 passing tests with comprehensive coverage
- Full integration tests for both pipelines
- Mock-free testing with real components
- Performance benchmarks for concurrent processing

### 🚀 Infrastructure Improvements

- **PyTorch Lightning Bug Workaround**: Pure PyTorch training script
- **Professional Documentation**: Complete overhaul of all docs
- **CI/CD Fixes**: Resolved trailing whitespace issues
- **Branch Synchronization**: All branches aligned (dev/staging/main)
- **tmux Session Management**: Persistent training sessions
- **Cache Infrastructure**: 4s and 8s window caches

### 📊 Current Training Status

- **Active**: 4-second window training (paper-aligned)
- **Session**: `tmux attach -t eegpt_4s_final`
- **Target**: AUROC ≥ 0.869
- **Production Readiness**: 75%

### 🐛 Bug Fixes

- Fixed PyTorch Lightning 2.5.2 hanging with large cached datasets
- Resolved channel mapping issues (T3→T7, T4→T8, T5→P7, T6→P8)
- Fixed cache index path requirements
- Corrected environment variable resolution
- Fixed all dimension mismatches in model

### 📚 Documentation

- **TRAINING_STATUS.md**: Live training updates
- **ISSUES_AND_FIXES.md**: Complete problem/solution guide
- **SETUP_COOKBOOK.md**: Detailed setup instructions
- **INDEX.md**: Clean directory structure guide
- **PROJECT_STATUS.md**: Updated to 75% production ready

### 🧪 Testing

- 454 unit tests passing
- 136 integration tests (marked for nightly runs)
- Full pipeline E2E tests
- YASA integration validated
- Hierarchical pipeline tested

### 🔬 Technical Specifications

#### Pipeline Architecture
```
Input EEG → QC Check → EEGPT Features → Dual Pipeline:
                                        ├── Abnormality Detection
                                        │   └── (if abnormal) → IED Detection
                                        └── Sleep Staging (parallel)
```

#### Training Configuration
- Window duration: 4.0 seconds (1024 samples @ 256Hz)
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 200
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau

### 📈 Performance Metrics

| Component | Status | Performance |
|-----------|--------|-------------|
| Abnormality Detection | Training | Target: 0.869 AUROC |
| Sleep Staging | ✅ Implemented | 87.46% accuracy (YASA) |
| IED Detection | 🟡 Mock Ready | Awaiting training |
| QC Pipeline | ✅ Complete | >95% accuracy |
| API Endpoints | ✅ Ready | <100ms response |

### 🚀 Usage

#### Monitor Current Training
```bash
tmux attach -t eegpt_4s_final
tail -f output/tuab_4s_paper_aligned_*/training.log
```

#### Run Dual Pipeline
```python
from brain_go_brrr.services.hierarchical_pipeline import HierarchicalPipeline

pipeline = HierarchicalPipeline()
result = await pipeline.analyze(eeg_data)
print(f"Abnormality: {result.abnormality_score}")
print(f"Sleep Stage: {result.sleep_stage}")
```

### 📥 Installation

```bash
pip install brain-go-brrr==0.6.0
```

Or with uv:
```bash
uv pip install brain-go-brrr==0.6.0
```

### 🔄 Migration from 0.5.0

1. **Update window size**: Change from 8s to 4s windows
2. **Use new training script**: `train_paper_aligned.py` instead of Lightning
3. **Update configs**: Use `tuab_4s_paper_aligned.yaml`
4. **Clear caches**: Rebuild with 4s windows

### 📋 Next Steps

- [ ] Complete 4s window training (3-4 hours remaining)
- [ ] Validate AUROC ≥ 0.869
- [ ] Implement real IED detection module
- [ ] Add clinical validation pipeline
- [ ] Deploy to production infrastructure

### 🙏 Acknowledgments

- EEGPT team for foundation model insights
- YASA developers for sleep staging algorithms
- GitHub Copilot for development assistance
- Clinical partners for domain expertise

Full Changelog: v0.5.0...v0.6.0

---

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

### 🎯 EEGPT Model Fixed - Input Normalization Solution

After extensive debugging, the EEGPT model integration is now fully functional with proper feature discrimination.

### ✨ Added

- **EEGPTWrapper**: New wrapper class with automatic input normalization
- **Normalization statistics**: Computed and saved from Sleep-EDF dataset
- **Custom Attention module**: Exact implementation matching EEGPT paper
- **Rotary Position Embeddings**: Enabled for temporal encoding
- **Minimal test checkpoint**: 96MB test model for CI/CD (vs 1GB full)

### 🐛 Fixed

- **Critical normalization bug**: Raw EEG (~50μV) was 115x smaller than model bias
- **Channel embeddings**: Fixed to use 62 channels (0-61 indexed)
- **Transformer blocks**: Now loads all 8 blocks (was missing intermediate)
- **Feature discrimination**: Cosine similarity now ~0.486 (was 1.0)
- **Test fixtures**: Fixed scoping issues in pytest fixtures

### 📊 Performance

- Features now properly discriminative between different EEG samples
- Model outputs show appropriate variance (std ~0.015)
- All 368 tests passing with full type safety

## [0.3.0-alpha] - 2025-07-25

### 🚀 Initial Alpha Release

First functional release with core EEG analysis capabilities.

### ✨ Features

- **Quality Control Module**: Automated bad channel detection
- **Sleep Analysis**: YASA integration for 5-stage classification  
- **EEGPT Integration**: Foundation model for feature extraction
- **FastAPI REST API**: Production-ready endpoints
- **Redis Caching**: High-performance result caching

### 🧪 Testing

- 361 unit tests passing
- 63.47% code coverage
- TDD approach throughout

### 📚 Documentation

- Comprehensive README
- API documentation
- Clinical integration guides

## [0.2.0] - 2025-07-20

### ✨ Added

- FastAPI application structure
- Redis caching layer
- Basic EEGPT model loading
- Project scaffolding

## [0.1.0] - 2025-07-15

### 🎉 Initial Development

- Project initialization
- Basic package structure
- Development environment setup
- Pre-commit hooks configuration