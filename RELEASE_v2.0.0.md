# Release v2.0.0 - SeizureTransformer Architecture & Repository Transition

## 🚀 Major Announcement

We are transitioning our active development to specialized repositories for better modularity:

- **Clinical Evaluation & NEDC Scoring**: Active development has moved to [SeizureTransformer](https://github.com/Clarity-Digital-Twin/SeizureTransformer)
- **Core EEG Processing**: Remains in this repository

## What's New in v2.0.0

### ✅ SeizureTransformer Implementation
- **Wu 2025 Architecture**: Full CNN+Transformer implementation (`seizure_transformer_wu2025.py`)
- **41M Parameters**: Production-ready model matching paper specifications
- **TUSZ Support**: Complete dataset pipeline for temporal seizure detection
- **CI/CD Compliant**: All type checking and linting issues resolved

### 📊 TUEV Event Classification Status
- **Implementation**: ✅ Verified correct, matches EEGPT reference exactly
- **Performance**: 22% BAC (vs 62% paper claim)
- **Status**: ARCHIVED - Dataset has fatal class imbalance (22 samples for minority class)
- **Recommendation**: Use TUAB or other balanced datasets

### 🔄 Repository Status

This repository (`brain-go-brrr`) will continue to maintain:
- EEGPT foundation model integration
- YASA sleep analysis (87% accuracy)
- Autoreject QC pipeline
- Core API infrastructure

For active development on seizure detection and clinical evaluation:
- **Please visit**: [github.com/Clarity-Digital-Twin/SeizureTransformer](https://github.com/Clarity-Digital-Twin/SeizureTransformer)
- **Features there**: NEDC scoring, clinical metrics, operating point optimization

## Breaking Changes
- Removed `seizure_transformer_toy_deprecated.py` (duplicate classes)
- SeizureTransformer wrapper now uses Wu 2025 architecture by default

## Migration Guide

If you're working on seizure detection:
```bash
# Clone the new repository
git clone https://github.com/Clarity-Digital-Twin/SeizureTransformer
cd SeizureTransformer
uv pip install -e .
```

If you're using core EEG features:
```bash
# Continue using this repository
uv pip install brain-go-brrr==2.0.0
```

## Contributors
Thanks to the team for the SeizureTransformer implementation and architecture improvements.

## Full Changelog
- Wu 2025 SeizureTransformer architecture implementation
- Complete TUSZ dataset support
- TUEV implementation verified but archived (poor performance)
- All mypy type checking issues resolved
- CI/CD pipeline fully passing

**Full Changelog**: v1.2.0...v2.0.0
