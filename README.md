# Brain-Go-Brrr ⚡

**Production-Ready EEG Analysis with EEGPT Foundation Model**

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> 📢 Note: Active seizure detection development has moved to SeizureTransformer (https://github.com/Clarity-Digital-Twin/SeizureTransformer). This repository maintains core EEG processing infrastructure.

## What This Is

A Python wrapper around the EEGPT foundation model (10M parameters) for EEG analysis, providing:
- EEGPT Features: 4×512 token embeddings (2048-d flattened for training; 512-d pooled in API by default)
- Sleep Analysis: 87% accuracy using YASA (off-the-shelf)
- Quality Control: Autoreject for bad channel/artifact detection
- Abnormality Detection: Linear probe on TUAB dataset (training)
- REST API: FastAPI with Redis caching

## Quick Start

### Requirements
- Python 3.11-3.12 (3.13 not supported - scipy issues)
- 8GB+ RAM recommended (16GB for training)
- GPU optional (speeds up training)
- Ubuntu/WSL2 recommended

### Install
```bash
# Clone and install
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run API
uv run uvicorn brain_go_brrr.api.main:app --reload

# Test
curl http://localhost:8000/api/v1/health
```

## Core Capabilities

### ✅ Working Now
- Sleep Staging: 5-stage classification (W, N1, N2, N3, REM) via YASA
- Feature Extraction: EEGPT embeddings (4×512 tokens → 2048 dims)
- Quality Control: Bad channel detection, artifact rejection
- API Server: REST endpoints with async processing

### 📊 Performance
| Component | Metric | Status |
|-----------|--------|--------|
| Sleep Staging | 87% accuracy (YASA paper) | ✅ Working |
| QC (Autoreject) | 87% expert agreement (paper) | ✅ API only* |
| TUAB Abnormality | 0.869 AUROC (EEGPT paper) | 🟡 Training |
| API Response | <100ms cached | ✅ Working |

*QC integrated in API flows; training scripts bypass QC for speed

### ⚠️ Archived
- TUEV Events: Implementation correct but 22% BAC (vs 62% paper claim) - dataset has fatal class imbalance

## Architecture

```
EEG Input → Quality Control → Parallel Processing:
                              ├── EEGPT (19+ channels, 256Hz) → Features → Abnormality Detection
                              └── YASA (any channels, 100Hz) → Sleep Stages → Metrics
```

Key points:
- EEGPT requires 19+ channels at 256Hz
- YASA works with any channel count
- Processing is parallel, not sequential

## Project Structure

```
brain-go-brrr/
├── src/brain_go_brrr/
│   ├── domain/          # Pure business logic
│   ├── application/     # Use cases
│   ├── infra/          # External adapters (EEGPT, YASA)
│   └── api/            # REST endpoints
├── experiments/        # Training scripts
├── tests/             # Unit & integration tests
└── docs/              # Documentation
```

## Data & Models

### Required Downloads
1. EEGPT Model: Download from Figshare (https://figshare.com/s/e37df4f8a907a866df4b)
   - Place in: `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`

2. Datasets (obtain separately):
   - TUAB: Temple University (https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) (requires agreement)
   - Sleep-EDF: PhysioNet (https://physionet.org/content/sleep-edfx/1.0.0/) (free registration)

### Configuration
```bash
# Required
export BGB_DATA_ROOT=/path/to/data

# Optional
export BGB_TUAB_VERSION=""         # Versionless layout
export BGB_CACHE_DIR=$BGB_DATA_ROOT/cache  # Cache location
export BGB_SLEEP_EDF_FILE=/path/to/file.edf  # Specific test file
export EEGPT_CKPT_PATH=$BGB_DATA_ROOT/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt  # Override checkpoint
```

## Development

### Core Commands
```bash
# Run tests
make test

# Watch tests (TDD mode)
make test-watch

# Code quality
make lint        # Ruff linting
make typecheck   # Mypy checking
make format      # Auto-format code

# Full CI check (MUST PASS before push)
make check-all
```

### Training Models
```bash
# Build cache and train TUAB
cd experiments/eegpt_linear_probe
./scripts/build_mne_cache.sh
./scripts/launch_tuab_mne.sh
```

### Integration Tests
```bash
# With real data
uv run pytest -m "integration and data" --run-integration --run-data

# Specific dataset
export BGB_SLEEP_EDF_FILE=/path/to/file.edf
uv run pytest tests/unit/domain/sleep -v
```

## Key Technical Details

- Sampling: 256Hz for EEGPT, 100Hz for YASA
- Windows: 4 seconds for EEGPT (some legacy 8s code exists)
- Channels: Modern naming (T7/T8/P7/P8), not legacy (T3/T4/T5/T6)
- Features: EEGPT outputs 4×512 summary tokens, flattened to 2048 dims
- Architecture: Clean separation - domain layer has zero infrastructure dependencies

## Related Work

- SeizureTransformer: Wu 2025 CNN+Transformer implementation moved to separate repo (https://github.com/Clarity-Digital-Twin/SeizureTransformer)
- Clinical Evaluation: NEDC scoring and metrics in SeizureTransformer repo

## Documentation

| Guide | Purpose |
|-------|---------|
| docs/ARCHITECTURE.md | System design patterns |
| docs/API.md | REST endpoint reference |
| docs/TRAINING.md | Model training guide |
| docs/QUICK_START.md | Detailed setup |
| docs/CI_CD_SETUP.md | GitHub Actions pipeline |
| ROADMAP.md | Future plans |

## Limitations

- Research software, not a medical device
- Not FDA approved for clinical use
- Models trained on specific datasets - generalization pending validation
- Performance varies across EEG systems and populations

## License & Citation

Apache 2.0 License. See LICENSE.

If using in research, please cite:
- EEGPT: Wang et al. (2024) NeurIPS
- YASA: Vallat & Walker (2021) bioRxiv
- MNE-Python: Gramfort et al. (2013) Frontiers
- Datasets: See documentation for specific citations

---

For AI assistants: See CLAUDE.md for development context

