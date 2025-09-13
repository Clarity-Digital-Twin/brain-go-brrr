# Brain-Go-Brrr ⚡

**Production-Ready EEG Analysis with EEGPT Foundation Model**

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> 📢 **Note**: Active seizure detection development has moved to [SeizureTransformer](https://github.com/Clarity-Digital-Twin/SeizureTransformer). This repository maintains core EEG processing infrastructure.

## What This Is

A Python wrapper around the EEGPT foundation model (10M parameters) for EEG analysis, providing:
- **EEGPT Features**: 2048-dimensional embeddings from transformer model
- **Sleep Analysis**: 87% accuracy using YASA integration
- **Quality Control**: Automated artifact rejection via Autoreject
- **REST API**: FastAPI with Redis caching

## Quick Start

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
- **Sleep Staging**: 5-stage classification (W, N1, N2, N3, REM) via YASA
- **Feature Extraction**: EEGPT embeddings (4×512 tokens → 2048 dims)
- **Quality Control**: Bad channel detection, artifact rejection
- **API Server**: REST endpoints with async processing

### 📊 Performance
| Component | Metric | Status |
|-----------|--------|--------|
| Sleep Staging | 87% accuracy | ✅ Production |
| QC Agreement | 87% with experts | ✅ Production |
| TUAB Abnormality | 83% AUROC | ✅ Complete |
| API Response | <100ms cached | ✅ Production |

### ⚠️ Archived
- **TUEV Events**: Implementation correct but 22% BAC (vs 62% paper claim) - dataset has fatal class imbalance

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
1. **EEGPT Model** (~40MB): [Download from Figshare](https://figshare.com/s/e37df4f8a907a866df4b)
   - Place in: `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`

2. **Datasets** (obtain separately):
   - **TUAB**: [Temple University](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) (requires agreement)
   - **Sleep-EDF**: [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) (free registration)

### Configuration
```bash
export BGB_DATA_ROOT=/path/to/data
export BGB_TUAB_VERSION=""  # For versionless layout (tuab/edf not tuab/v3.0.1/edf)
```

## Development

```bash
# Run tests
make test

# Check code quality
make lint typecheck

# Full CI check
make check-all

# Train models (if you have data)
cd experiments/eegpt_linear_probe
./scripts/build_mne_cache.sh
./scripts/launch_tuab_mne.sh
```

## Key Technical Details

- **Sampling**: 256Hz for EEGPT, 100Hz for YASA
- **Windows**: 4 seconds for TUAB abnormality detection
- **Channels**: Modern naming (T7/T8/P7/P8), not legacy (T3/T4/T5/T6)
- **Features**: EEGPT outputs 4×512 summary tokens, flattened to 2048 dims
- **Architecture**: Clean separation - domain layer has zero infrastructure dependencies

## Related Work

- **SeizureTransformer**: Wu 2025 CNN+Transformer implementation moved to [separate repo](https://github.com/Clarity-Digital-Twin/SeizureTransformer)
- **Clinical Evaluation**: NEDC scoring and metrics in SeizureTransformer repo

## Documentation

| Guide | Purpose |
|-------|---------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design patterns |
| [API.md](docs/API.md) | REST endpoint reference |
| [TRAINING.md](docs/TRAINING.md) | Model training guide |
| [QUICK_START.md](docs/QUICK_START.md) | Detailed setup |

## Limitations

- Research software, not a medical device
- Not FDA approved for clinical use
- Models trained on specific datasets - generalization pending validation
- Performance varies across EEG systems and populations

## License & Citation

Apache 2.0 License. See [LICENSE](LICENSE).

If using in research, please cite:
- EEGPT: Wang et al. (2024) NeurIPS
- YASA: Vallat & Walker (2021) bioRxiv
- MNE-Python: Gramfort et al. (2013) Frontiers
- Datasets: See documentation for specific citations

---

*For AI assistants: See [CLAUDE.md](CLAUDE.md) for development context*