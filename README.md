# Brain-Go-Brrr 🧠⚡

**Research-Grade EEG Analysis Platform Powered by EEGPT Foundation Model**

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)

> 🏆 **Built for GitHub's #ForTheLoveOfCode 2025 Hackathon** - Category: Agents of Change
> Leveraging AI to democratize EEG analysis and accelerate neuroscience research.

## 🎯 Overview

Brain-Go-Brrr is an open-source EEG analysis platform that integrates the EEGPT foundation model [1] with established neuroscience tools to provide automated quality control, sleep staging, and abnormality detection capabilities. The project demonstrates how modern transformer architectures can be applied to biosignal processing while maintaining scientific rigor.

### Current Capabilities

- **🔍 Quality Control**: Automated bad channel detection using Autoreject [2] with >87% expert agreement
- **😴 Sleep Analysis**: 5-stage classification achieving 87% accuracy using YASA [3]
- **🧠 Feature Extraction**: EEGPT-based universal EEG representations (512-dim embeddings)
- **⚡ Fast Processing**: Analyze 20-minute recordings in <2 minutes
- **🔧 Modular Architecture**: Clean DDD/hexagonal architecture for maintainability

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up environment and install dependencies
uv sync --all-extras

# Run tests to verify installation
uv run pytest tests/smoke -q

# Start API server
uv run python -m brain_go_brrr.api
```

## 📊 What We've Built

### 1. EEGPT Integration ✅

Successfully integrated the 10M parameter EEGPT model for feature extraction:
- 4-second window processing at 256Hz
- Support for variable channel configurations (up to 58 channels)
- Compatibility layer for both old and new EEGPT APIs
- Summary token extraction without duplication

### 2. Sleep Staging Module ✅

Production-ready sleep analysis using YASA:
- **Accuracy**: 87% on Sleep-EDF dataset
- **Stages**: W, N1, N2, N3, REM classification
- **Metrics**: Sleep efficiency, REM%, N3%, WASO, total sleep time
- **Channel Aliasing**: Automatic handling of Sleep-EDF's Fpz-Cz channels

### 3. Quality Control System ✅

Automated artifact detection using Autoreject:
- Bad channel identification with visualization
- Epoch-level artifact rejection
- Compatible with clinical montages
- Requires 2+ minutes of data for cross-validation

### 4. API Infrastructure ✅

FastAPI-based REST API with:
- File upload endpoints for EDF/BDF files
- Redis caching with circuit breaker pattern
- Async processing support
- Comprehensive error handling

## 🏗️ Architecture

```
brain-go-brrr/
├── src/brain_go_brrr/         # Main package
│   ├── api/                   # FastAPI endpoints
│   ├── application/           # Use cases & services
│   ├── domain/               # Business logic & entities
│   │   ├── quality/         # QC controller
│   │   └── sleep/           # Sleep analysis
│   ├── infra/                # External integrations
│   │   ├── ml_models/       # EEGPT & model wrappers
│   │   └── external/        # YASA, Autoreject adapters
│   └── services/             # Service orchestration
├── experiments/              # Training scripts
│   └── eegpt_linear_probe/  # TUAB abnormality detection
├── tests/                    # Comprehensive test suite
│   ├── unit/                # 800+ unit tests
│   ├── integration/         # End-to-end tests
│   └── benchmarks/          # Performance tests
└── data/                    # Data & models (gitignored)
    └── models/
        └── eegpt/pretrained/  # EEGPT checkpoint
```

## 🔬 Technical Implementation

### Foundation Model

We use EEGPT [1], a 10M parameter transformer pretrained on diverse EEG datasets:
- **Architecture**: Vision Transformer with masked autoencoding
- **Input**: 256Hz, 4-second windows, up to 58 channels
- **Output**: 4 summary tokens × 512 dimensions
- **Pretraining**: PhysioNet, TUAB, and other public datasets

### Data Processing Pipeline

```python
# Standard pipeline
Raw EEG → Resampling (256Hz) → Filtering (0.5-50Hz) →
Window Extraction → EEGPT Features → Task-Specific Head
```

### Key Design Decisions

1. **Clean Architecture**: Domain-driven design with clear boundaries
2. **Compatibility First**: Support for multiple EEG formats and montages
3. **Scientific Validation**: All accuracy claims backed by test data
4. **Modular Components**: Easy to swap implementations

## 📈 Performance Metrics

### Achieved Performance

| Task | Method | Accuracy | Notes |
|------|--------|----------|-------|
| Sleep Staging | YASA | 87% | 5-stage classification on Sleep-EDF |
| Bad Channels | Autoreject | 87% | Expert agreement rate |
| Feature Quality | EEGPT | - | 512-dim embeddings per window |
| API Latency | FastAPI | <100ms | Health check endpoint |
| Processing Speed | Pipeline | <2min | 20-minute recording |

### In Development

| Task | Target | Status |
|------|--------|--------|
| Abnormality Detection | 86.9% AUROC | Training linear probe on TUAB |
| Event Detection | - | Architecture designed |
| Real-time Streaming | 30s latency | CLI prototype working |

## 🛠️ Development

### Prerequisites

- Python 3.11 or 3.12 (3.13 not yet supported due to scipy)
- 8GB RAM minimum
- CUDA GPU (optional, for faster inference)
- Redis (optional, for caching)

### Running Tests

```bash
# Quick smoke tests
uv run pytest tests/smoke -q

# Unit tests
uv run pytest tests/unit -q

# Integration tests (requires data)
uv run pytest tests/integration --run-integration

# Full test suite with coverage
make test-all-cov
```

### Code Quality

```bash
# Format code
make format

# Lint checks
make lint

# Type checking
make typecheck

# All checks
make check-all
```

## 📚 Scientific References

[1] Wang, G., He, Y., Ma, L., Liu, W., Xu, C., & Li, H. (2024). **EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals**. *38th Conference on Neural Information Processing Systems (NeurIPS 2024)*. [GitHub](https://github.com/BINE022/EEGPT)

[2] Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). **Autoreject: Automated artifact rejection for MEG and EEG data**. *NeuroImage*, 159, 417-429.

[3] Vallat, R., & Walker, M. P. (2021). **An open-source, high-performance tool for automated sleep staging**. *eLife*, 10:e70092. [GitHub](https://github.com/raphaelvallat/yasa)

## 🚧 Known Limitations

### Technical Debt

1. **Sleep-EDF Resampling**: Data is at 100Hz but EEGPT requires 256Hz
2. **TestClient Limitations**: FastAPI TestClient doesn't properly handle dependency overrides with file uploads
3. **AutoReject Memory**: Requires 100+ epochs (2+ minutes of data) for cross-validation
4. **Channel Mapping**: TUAB uses old nomenclature (T3→T7, T4→T8, etc.)

See [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) for detailed documentation.

### Current Constraints

- EEGPT model requires exact 4-second windows at 256Hz
- Limited to 10-20 electrode montages for clinical accuracy
- No real-time streaming for clinical use (research only)
- Binary abnormality detection only (not diagnostic)

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

1. **Model Training**: Fine-tuning EEGPT on specific tasks
2. **Data Loaders**: Support for more EEG formats
3. **Visualization**: Better plotting and reporting tools
4. **Documentation**: Tutorials and examples
5. **Testing**: More integration tests with real data

Please follow our coding standards:
- Type hints for all functions
- Docstrings with Args/Returns
- Unit tests for new features
- No hardcoded paths

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for **research purposes only**. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for clinical EEG interpretation.

## 🙏 Acknowledgments

- EEGPT team at Harbin Institute of Technology for the foundation model
- MNE-Python community for signal processing tools
- Raphael Vallat for YASA sleep staging
- Alexandre Gramfort and team for Autoreject
- All contributors to the open EEG datasets used in development

---

*For technical documentation, see [docs/](docs/). For API reference, see [docs/03-api/](docs/03-api/).*
