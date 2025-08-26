# Brain-Go-Brrr 🧠⚡

**Production-Ready EEG Analysis System with EEGPT Foundation Model**

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-790%20passing-brightgreen)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions)
[![Coverage](https://img.shields.io/badge/coverage-66%25-yellow)](https://github.com/Clarity-Digital-Twin/brain-go-brrr)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

Brain-Go-Brrr provides automated EEG analysis using frozen EEGPT features with specialized task heads for sleep staging, quality control, and abnormality detection.

### 🎉 New: MNE+Autoreject Integration
- **Improved TUAB accuracy**: 56% → 87% AUROC (expected)
- **Clinical-grade preprocessing**: Automatic artifact rejection with validated parameters
- **Smart channel handling**: Automatic T3→T7, T4→T8, T5→P7, T6→P8 mapping
- **Adaptive filtering**: Muscle artifact detection adapted to sampling rate

## 🚀 Quick Start

```bash
# Install (takes 30 seconds)
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run API server
uv run uvicorn brain_go_brrr.api.main:app --reload

# Test the system
curl http://localhost:8000/api/v1/health
```

**Next Steps:**
- 📖 [Full Setup Guide](docs/QUICK_START.md) - Detailed installation and configuration
- 🏗️ [Architecture Overview](docs/ARCHITECTURE.md) - Understand the system design
- 🔌 [API Documentation](docs/API.md) - REST endpoints and examples

## 🧬 How It Works

We use **parallel processing pipelines** optimized for different analysis tasks:

```
        Raw EEG (.edf files)
               │
        Quality Control
         (Autoreject QC)
               │
    ┌──────────┴──────────┐
    │                     │
EEGPT Pipeline      YASA Pipeline
(19+ channels)      (Any channels)
    │                     │
Abnormality          Sleep Staging
Detection            (W,N1,N2,N3,REM)
    │                     │
Event Detection      Sleep Metrics
(if abnormal)        (efficiency, TST)
```

**Key Design Principles:**
- **Parallel, not sequential** - EEGPT and YASA run independently
- **Flexible channel support** - YASA works with 1-256 channels
- **Clinical accuracy** - 87% sleep staging, targeting 87% abnormality AUROC
- **Production-ready** - Clean architecture, dependency injection, comprehensive testing

## 💻 For Developers

### Project Structure
```
brain-go-brrr/
├── src/brain_go_brrr/     # Main package
│   ├── domain/            # Business logic (pure, no dependencies)
│   ├── application/       # Use cases and orchestration
│   ├── infra/            # External adapters (EEGPT, YASA, etc.)
│   └── api/              # REST API endpoints
├── experiments/          # Training scripts and research
├── tests/               # 790+ unit, integration, and smoke tests
└── docs/               # Comprehensive documentation
```

### Development Workflow
```bash
# Run tests with coverage
make test

# Check code quality
make lint typecheck

# Watch tests during development
make test-watch

# Full CI/CD check before pushing
make check-all
```

### Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation:

1. **Fork & Clone** the repository
2. **Read** [ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the design
3. **Follow** [TESTING.md](docs/TESTING.md) for test guidelines
4. **Create** a pull request with clear description

**Good First Issues:**
- Improve test coverage (currently 66%, target 70%)
- Add more preprocessing options
- Enhance documentation
- Create example notebooks

## 🔬 For Researchers

### Training Custom Models

We provide training scripts for TUAB (abnormality) and TUEV (events) datasets:

```bash
# Train abnormality detection
cd experiments/eegpt_linear_probe
./scripts/launch_tuab.sh

# Monitor training
tmux attach -t tuab_training
```

See [TRAINING.md](docs/TRAINING.md) for detailed instructions.

### Pretrained Models

**EEGPT Foundation Model:**
- Download from [Figshare](https://figshare.com/s/e37df4f8a907a866df4b)
- Place in `data/models/pretrained/`
- 10M parameters, trained on 58 channels

### Datasets

Not included due to size/licensing. Obtain separately:

- **TUAB/TUEV** - [Temple University](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) (requires agreement)
- **Sleep-EDF** - [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) (free with registration)

## 📊 Performance Metrics

| Metric | Value | Details |
|--------|-------|---------|
| Sleep Staging Accuracy | 87% | 5-stage classification (YASA) |
| Test Coverage | 66% | 790+ passing tests |
| API Response Time | <100ms | With Redis caching |
| Processing Speed | <2 min | For 20-minute recording |
| Supported Channels | 1-256 | YASA: any, EEGPT: 19+ |

## 🛠️ System Requirements

- **Python:** 3.11 or 3.12
- **RAM:** 16GB minimum
- **GPU:** Optional (speeds up training)
- **OS:** Linux, macOS, Windows (WSL2)

## 📚 Documentation

| Guide | Description |
|-------|------------|
| [QUICK_START.md](docs/QUICK_START.md) | Get running in 5 minutes |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and patterns |
| [API.md](docs/API.md) | REST endpoint reference |
| [TRAINING.md](docs/TRAINING.md) | Model training guide |
| [TESTING.md](docs/TESTING.md) | Test philosophy and guidelines |

## 📄 License & Citations

**License:** Apache 2.0 - See [LICENSE](LICENSE)

**If you use this software in research, please cite:**

<details>
<summary>EEGPT Model (click to expand)</summary>

```bibtex
@inproceedings{wang2024eegpt,
  title={EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals},
  author={Wang, Guangyu and He, Yuhong and Ma, Lin and Liu, Wenchao and Xu, Cong and Li, Haifeng},
  booktitle={38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}
```
</details>

<details>
<summary>YASA Sleep Staging (click to expand)</summary>

```bibtex
@article{vallat2021yasa,
  title={YASA: Yet Another Spindle Algorithm},
  author={Vallat, Raphael and Walker, Matthew P},
  journal={bioRxiv},
  year={2021},
  doi={10.1101/2021.05.28.446165}
}
```
</details>

<details>
<summary>Autoreject (click to expand)</summary>

```bibtex
@article{jas2017autoreject,
  title={Autoreject: Automated artifact rejection for MEG and EEG data},
  author={Jas, Mainak and Engemann, Denis A and Bekhti, Yousra and Raimondo, Federico and Gramfort, Alexandre},
  journal={NeuroImage},
  volume={159},
  pages={417--429},
  year={2017}
}
```
</details>

## 🤝 Support & Community

- **Issues:** [GitHub Issues](https://github.com/Clarity-Digital-Twin/brain-go-brrr/issues)
- **Discussions:** Coming soon
- **Email:** See maintainer profiles

---

*Built with ❤️ by the Clarity Digital Twin team*

*For AI assistants: See [CLAUDE.md](CLAUDE.md) for context and guidelines*