# Brain-Go-Brrr 🧠⚡

**Research-Grade EEG Analysis with State-of-the-Art Deep Learning**

> 🚧 **Work in Progress** - This repository is under active development for research purposes.
> Not a medical device. Not for clinical use. External validation pending.

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-899%20passing-brightgreen)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)](https://github.com/Clarity-Digital-Twin/brain-go-brrr)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## 🧠 The Problem: Making Sense of Brain Waves

**What is EEG?** Think of it as a "microphone for your brain" - small sensors on your scalp detect the tiny electrical signals your brain cells use to communicate. These signals contain rich information about sleep, seizures, mental states, and brain health.

**The Challenge:** Raw EEG data is like trying to hear individual conversations in a packed stadium:
- **Noisy**: Eye blinks, muscle movements, and electrical interference drown out brain signals
- **Expert-Intensive**: Neurologists spend hours manually reviewing each recording  
- **Variable**: Every brain is unique - what works for one person often fails for another
- **Time-Consuming**: Manual preprocessing and analysis can take weeks per study

**Why Now?** The same transformer technology that powers large language models is starting to work for brain analysis:
- **Large datasets** of labeled EEG recordings are now available for training
- **Foundation models** like EEGPT can learn patterns from thousands of patients
- **Faster processing** compared to traditional manual review methods
- **Research-grade accuracy** approaching expert performance on specific validated tasks

**Our Mission:** Make research-grade EEG analysis accessible by combining the EEGPT foundation model with production-ready infrastructure and clinical validation pipelines.

**What This Is Not:**
- Not a medical device - not FDA approved for clinical decision-making
- Not a replacement for neurologists - designed to assist, not diagnose
- Not validated on all populations - trained on specific research datasets

## 🎯 What We Build

A production-ready Python system that transforms raw brain recordings into structured insights:

**Current Users:**
- **Research Labs**: Clean and analyze EEG data with reproducible pipelines
- **Academic Teams**: Standardized analysis pipeline for EEG studies

**Potential Applications:**
- **Sleep Research**: Automated sleep staging using YASA integration
- **BCI Development**: Feature extraction pipeline for brain-computer interfaces

**Key Capabilities:**
- **Sleep Staging** - 87% accuracy reported in literature (YASA; Vallat & Walker 2021)
- **Quality Control** - Artifact rejection and bad channel detection via Autoreject
- **Abnormality Detection** - Binary classification (training in progress, targeting 87% AUROC from EEGPT paper)
- **Fast Processing** - Target: <2 minutes for 20-minute recordings (hardware dependent)
- **Clean Architecture** - 899+ tests with 86% code coverage

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
                    EEG Input (.edf files)
                   (Any channel count)
                          │
                          ▼
                   Quality Control (QC)
                  [Autoreject + Bad Channels]
                          │
          ┌───────────────┴───────────────┐
          │                               │
    EEGPT Pipeline                  YASA Pipeline
    (Requires 19+ ch)            (Works with ANY count)
    (256Hz sampling)              (Resamples to 100Hz)
          │                               │
          ▼                               ▼
    EEGPT Features               Channel Selection
    (4×512 summary tokens →      (Picks best central)
     2,048 flattened)
          │                               │
          ▼                               ▼
    Abnormality Detection          Sleep Staging
    (Normal vs Abnormal)           (5 stages: W,N1,N2,N3,REM)
          │                               │
      [IF ABNORMAL]                       ▼
          │                        Sleep Metrics
          ▼                        (Efficiency, TST, etc.)
    Event Detection
    (TUEV: SPSW/GPED/PLED/etc)
```

**Key Design Principles:**
- **Parallel, not sequential** - EEGPT and YASA run independently
- **Flexible channel support** - YASA works with 1-256 channels
- **Research accuracy** - 87% sleep staging (YASA), targeting 87% abnormality AUROC
- **Well-tested** - Clean architecture, dependency injection, comprehensive testing

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
├── tests/               # 899+ unit, integration, and smoke tests
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
- Improve test coverage (currently 86%, target 90%)
- Add more preprocessing options
- Enhance documentation
- Create example notebooks

## 🔬 For Researchers

### Training Custom Models

We provide training scripts for TUAB (abnormality) and TUEV (events) datasets:

```bash
# Train abnormality detection
cd experiments/eegpt_linear_probe

# Build preprocessed cache first
./scripts/build_mne_cache.sh

# Train with MNE preprocessing
./scripts/launch_tuab_mne.sh  # For TUAB abnormality detection
./scripts/launch_tuev_mne.sh  # For TUEV event detection

# Monitor training
tmux attach -t tuab_training
```

See [TRAINING.md](docs/TRAINING.md) for detailed instructions.

### Running with Real Data

For tests with real TUAB/TUEV datasets, you'll need to set environment variables:

```bash
# Set data root directory
export BGB_DATA_ROOT=/path/to/your/data

# For versionless directory layouts (e.g., tuab/edf instead of tuab/v3.0.1/edf)
export BGB_TUAB_VERSION=""
export BGB_TUEV_VERSION=""

# Run integration tests with real data
uv run pytest -m "integration and data" --run-integration --run-data \
    tests/integration/test_tuab_real_data.py
```

### Pretrained Models

**EEGPT Foundation Model:**
- Download from [Figshare](https://figshare.com/s/e37df4f8a907a866df4b)
- Place in `data/models/pretrained/`
- 10M parameters, trained on 58 channels

### Datasets

Not included due to size/licensing. Obtain separately:

- **TUAB/TUEV** - [Temple University](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) (requires agreement)
- **Sleep-EDF** - [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) (free with registration)

## 🚦 Current Status & Roadmap

### ✅ Completed
- Sleep staging integration (YASA baseline, 87% accuracy)
- Quality control pipeline (Autoreject + MNE)
- REST API with Redis caching
- 899+ tests with CI/CD

### 🚧 In Progress
- TUAB abnormality detection training (targeting 87% AUROC)
- TUEV event detection (6-class: SPSW, GPED, PLED, etc.)
- MNE preprocessing pipeline optimization

### 📋 Planned
- External validation on holdout datasets
- Multi-site performance evaluation
- Clinical validation study
- Real-time streaming support

## 📊 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Sleep Staging | 87% accuracy | ✅ Using YASA baseline |
| Abnormality Detection | Target: 87% AUROC | 🚧 Training in progress |
| Event Detection (TUEV) | Target: 62% BAC | 🚧 Implementation phase |
| Test Coverage | 86% | ✅ 899+ passing tests |
| API Response Time | <100ms | ✅ With Redis caching |
| Processing Speed | <2 min/20min EEG | 🎯 Target (hardware dependent) |

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

## ⚠️ Limitations & Disclaimers

- **Research Software**: This is an active research project, not a medical device
- **No Clinical Use**: Not validated for clinical decision-making
- **Performance Variability**: Results may vary across different EEG systems and populations
- **Training Data**: Models trained on specific datasets; generalization to other data pending validation
- **External Validation**: Independent validation studies not yet conducted

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

<details>
<summary>MNE-Python (click to expand)</summary>

```bibtex
@article{gramfort2013mne,
  title={MEG and EEG data analysis with MNE-Python},
  author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis A and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and H{\"a}m{\"a}l{\"a}inen, Matti},
  journal={Frontiers in Neuroscience},
  volume={7},
  pages={267},
  year={2013},
  doi={10.3389/fnins.2013.00267}
}
```
</details>

## 🔧 Environment Variables

The following environment variables can be used to configure data paths:

- `BGB_DATA_ROOT` - Root data directory (default: "data")
- `BGB_SLEEP_EDF_VERSION` - Sleep-EDF dataset version (default: "sleep-edf-database-expanded-1.0.0")
- `BGB_SLEEP_EDF_DIR` - Override entire Sleep-EDF root directory
- `BGB_SLEEP_EDF_FILE` - Specific PSG file to use (for testing)

Example usage:
```bash
# Use custom data location
export BGB_DATA_ROOT=/mnt/data/eeg
uv run python scripts/testing/test_sleep_analysis.py

# Use specific Sleep-EDF file
export BGB_SLEEP_EDF_FILE=/path/to/specific/file.edf
uv run pytest tests/unit/domain/sleep -v
```

## 🤝 Support & Community

- **Issues:** [GitHub Issues](https://github.com/Clarity-Digital-Twin/brain-go-brrr/issues)
- **Discussions:** Coming soon
- **Email:** See maintainer profiles

---

*Built with ❤️ by the Clarity Digital Twin team*

*For AI assistants: See [CLAUDE.md](CLAUDE.md) for context and guidelines*
