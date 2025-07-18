# Brain Go Brrr 🧠⚡

A modern, production-ready digital twin brain-computer interface project focused on EEG signal processing and neural representation learning using the EEGPT transformer architecture.

## 🎯 Project Overview

Brain Go Brrr is a comprehensive Python framework for EEG signal processing and neural representation learning. Built around the EEGPT (EEG Pretrained Transformer) architecture and 9 critical reference repositories, it provides a complete pipeline from raw EEG data to production-ready analysis.

## ✨ Key Features

- **🧠 EEGPT Integration**: Pre-trained transformer models for universal EEG representation
- **⚡ Modern Python Stack**: Built with uv, ruff, and 2025 best practices
- **🔧 Complete Pipeline**: From EDF loading to REST API deployment
- **📊 Comprehensive Analysis**: Quality control, sleep staging, feature extraction
- **🎨 Rich CLI**: Beautiful command-line interface with Typer and Rich
- **🚀 Production Ready**: Docker, CI/CD, comprehensive testing
- **📚 9 Reference Repos**: Carefully curated and integrated ML/EEG libraries

## 📁 Repository Structure

```
brain-go-brrr/
├── src/brain_go_brrr/           # Main source code
│   ├── core/                   # Core utilities and configuration
│   ├── models/                 # EEGPT model implementations
│   ├── data/                   # Data processing utilities
│   ├── training/               # Training pipelines
│   ├── inference/              # Inference and serving
│   └── cli.py                  # Command-line interface
├── services/                   # Processing services
│   ├── qc_flagger.py          # Quality control with autoreject + EEGPT
│   ├── sleep_metrics.py       # Sleep analysis with YASA
│   └── snippet_maker.py       # Snippet extraction + tsfresh features
├── reference_repos/            # 9 critical EEG/ML repositories
│   ├── EEGPT/                 # Official EEGPT implementation
│   ├── mne-python/            # MNE for EEG processing
│   ├── pyEDFlib/              # Fast EDF/BDF reading
│   ├── mne-bids/              # BIDS conversion
│   ├── braindecode/           # Deep learning for EEG
│   ├── yasa/                  # Sleep staging
│   ├── tueg-tools/            # TUAB/TUEV dataset tools
│   ├── autoreject/            # Automated artifact rejection
│   └── tsfresh/               # Time-series feature extraction
├── examples/                   # Example scripts and notebooks
├── tests/                      # Comprehensive test suite
├── docs/                       # Documentation
├── .github/workflows/          # CI/CD pipelines
├── pyproject.toml             # Modern Python project configuration
├── Makefile                   # Development commands
└── README.md                  # This file
```

## 🔬 Featured Research

### EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals

- **Paper**: [literature/markdown/EEGPT/EEGPT.md](literature/markdown/EEGPT/EEGPT.md)
- **Implementation**: [reference_repos/EEGPT/](reference_repos/EEGPT/)
- **Key Features**:
  - 10-million-parameter pretrained transformer model
  - Dual self-supervised learning with spatio-temporal representation alignment
  - Hierarchical structure for decoupled spatial and temporal processing
  - State-of-the-art performance on multiple EEG tasks

## 🚀 Key Technologies

- **EEG Signal Processing**: Advanced preprocessing and feature extraction
- **Transformer Architecture**: Self-supervised learning for neural signals
- **Brain-Computer Interface**: Real-time neural signal interpretation
- **Digital Twin**: Comprehensive brain activity modeling

## 🛠️ Getting Started

### Prerequisites

- Python 3.11+
- PyTorch
- NumPy, SciPy
- MNE-Python (for EEG processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
make dev-setup

# Run tests to verify installation
make test
```

## 📊 Research Applications

- **Motor Imagery Classification**: Decoding intended movements from EEG
- **Event-Related Potential (ERP) Detection**: Identifying neural responses to stimuli
- **Sleep Stage Classification**: Automated sleep pattern analysis
- **Brain-Computer Interface**: Real-time neural control systems

## 🔧 Development

This project serves as a foundation for:

- EEG signal processing research
- Neural representation learning
- Brain-computer interface development
- Digital twin neural modeling

## 📄 Literature

The `literature/` directory contains processed research papers with:

- Markdown conversions for easy reading and searching
- Extracted figures and diagrams
- Metadata for research organization

## 🤝 Contributing

This project is part of the CLARITY-DIGITAL-TWIN organization's research initiative. Contributions are welcome for:

- Additional EEG processing methods
- Novel neural architectures
- BCI applications
- Documentation improvements

## 📜 License

This project contains reference materials and implementations with various licenses:

- Research papers: Academic use
- EEGPT implementation: Apache-2.0 License
- Original contributions: To be determined

## 🔗 Related Projects

- [EEGPT Official Repository](https://github.com/BINE022/EEGPT)
- CLARITY-DIGITAL-TWIN Organization Projects

## 📞 Contact

For questions about this research project, please open an issue or contact the CLARITY-DIGITAL-TWIN organization.

---

**Note**: This repository contains research materials and reference implementations. Please refer to individual project licenses and citations when using this work.
