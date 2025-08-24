# Brain-Go-Brrr 🧠⚡

**Production-Ready EEG Analysis System with EEGPT Foundation Model**

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

Brain-Go-Brrr provides automated EEG analysis using frozen EEGPT features with specialized task heads for sleep staging, quality control, and abnormality detection.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run API server
uv run uvicorn brain_go_brrr.api.main:app --reload

# Run tests
uv run pytest tests/unit -q
```

📚 **Full documentation**: [docs/README.md](docs/README.md)

## Features

### ✅ Working
- **Sleep Analysis** - 5-stage classification with 87% accuracy (YASA)
- **Quality Control** - Bad channel detection, artifact rejection (Autoreject)
- **EEGPT Features** - 2,048-dim flattened features (4×512 summary tokens)
- **REST API** - FastAPI with Redis caching
- **CI/CD** - GitHub Actions on all branches

### 🟡 In Progress
- **Abnormality Detection** - Training TUAB linear probe (target: 0.87 AUROC)

### ❌ Not Implemented
- Event detection, authentication, production deployment

## Architecture - Parallel Processing Pathways

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

**KEY INSIGHTS**:
- **YASA works with ANY channel count** (not just 2) - it selects the best central channel (C3/C4)
- **Sleep-EDF has 2 channels** but that's dataset-specific, not a YASA requirement
- **Both pipelines run in PARALLEL** and can process the same data
- **EEGPT requires 19+ channels** for meaningful clinical results
- **YASA achieves 87% accuracy** with just 1 central EEG channel

Clean Architecture with dependency injection and adapter pattern for third-party libraries.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/QUICK_START.md](docs/QUICK_START.md) | Get running in 5 minutes |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design |
| [docs/API.md](docs/API.md) | REST endpoints |
| [docs/TRAINING.md](docs/TRAINING.md) | Model training |
| [docs/TESTING.md](docs/TESTING.md) | Test guidelines |

## Performance

- Process 20-minute EEG in <2 minutes
- 454 passing tests
- <100ms API response (cached)
- 87% sleep staging accuracy

## Requirements

- Python 3.11 or 3.12
- 16GB RAM minimum
- GPU optional (for training)
- WSL2 (Windows users)

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
make check-all

# Watch tests
make test-watch
```

## Training

```bash
# Train TUAB abnormality detection
cd experiments/eegpt_linear_probe
./scripts/launch_tuab.sh

# Monitor progress
tmux attach -t tuab_training
```

## Contributing

1. Read [ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. Follow [TESTING.md](docs/TESTING.md)
3. All PRs require passing CI/CD

## Citations

If you use this software in your research, please cite:

### EEGPT Model
```bibtex
@inproceedings{wang2024eegpt,
  title={EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals},
  author={Wang, Guangyu and He, Yuhong and Ma, Lin and Liu, Wenchao and Xu, Cong and Li, Haifeng},
  booktitle={38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024},
  url={https://github.com/BINE022/EEGPT}
}
```

### YASA Sleep Staging
```bibtex
@article{vallat2021yasa,
  title={YASA: Yet Another Spindle Algorithm},
  author={Vallat, Raphael and Walker, Matthew P},
  journal={bioRxiv},
  year={2021},
  doi={10.1101/2021.05.28.446165}
}
```

### Autoreject
```bibtex
@article{jas2017autoreject,
  title={Autoreject: Automated artifact rejection for MEG and EEG data},
  author={Jas, Mainak and Engemann, Denis A and Bekhti, Yousra and Raimondo, Federico and Gramfort, Alexandre},
  journal={NeuroImage},
  volume={159},
  pages={417--429},
  year={2017},
  doi={10.1016/j.neuroimage.2017.06.030}
}
```

## Datasets

### Dataset Sources (NOT Included)
**Important**: Datasets are NOT included in this repository due to size and licensing. You must obtain them separately:

- **Temple University Hospital EEG Corpus**: [isip.piconepress.com/projects/nedc/html/tuh_eeg](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
  - TUAB (Abnormal EEG Corpus) - Binary classification (120GB compressed)
  - TUEV (EEG Events) - Event detection (60GB compressed)
  - Requires academic agreement and registration

- **PhysioNet Sleep-EDF**: [physionet.org/content/sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
  - 197 whole-night PSG recordings
  - Free download after PhysioNet registration
  - Place in: `data/datasets/external/sleep-edf/`

See [docs/TRAINING.md](docs/TRAINING.md) for detailed download and setup instructions.

## Model Weights

### EEGPT Pretrained Model
- **Official Repository**: [github.com/BINE022/EEGPT](https://github.com/BINE022/EEGPT)
- **Paper**: "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals" (NeurIPS 2024)

Download the pretrained weights:
- **Figshare**: [EEGPT Large Model](https://figshare.com/s/e37df4f8a907a866df4b)
  - Navigate to: `Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
  - Place in: `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
  - Size: ~40MB
  - Architecture: 10M parameters, 58 channels, 256Hz, 4s windows

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/Clarity-Digital-Twin/brain-go-brrr/issues)
- Documentation: [docs/](docs/)

---

*For AI assistants: See [CLAUDE.md](CLAUDE.md) for context and guidelines*
