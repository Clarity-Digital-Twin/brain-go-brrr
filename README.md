# Brain-Go-Brrr 🧠⚡

**Production-Ready EEG Analysis System with EEGPT Foundation Model**

[![CI/CD Pipeline](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Clarity-Digital-Twin/brain-go-brrr/actions/workflows/ci.yml)
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
- **EEGPT Features** - 512-dimensional embeddings from frozen backbone
- **REST API** - FastAPI with Redis caching
- **CI/CD** - GitHub Actions on all branches

### 🟡 In Progress
- **Abnormality Detection** - Training TUAB linear probe (target: 0.87 AUROC)

### ❌ Not Implemented
- Event detection, authentication, production deployment

## Architecture

```
EEG Input → Quality Control → EEGPT Features → Task Heads
                                               ├── Sleep Staging
                                               └── Abnormality Detection
```

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

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/Clarity-Digital-Twin/brain-go-brrr/issues)
- Documentation: [docs/](docs/)

---

*For AI assistants: See [CLAUDE.md](CLAUDE.md) for context and guidelines*
