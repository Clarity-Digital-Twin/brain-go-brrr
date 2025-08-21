# Brain-Go-Brrr Documentation Hub

## 📚 Documentation Structure

### Current Implementation Status (Aug 2025)

Brain-Go-Brrr is a research-grade EEG analysis platform integrating the EEGPT foundation model with established neuroscience tools. The project follows Clean Architecture principles with Domain-Driven Design.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRESENTATION                            │
│                    (API Endpoints, CLI)                         │
├─────────────────────────────────────────────────────────────────┤
│                         APPLICATION                             │
│              (Use Cases, Factories, Orchestration)              │
├─────────────────────────────────────────────────────────────────┤
│                           DOMAIN                                │
│            (Business Logic, Ports, Domain Services)             │
│                     ZERO DEPENDENCIES                           │
└─────────────────────────────────────────────────────────────────┘
        ↑                       ↑                        ↑
┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ INFRASTRUCTURE│    │  INFRASTRUCTURE  │    │  INFRASTRUCTURE  │
│   (ML Models) │    │    (External)    │    │   (Data Access)  │
└───────────────┘    └──────────────────┘    └──────────────────┘
```

## ✅ What's Actually Working

### Production-Ready Components

| Component | Status | Notes |
|-----------|--------|-------|
| **EEGPT Integration** | ✅ 100% | 10M parameter model, 512-dim embeddings |
| **Sleep Staging (YASA)** | ✅ 100% | 87% accuracy on Sleep-EDF |
| **Quality Control (Autoreject)** | ✅ 100% | 87% expert agreement |
| **FastAPI Framework** | ✅ 100% | Full REST API with file uploads |
| **Redis Caching** | ✅ 100% | Circuit breaker pattern |
| **CI/CD Pipeline** | ✅ 100% | All green on main/staging/development |

### In Development

| Component | Status | Notes |
|-----------|--------|-------|
| **Abnormality Detection** | 🟡 Training | Linear probe on TUAB dataset |
| **Event Detection** | 🔴 Planned | Architecture designed, not implemented |
| **Real-time Streaming** | 🟡 Prototype | CLI works, not production-ready |
| **Authentication** | 🔴 Not Started | JWT/OAuth2 planned |

## 📁 Documentation Categories

### [00-overview/](.)
Project status and high-level documentation
- **README.md** (this file) - Documentation hub
- **[IMPLEMENTATION_STATUS_DETAILED.md](IMPLEMENTATION_STATUS_DETAILED.md)** - Component status

### [01-architecture/](../01-architecture/)
System design documentation
- **[CLEAN_ARCHITECTURE.md](../CLEAN_ARCHITECTURE.md)** - DDD/Clean Architecture implementation
- **[QUALITY_CONTROL_SYSTEM.md](../01-architecture/QUALITY_CONTROL_SYSTEM.md)** - Autoreject integration
- **[SLEEP_ANALYSIS_INTEGRATION.md](../01-architecture/SLEEP_ANALYSIS_INTEGRATION.md)** - YASA integration

### [02-implementation/](../02-implementation/)
Technical implementation guides
- **[EEGPT_IMPLEMENTATION_GUIDE.md](../02-implementation/EEGPT_IMPLEMENTATION_GUIDE.md)** - Model integration
- **[DOCKER_QUICKSTART.md](../02-implementation/DOCKER_QUICKSTART.md)** - Container deployment
- **[GITHUB_ACTIONS_CLAUDE_CODE.md](../02-implementation/GITHUB_ACTIONS_CLAUDE_CODE.md)** - CI/CD with Claude

### [03-api/](../03-api/)
API documentation
- **API_REFERENCE.md** - Endpoint documentation
- **API_DESIGN_PATTERNS.md** - RESTful patterns

### [04-testing/](../04-testing/)
Testing documentation
- **INTEGRATION_TEST_SCENARIOS.md** - E2E test cases
- **TEST_COVERAGE_GUIDE.md** - Coverage optimization

### [05-deployment/](../05-deployment/)
Deployment guides
- **DEPLOYMENT_ARCHITECTURE.md** - Infrastructure design
- **FAILURE_MODE_ANALYSIS.md** - Reliability analysis

### [06-clinical/](../06-clinical/)
Clinical and regulatory (aspirational)
- **CLINICAL_REQUIREMENTS.md** - Medical standards
- **Note**: We are NOT pursuing FDA approval currently

## 📊 Key Metrics

### Achieved Performance

| Metric | Value | Verified |
|--------|-------|----------|
| **Sleep Staging Accuracy** | 87% | ✅ Sleep-EDF dataset |
| **Bad Channel Detection** | 87% | ✅ Expert agreement |
| **Processing Speed** | <2 min | ✅ 20-min recording |
| **API Latency** | <100ms | ✅ Health endpoint |
| **Test Coverage** | ~66% | ✅ 800+ tests |
| **CI/CD Status** | 100% Green | ✅ All branches |

### System Requirements

- **Python**: 3.11 or 3.12 (3.13 not supported due to scipy)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (CUDA for faster inference)
- **Storage**: ~5GB for models and dependencies

## 🚀 Quick Start Paths

### For Developers
```bash
# Clone and setup
git clone https://github.com/Clarity-Digital-Twin/brain-go-brrr.git
cd brain-go-brrr
uv sync --all-extras

# Run tests
uv run pytest tests/smoke -q

# Start API
uv run python -m brain_go_brrr.api
```

### For Researchers
1. Review [EEGPT Implementation](../02-implementation/EEGPT_IMPLEMENTATION_GUIDE.md)
2. Check [Training Scripts](../../experiments/eegpt_linear_probe/)
3. See [Literature References](../../literature/markdown/)

### For DevOps
1. See [Docker Quickstart](../02-implementation/DOCKER_QUICKSTART.md)
2. Review [CI/CD Workflows](../../.github/workflows/)
3. Check [Deployment Architecture](../05-deployment/)

## 📝 Critical Project Files

- **[CLAUDE.md](../../CLAUDE.md)** - AI assistant instructions & project context
- **[TECHNICAL_DEBT.md](../../TECHNICAL_DEBT.md)** - Known limitations & roadmap
- **[README.md](../../README.md)** - Public-facing project overview

## ⚠️ Important Notes

1. **Research Software**: This is NOT clinical software. No FDA approval pursued.
2. **Model Requirements**: EEGPT needs exact 4-second windows at 256Hz
3. **Data Limitations**: Sleep-EDF is at 100Hz, requires resampling
4. **Memory Requirements**: Autoreject needs 2+ minutes of data

## 🔗 External Resources

- [EEGPT Paper (NeurIPS 2024)](https://github.com/BINE022/EEGPT)
- [YASA Documentation](https://raphaelvallat.com/yasa/build/html/index.html)
- [Autoreject Documentation](https://autoreject.github.io/)
- [MNE-Python](https://mne.tools/stable/index.html)

---

*Last Updated: August 21, 2025*
*Status: Production components working, research features in development*
