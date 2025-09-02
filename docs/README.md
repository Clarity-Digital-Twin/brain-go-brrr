# Brain-Go-Brrr Documentation

## Quick Navigation

| Document | Purpose | Status |
|----------|---------|--------|
| [QUICK_START.md](QUICK_START.md) | Get running in 5 minutes | ✅ Current |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and components | ✅ Current |
| [API.md](API.md) | REST API endpoints | ✅ Current |
| [TRAINING.md](TRAINING.md) | Model training guide | ✅ Current |
| [TESTING.md](TESTING.md) | Testing guidelines | ✅ Current |
| [CI_CD_SETUP.md](CI_CD_SETUP.md) | CI/CD and pre-commit setup | ✅ Current |
| [TUH_CORPUS_GUIDE.md](TUH_CORPUS_GUIDE.md) | TUH dataset download guide | ✅ Current |

## Project Status

### ✅ Working Components
- **YASA Sleep Analysis** - 5-stage classification, 87% accuracy
- **Autoreject QC** - Bad channel detection, artifact rejection
- **EEGPT Features** - 2,048-dim flattened features (4×512 summary tokens)
- **FastAPI Server** - REST API with Redis caching
- **CI/CD Pipeline** - GitHub Actions on all branches

### 🟡 In Progress
- **TUAB Abnormality Detection** - Linear probe training (AUROC improving toward 87% target)
- **Documentation Cleanup** - Actively auditing and updating all docs

### ❌ Not Implemented
- **Event Detection** - Seizure/IED detection
- **Authentication** - OAuth2/JWT
- **Production Deployment** - Kubernetes, PostgreSQL
- **Message Queue** - Celery async processing

## Getting Started

1. **New Users**: Start with [QUICK_START.md](QUICK_START.md)
2. **Developers**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
3. **API Users**: See [API.md](API.md)
4. **ML Engineers**: Check [TRAINING.md](TRAINING.md)

## System Overview - Parallel Processing Pathways

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
    (256Hz sampling)              (Auto-selects best channel)
          │                               │
          ▼                               ▼
    EEGPT Features                 Sleep Staging
    (4×512 summary tokens →        (85%+ accuracy with 1 ch)
     2,048 flattened)
          │                               │
          ▼                               ▼
    Abnormality Detection           Sleep Metrics
    (Normal vs Abnormal)            (Efficiency, TST, etc.)
          │
      [IF ABNORMAL]
          │
          ▼
    Event Detection
    (TUEV: SPSW/GPED/PLED/etc)
```

**Key Insights**:
- EEGPT and YASA are PARALLEL pathways that can process the same data
- YASA works with ANY channel count (not limited to 2) - it intelligently selects the best central channel
- Sleep-EDF's 2 channels are dataset-specific, not a YASA limitation

## Key Technologies

- **Models**: EEGPT (10M params), YASA, Autoreject
- **Backend**: FastAPI, PyTorch, MNE-Python
- **Infrastructure**: Redis, Docker, GitHub Actions
- **Languages**: Python 3.11+

## Performance Metrics

| Component | Performance | Status |
|-----------|------------|--------|
| Sleep Staging | 87% accuracy | ✅ Production |
| QC Agreement | 87% with experts | ✅ Production |
| API Response | <100ms cached | ✅ Production |
| TUAB Target | 0.87 AUROC | 🟡 Training |

## Repository Structure

```
brain-go-brrr/
├── src/brain_go_brrr/    # Main package
│   ├── domain/           # Business logic
│   ├── application/      # Use cases
│   ├── infra/           # External services
│   └── api/             # REST API
├── experiments/         # Training scripts
├── tests/              # Test suite
└── docs/               # Documentation (you are here)
```

## Contributing

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for design principles
2. Follow [TESTING.md](TESTING.md) for test requirements
3. Use pre-commit hooks: `pre-commit install`
4. All PRs require passing CI/CD

## Support

- **Issues**: [GitHub Issues](https://github.com/Clarity-Digital-Twin/brain-go-brrr/issues)
- **Discussions**: Use issue comments
- **Security**: Report to security@clarity.ai

## Archive Notice

Previous documentation (130+ files) has been archived in `docs/archive/`. The current documentation set is intentionally minimal and focused on actual implemented features.

## License

Apache 2.0 - See [LICENSE](../LICENSE)

---

_Last Updated: September 2, 2025_
_Documentation Version: 2.1 (Post-Audit)_
