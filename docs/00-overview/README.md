# Brain-Go-Brrr Documentation Hub

## 📚 Documentation Structure

### 00-overview/
Core project documentation and status reports
- [Implementation Status](IMPLEMENTATION_STATUS_DETAILED.md) - Complete implementation details
- [Project Roadmap](PROJECT_ROADMAP.md) - Future development plans

### 01-architecture/
System design and architecture documentation
- [Dual Pipeline Architecture](DUAL_PIPELINE_ARCHITECTURE.md) - Complete dual pipeline design
- [Hierarchical Pipeline](HIERARCHICAL_PIPELINE_DESIGN.md) - Detailed pipeline flow
- [Quality Control System](QUALITY_CONTROL_SYSTEM.md) - QC pipeline architecture
- [Sleep Analysis Integration](SLEEP_ANALYSIS_INTEGRATION.md) - YASA integration design
- [Event Detection](EVENT_DETECTION_ARCHITECTURE.md) - IED detection architecture
- [Abnormality Detection](ABNORMALITY_DETECTION_PIPELINE.md) - Binary classification pipeline

### 02-implementation/
Implementation guides and technical specifications
- [EEGPT Implementation Guide](../EEGPT_IMPLEMENTATION_GUIDE.md) - Model integration details
- [AutoReject Integration](AUTOREJECT_PRECISE_INTEGRATION_SPEC.md) - QC implementation
- [Performance Benchmarks](../PERFORMANCE_BENCHMARKING_GUIDE.md) - Speed and accuracy metrics

### 03-api/
API documentation and design patterns
- [API Design Patterns](../API_DESIGN_PATTERNS.md) - RESTful API design
- [API Reference](API_REFERENCE.md) - Endpoint documentation

### 04-testing/
Testing strategies and specifications
- [Integration Test Scenarios](../INTEGRATION_TEST_SCENARIOS.md) - E2E test cases
- [Test Coverage Guide](../coverage-optimization-guide.md) - Coverage optimization

### 05-deployment/
Deployment and infrastructure documentation
- [Deployment Architecture](../DEPLOYMENT_ARCHITECTURE.md) - K8s and scaling
- [Failure Mode Analysis](../FAILURE_MODE_ANALYSIS.md) - System reliability

### 06-clinical/
Clinical validation and regulatory documentation
- [Clinical Requirements](CLINICAL_REQUIREMENTS.md) - Medical standards
- [Regulatory Compliance](REGULATORY_COMPLIANCE.md) - FDA pathway

## 🎯 Quick Links

### Current Status
- **Production Readiness**: 75%
- **Active Training**: 4-second window EEGPT linear probe
- **Test Coverage**: 63.47% (454 unit tests passing)
- **API Status**: Complete, missing auth only

### Key Components Status

| Component | Status | Production Ready |
|-----------|--------|------------------|
| Quality Control | ✅ 100% | YES |
| EEGPT Integration | ✅ 100% | YES |
| Abnormality Detection | 🟡 80% | Training |
| Sleep Staging (YASA) | ✅ 95% | YES* |
| IED Detection | 🟡 40% | NO |
| API Framework | ✅ 90% | YES |
| Deployment | ❌ 20% | NO |

### Critical Documents
1. [CLAUDE.md](../../CLAUDE.md) - AI assistant instructions
2. [PROJECT_STATUS.md](../../PROJECT_STATUS.md) - Live project status
3. [CHANGELOG.md](../../CHANGELOG.md) - Version history
4. [Training Status](../../experiments/eegpt_linear_probe/TRAINING_STATUS.md) - Current training progress

## 📊 Architecture Overview

```
EEG Input → Quality Control → EEGPT Features → Dual Pipeline:
                                               ├── Abnormality Detection
                                               │   └── (if abnormal) → IED Detection
                                               └── Sleep Staging (parallel)
```

## 🚀 Getting Started

### For Developers
1. Review [Implementation Status](IMPLEMENTATION_STATUS_DETAILED.md)
2. Check [Architecture Documentation](../01-architecture/)
3. See [API Design](../03-api/)

### For Clinical Users
1. Read [Clinical Requirements](../06-clinical/CLINICAL_REQUIREMENTS.md)
2. Review [Performance Metrics](IMPLEMENTATION_STATUS_DETAILED.md#performance-benchmarks)
3. Check [Regulatory Status](../06-clinical/REGULATORY_COMPLIANCE.md)

### For DevOps
1. See [Deployment Architecture](../05-deployment/)
2. Review [Failure Modes](../FAILURE_MODE_ANALYSIS.md)
3. Check [Performance Benchmarks](../PERFORMANCE_BENCHMARKING_GUIDE.md)

## 📈 Key Metrics

- **Processing Speed**: 20-min EEG in <2 minutes
- **Concurrent Capacity**: 50 files
- **API Response**: <100ms (p95)
- **Target AUROC**: ≥0.869 (4s windows)
- **Sleep Staging Accuracy**: 87.46% (YASA)

## 🔄 Recent Updates

- **August 5, 2025**: 4-second window training active
- **August 4, 2025**: Discovered EEGPT requires 4s windows
- **August 3, 2025**: YASA integration complete
- **August 2, 2025**: Hierarchical pipeline implemented
- **July 31, 2025**: v0.5.0 released

---

_Last Updated: August 5, 2025_